#include "jit_core.h"
#include "opcodes.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <sstream>

// Python code object flags - define if not available
// CO_ITERABLE_COROUTINE marks generators that can be used in await expressions
// (i.e., decorated with @types.coroutine)
#ifndef CO_ITERABLE_COROUTINE
#define CO_ITERABLE_COROUTINE 0x0100
#endif

// C helper function for NULL-safe Py_XINCREF (since Py_XINCREF is a macro)
extern "C" void jit_xincref(PyObject *obj)
{
    Py_XINCREF(obj);
}

// C helper function for CALL_KW opcode
// Splits args array into positional tuple and kwargs dict based on kwnames tuple
extern "C" PyObject *jit_call_with_kwargs(
    PyObject *callable,
    PyObject **args,
    Py_ssize_t nargs,
    PyObject *kwnames)
{
    // Get number of keyword arguments from kwnames tuple
    Py_ssize_t nkwargs = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    Py_ssize_t npos = nargs - nkwargs;

    if (npos < 0)
    {
        PyErr_SetString(PyExc_SystemError, "jit_call_with_kwargs: nkwargs > nargs");
        return NULL;
    }

    // Build positional args tuple
    PyObject *pos_tuple = PyTuple_New(npos);
    if (!pos_tuple)
        return NULL;

    for (Py_ssize_t i = 0; i < npos; i++)
    {
        PyObject *arg = args[i];
        Py_INCREF(arg);
        PyTuple_SET_ITEM(pos_tuple, i, arg);
    }

    // Build kwargs dict
    PyObject *kwargs_dict = NULL;
    if (nkwargs > 0)
    {
        kwargs_dict = PyDict_New();
        if (!kwargs_dict)
        {
            Py_DECREF(pos_tuple);
            return NULL;
        }

        for (Py_ssize_t i = 0; i < nkwargs; i++)
        {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = args[npos + i];
            if (PyDict_SetItem(kwargs_dict, key, value) < 0)
            {
                Py_DECREF(pos_tuple);
                Py_DECREF(kwargs_dict);
                return NULL;
            }
        }
    }

    // Call the callable
    PyObject *result = PyObject_Call(callable, pos_tuple, kwargs_dict);

    Py_DECREF(pos_tuple);
    Py_XDECREF(kwargs_dict);

    return result;
}

// C helper function for GET_AWAITABLE opcode
// Gets an awaitable from an object:
// - If it's a coroutine, return it directly
// - If it's a generator (from types.coroutine decorator), return it
// - Otherwise, call __await__ and return the iterator
extern "C" PyObject *_PyJIT_GetAwaitable(PyObject *obj)
{
    // Check if it's a native coroutine
    if (PyCoro_CheckExact(obj)) {
        Py_INCREF(obj);
        return obj;
    }
    
    // Check if it's a generator with CO_ITERABLE_COROUTINE flag
    // (decorated with @types.coroutine)
    if (PyGen_CheckExact(obj)) {
        // In Python 3.13+, use PyGen_GetCode() instead of accessing gi_code directly
        PyCodeObject *code = PyGen_GetCode((PyGenObject *)obj);
        if (code != NULL) {
            int flags = code->co_flags;
            Py_DECREF(code);  // PyGen_GetCode returns a new reference
            if (flags & CO_ITERABLE_COROUTINE) {
                Py_INCREF(obj);
                return obj;
            }
        }
    }
    
    // Try to get __await__ method
    PyObject *await_method = PyObject_GetAttrString(obj, "__await__");
    if (await_method == NULL) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError,
                "object %.100s can't be used in 'await' expression",
                Py_TYPE(obj)->tp_name);
        }
        return NULL;
    }
    
    // Call __await__()
    PyObject *result = PyObject_CallNoArgs(await_method);
    Py_DECREF(await_method);
    
    if (result == NULL) {
        return NULL;
    }
    
    // Verify the result is an iterator
    if (!PyIter_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "__await__() returned non-iterator of type '%.100s'",
            Py_TYPE(result)->tp_name);
        Py_DECREF(result);
        return NULL;
    }
    
    return result;
}

// C helper function for MATCH_KEYS opcode
// Extracts values from a mapping for the given keys tuple
// Returns a tuple of values if all keys found, Py_None (incref'd) otherwise
extern "C" PyObject *_PyJIT_MatchKeys(PyObject *subject, PyObject *keys)
{
    if (!PyTuple_Check(keys)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    Py_ssize_t nkeys = PyTuple_GET_SIZE(keys);
    PyObject *values = PyTuple_New(nkeys);
    if (values == NULL) {
        PyErr_Clear();
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    for (Py_ssize_t i = 0; i < nkeys; i++) {
        PyObject *key = PyTuple_GET_ITEM(keys, i);
        PyObject *value = PyObject_GetItem(subject, key);
        
        if (value == NULL) {
            // Key not found - clean up and return Py_None
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                PyErr_Clear();
            }
            Py_DECREF(values);
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        PyTuple_SET_ITEM(values, i, value);  // Steals reference
    }
    
    return values;
}

// C helper function for MATCH_CLASS opcode
// Matches a subject against a class pattern and extracts attributes
// nargs = number of positional patterns (for __match_args__)
// names = tuple of keyword attribute names
// Returns tuple of matched attributes if successful, Py_None (incref'd) otherwise
extern "C" PyObject *_PyJIT_MatchClass(PyObject *subject, PyObject *cls, int nargs, PyObject *names)
{
    // First, check that subject is an instance of cls
    int is_instance = PyObject_IsInstance(subject, cls);
    if (is_instance < 0) {
        PyErr_Clear();
        Py_INCREF(Py_None);
        return Py_None;  // Error during isinstance check
    }
    if (!is_instance) {
        Py_INCREF(Py_None);
        return Py_None;  // Not an instance
    }
    
    // Get __match_args__ from the class if we have positional patterns
    PyObject *match_args = NULL;
    if (nargs > 0) {
        match_args = PyObject_GetAttrString(cls, "__match_args__");
        if (match_args == NULL) {
            if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
                // No __match_args__ and nargs > 0 means no match for positional patterns
                // Unless nargs == 0, in which case we're fine
                if (nargs > 0) {
                    Py_INCREF(Py_None);
                    return Py_None;
                }
            } else {
                PyErr_Clear();
                Py_INCREF(Py_None);
                return Py_None;  // Other error
            }
        }
    }
    
    // Calculate total number of attributes to extract
    Py_ssize_t nkwargs = names ? PyTuple_GET_SIZE(names) : 0;
    Py_ssize_t total = nargs + nkwargs;
    
    PyObject *attrs = PyTuple_New(total);
    if (attrs == NULL) {
        PyErr_Clear();
        Py_XDECREF(match_args);
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    // Extract positional attributes from __match_args__
    for (int i = 0; i < nargs; i++) {
        if (match_args == NULL || !PyTuple_Check(match_args) || 
            i >= PyTuple_GET_SIZE(match_args)) {
            Py_DECREF(attrs);
            Py_XDECREF(match_args);
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        PyObject *attr_name = PyTuple_GET_ITEM(match_args, i);
        if (!PyUnicode_Check(attr_name)) {
            Py_DECREF(attrs);
            Py_XDECREF(match_args);
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        PyObject *attr_value = PyObject_GetAttr(subject, attr_name);
        if (attr_value == NULL) {
            if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
            }
            Py_DECREF(attrs);
            Py_XDECREF(match_args);
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        PyTuple_SET_ITEM(attrs, i, attr_value);  // Steals reference
    }
    
    Py_XDECREF(match_args);
    
    // Extract keyword attributes
    for (Py_ssize_t i = 0; i < nkwargs; i++) {
        PyObject *attr_name = PyTuple_GET_ITEM(names, i);
        PyObject *attr_value = PyObject_GetAttr(subject, attr_name);
        if (attr_value == NULL) {
            if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
            }
            Py_DECREF(attrs);
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        PyTuple_SET_ITEM(attrs, nargs + i, attr_value);  // Steals reference
    }
    
    return attrs;
}

namespace justjit
{

    JITCore::JITCore()
    {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        auto jit_builder = llvm::orc::LLJITBuilder();
        auto jit_result = jit_builder.create();

        if (!jit_result)
        {
            llvm::errs() << "Failed to create LLJIT: " << toString(jit_result.takeError()) << "\n";
            return;
        }

        jit = std::move(*jit_result);
        context = std::make_unique<llvm::LLVMContext>();

        // Register our C helper functions with the JIT as absolute symbols
        // This makes them available for the JIT-compiled code to call
        llvm::orc::SymbolMap helper_symbols;

        // Register jit_call_with_kwargs helper
        auto &es = jit->getExecutionSession();
        auto &jd = jit->getMainJITDylib();

        helper_symbols[es.intern("jit_call_with_kwargs")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_with_kwargs),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register jit_xincref helper (NULL-safe Py_XINCREF)
        helper_symbols[es.intern("jit_xincref")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_xincref),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register _PyJIT_GetAwaitable helper for GET_AWAITABLE opcode
        helper_symbols[es.intern("_PyJIT_GetAwaitable")] = {
            llvm::orc::ExecutorAddr::fromPtr(_PyJIT_GetAwaitable),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register _PyJIT_MatchKeys helper for MATCH_KEYS opcode
        helper_symbols[es.intern("_PyJIT_MatchKeys")] = {
            llvm::orc::ExecutorAddr::fromPtr(_PyJIT_MatchKeys),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register _PyJIT_MatchClass helper for MATCH_CLASS opcode
        helper_symbols[es.intern("_PyJIT_MatchClass")] = {
            llvm::orc::ExecutorAddr::fromPtr(_PyJIT_MatchClass),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        auto err = jd.define(llvm::orc::absoluteSymbols(helper_symbols));
        if (err)
        {
            llvm::errs() << "Failed to define helper symbols: " << toString(std::move(err)) << "\n";
        }
    }

    JITCore::~JITCore()
    {
        // Release all stored Python object references
        for (PyObject *obj : stored_constants)
        {
            if (obj != nullptr)
            {
                Py_DECREF(obj);
            }
        }
        for (PyObject *obj : stored_names)
        {
            if (obj != nullptr)
            {
                Py_DECREF(obj);
            }
        }
        for (PyObject *obj : stored_closure_cells)
        {
            if (obj != nullptr)
            {
                Py_DECREF(obj);
            }
        }
        // Release globals and builtins dicts
        if (globals_dict_ptr != nullptr)
        {
            Py_DECREF(globals_dict_ptr);
        }
        if (builtins_dict_ptr != nullptr)
        {
            Py_DECREF(builtins_dict_ptr);
        }
        stored_constants.clear();
        stored_names.clear();
        stored_closure_cells.clear();
    }

    void JITCore::set_opt_level(int level)
    {
        opt_level = std::min(std::max(level, 0), 3);
    }

    int JITCore::get_opt_level() const
    {
        return opt_level;
    }

    void JITCore::set_dump_ir(bool dump)
    {
        dump_ir = dump;
    }

    bool JITCore::get_dump_ir() const
    {
        return dump_ir;
    }

    std::string JITCore::get_last_ir() const
    {
        return last_ir;
    }

    nb::object JITCore::get_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (func_ptr == 0)
        {
            return nb::none();
        }

        switch (param_count)
        {
        case 0:
            return create_callable_0(func_ptr);
        case 1:
            return create_callable_1(func_ptr);
        case 2:
            return create_callable_2(func_ptr);
        case 3:
            return create_callable_3(func_ptr);
        case 4:
            return create_callable_4(func_ptr);
        default:
            return nb::none();
        }
    }

    void JITCore::declare_python_api_functions(llvm::Module *module, llvm::IRBuilder<> *builder)
    {
        llvm::Type *ptr_type = builder->getPtrTy();
        llvm::Type *i64_type = builder->getInt64Ty();
        llvm::Type *void_type = builder->getVoidTy();

        // PyObject* PyList_New(Py_ssize_t len)
        llvm::FunctionType *list_new_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
        py_list_new_func = llvm::Function::Create(list_new_type, llvm::Function::ExternalLinkage, "PyList_New", module);

        // int PyList_SetItem(PyObject* list, Py_ssize_t index, PyObject* item)
        // Returns 0 on success, -1 on failure - steals reference to item
        llvm::FunctionType *list_setitem_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, i64_type, ptr_type}, false);
        py_list_setitem_func = llvm::Function::Create(list_setitem_type, llvm::Function::ExternalLinkage, "PyList_SetItem", module);

        // PyObject* PyObject_GetItem(PyObject* o, PyObject* key)
        // Returns new reference or NULL on error
        llvm::FunctionType *object_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_object_getitem_func = llvm::Function::Create(object_getitem_type, llvm::Function::ExternalLinkage, "PyObject_GetItem", module);

        // void Py_IncRef(PyObject* o)
        llvm::FunctionType *incref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
        py_incref_func = llvm::Function::Create(incref_type, llvm::Function::ExternalLinkage, "Py_IncRef", module);

        // void jit_xincref(PyObject* o) - our NULL-safe wrapper for Py_XINCREF
        llvm::FunctionType *xincref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
        py_xincref_func = llvm::Function::Create(xincref_type, llvm::Function::ExternalLinkage, "jit_xincref", module);

        // void Py_DecRef(PyObject* o)
        llvm::FunctionType *decref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
        py_decref_func = llvm::Function::Create(decref_type, llvm::Function::ExternalLinkage, "Py_DecRef", module);

        // PyObject* PyLong_FromLong(long value)
        llvm::FunctionType *long_fromlong_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
        py_long_fromlong_func = llvm::Function::Create(long_fromlong_type, llvm::Function::ExternalLinkage, "PyLong_FromLong", module);

        // PyObject* PyLong_FromLongLong(long long value) - for proper 64-bit support on Windows
        llvm::FunctionType *long_fromlonglong_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
        py_long_fromlonglong_func = llvm::Function::Create(long_fromlonglong_type, llvm::Function::ExternalLinkage, "PyLong_FromLongLong", module);

        // PyObject* PyTuple_New(Py_ssize_t len)
        llvm::FunctionType *tuple_new_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
        py_tuple_new_func = llvm::Function::Create(tuple_new_type, llvm::Function::ExternalLinkage, "PyTuple_New", module);

        // void PyTuple_SetItem(PyObject* tuple, Py_ssize_t index, PyObject* item)
        // Steals reference to item, unlike PyList_SetItem which returns int
        llvm::FunctionType *tuple_setitem_type = llvm::FunctionType::get(
            void_type, {ptr_type, i64_type, ptr_type}, false);
        py_tuple_setitem_func = llvm::Function::Create(tuple_setitem_type, llvm::Function::ExternalLinkage, "PyTuple_SetItem", module);

        // PyObject* PyNumber_Add(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_add_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_add_func = llvm::Function::Create(number_add_type, llvm::Function::ExternalLinkage, "PyNumber_Add", module);

        // PyObject* PyNumber_Subtract(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_subtract_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_subtract_func = llvm::Function::Create(number_subtract_type, llvm::Function::ExternalLinkage, "PyNumber_Subtract", module);

        // PyObject* PyNumber_Multiply(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_multiply_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_multiply_func = llvm::Function::Create(number_multiply_type, llvm::Function::ExternalLinkage, "PyNumber_Multiply", module);

        // PyObject* PyNumber_MatrixMultiply(PyObject* o1, PyObject* o2) - for @ operator
        llvm::FunctionType *number_matrixmultiply_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_matrixmultiply_func = llvm::Function::Create(number_matrixmultiply_type, llvm::Function::ExternalLinkage, "PyNumber_MatrixMultiply", module);

        // PyObject* PyNumber_TrueDivide(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_truedivide_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_truedivide_func = llvm::Function::Create(number_truedivide_type, llvm::Function::ExternalLinkage, "PyNumber_TrueDivide", module);

        // PyObject* PyNumber_FloorDivide(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_floordivide_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_floordivide_func = llvm::Function::Create(number_floordivide_type, llvm::Function::ExternalLinkage, "PyNumber_FloorDivide", module);

        // PyObject* PyNumber_Remainder(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_remainder_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_remainder_func = llvm::Function::Create(number_remainder_type, llvm::Function::ExternalLinkage, "PyNumber_Remainder", module);

        // PyObject* PyNumber_Power(PyObject* o1, PyObject* o2, PyObject* o3)
        // o3 is for modular exponentiation, pass Py_None to ignore
        llvm::FunctionType *number_power_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type, ptr_type}, false);
        py_number_power_func = llvm::Function::Create(number_power_type, llvm::Function::ExternalLinkage, "PyNumber_Power", module);

        // PyObject* PyNumber_Negative(PyObject* o)
        llvm::FunctionType *number_negative_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_number_negative_func = llvm::Function::Create(number_negative_type, llvm::Function::ExternalLinkage, "PyNumber_Negative", module);

        // PyObject* PyNumber_Positive(PyObject* o) - implements unary + operator
        llvm::FunctionType *number_positive_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_number_positive_func = llvm::Function::Create(number_positive_type, llvm::Function::ExternalLinkage, "PyNumber_Positive", module);

        // PyObject* PyObject_Str(PyObject* o)
        llvm::FunctionType *object_str_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_object_str_func = llvm::Function::Create(object_str_type, llvm::Function::ExternalLinkage, "PyObject_Str", module);

        // PyObject* PyUnicode_Concat(PyObject* left, PyObject* right)
        llvm::FunctionType *unicode_concat_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_unicode_concat_func = llvm::Function::Create(unicode_concat_type, llvm::Function::ExternalLinkage, "PyUnicode_Concat", module);

        // PyObject* PyObject_GetAttr(PyObject* o, PyObject* attr_name)
        llvm::FunctionType *getattr_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_object_getattr_func = llvm::Function::Create(getattr_type, llvm::Function::ExternalLinkage, "PyObject_GetAttr", module);

        // int PyObject_SetAttr(PyObject* o, PyObject* attr_name, PyObject* value)
        llvm::FunctionType *setattr_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
        py_object_setattr_func = llvm::Function::Create(setattr_type, llvm::Function::ExternalLinkage, "PyObject_SetAttr", module);

        // int PyObject_SetItem(PyObject* o, PyObject* key, PyObject* value)
        llvm::FunctionType *object_setitem_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
        py_object_setitem_func = llvm::Function::Create(object_setitem_type, llvm::Function::ExternalLinkage, "PyObject_SetItem", module);

        // PyObject* PyObject_Call(PyObject* callable, PyObject* args, PyObject* kwargs)
        // args must be tuple, kwargs can be NULL (nullptr)
        llvm::FunctionType *object_call_type = llvm::FunctionType::get(
            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
        py_object_call_func = llvm::Function::Create(object_call_type, llvm::Function::ExternalLinkage, "PyObject_Call", module);

        // long PyLong_AsLong(PyObject* obj) - for unboxing
        llvm::FunctionType *long_aslong_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
        py_long_aslong_func = llvm::Function::Create(long_aslong_type, llvm::Function::ExternalLinkage, "PyLong_AsLong", module);

        // int PyObject_RichCompareBool(PyObject* o1, PyObject* o2, int opid)
        // opid: Py_LT=0, Py_LE=1, Py_EQ=2, Py_NE=3, Py_GT=4, Py_GE=5
        // Returns -1 on error, 0 if false, 1 if true
        llvm::FunctionType *richcompare_bool_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type, builder->getInt32Ty()}, false);
        py_object_richcompare_bool_func = llvm::Function::Create(richcompare_bool_type, llvm::Function::ExternalLinkage, "PyObject_RichCompareBool", module);

        // int PyObject_IsTrue(PyObject* o)
        // Returns 1 if true, 0 if false, -1 on error
        llvm::FunctionType *istrue_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type}, false);
        py_object_istrue_func = llvm::Function::Create(istrue_type, llvm::Function::ExternalLinkage, "PyObject_IsTrue", module);

        // int PyObject_IsInstance(PyObject* obj, PyObject* cls)
        // Returns 1 if instance, 0 if not, -1 on error
        llvm::FunctionType *isinstance_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_object_isinstance_func = llvm::Function::Create(isinstance_type, llvm::Function::ExternalLinkage, "PyObject_IsInstance", module);

        // PyObject* PyNumber_Invert(PyObject* o) - bitwise NOT (~)
        llvm::FunctionType *number_invert_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_number_invert_func = llvm::Function::Create(number_invert_type, llvm::Function::ExternalLinkage, "PyNumber_Invert", module);

        // int PyObject_Not(PyObject* o) - logical NOT
        // Returns 0 if true, 1 if false, -1 on error
        llvm::FunctionType *object_not_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type}, false);
        py_object_not_func = llvm::Function::Create(object_not_type, llvm::Function::ExternalLinkage, "PyObject_Not", module);

        // PyObject* PyObject_GetIter(PyObject* o) - get iterator
        llvm::FunctionType *getiter_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_object_getiter_func = llvm::Function::Create(getiter_type, llvm::Function::ExternalLinkage, "PyObject_GetIter", module);

        // PyObject* PyIter_Next(PyObject* o) - get next item from iterator
        // Returns NULL when exhausted (no exception) or on error (exception set)
        llvm::FunctionType *iter_next_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_iter_next_func = llvm::Function::Create(iter_next_type, llvm::Function::ExternalLinkage, "PyIter_Next", module);

        // PyObject* PyDict_New() - create new empty dict
        llvm::FunctionType *dict_new_type = llvm::FunctionType::get(ptr_type, {}, false);
        py_dict_new_func = llvm::Function::Create(dict_new_type, llvm::Function::ExternalLinkage, "PyDict_New", module);

        // int PyDict_SetItem(PyObject* p, PyObject* key, PyObject* val)
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *dict_setitem_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
        py_dict_setitem_func = llvm::Function::Create(dict_setitem_type, llvm::Function::ExternalLinkage, "PyDict_SetItem", module);

        // PyObject* PySet_New(PyObject* iterable) - create new set (iterable can be NULL for empty)
        llvm::FunctionType *set_new_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_set_new_func = llvm::Function::Create(set_new_type, llvm::Function::ExternalLinkage, "PySet_New", module);

        // int PySet_Add(PyObject* set, PyObject* key)
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *set_add_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_set_add_func = llvm::Function::Create(set_add_type, llvm::Function::ExternalLinkage, "PySet_Add", module);

        // int PyList_Append(PyObject* list, PyObject* item)
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *list_append_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_list_append_func = llvm::Function::Create(list_append_type, llvm::Function::ExternalLinkage, "PyList_Append", module);

        // int PyList_Extend(PyObject* list, PyObject* iterable)
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *list_extend_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_list_extend_func = llvm::Function::Create(list_extend_type, llvm::Function::ExternalLinkage, "PyList_Extend", module);

        // int PySequence_Contains(PyObject* o, PyObject* value)
        // Returns 1 if contains, 0 if not, -1 on error
        llvm::FunctionType *sequence_contains_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_sequence_contains_func = llvm::Function::Create(sequence_contains_type, llvm::Function::ExternalLinkage, "PySequence_Contains", module);

        // Bitwise operations
        // PyObject* PyNumber_Lshift(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_lshift_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_lshift_func = llvm::Function::Create(number_lshift_type, llvm::Function::ExternalLinkage, "PyNumber_Lshift", module);

        // PyObject* PyNumber_Rshift(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_rshift_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_rshift_func = llvm::Function::Create(number_rshift_type, llvm::Function::ExternalLinkage, "PyNumber_Rshift", module);

        // PyObject* PyNumber_And(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_and_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_and_func = llvm::Function::Create(number_and_type, llvm::Function::ExternalLinkage, "PyNumber_And", module);

        // PyObject* PyNumber_Or(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_or_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_or_func = llvm::Function::Create(number_or_type, llvm::Function::ExternalLinkage, "PyNumber_Or", module);

        // PyObject* PyNumber_Xor(PyObject* o1, PyObject* o2)
        llvm::FunctionType *number_xor_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_number_xor_func = llvm::Function::Create(number_xor_type, llvm::Function::ExternalLinkage, "PyNumber_Xor", module);

        // PyObject* PyCell_Get(PyObject* cell) - get contents of a cell object
        // Returns new reference to the cell contents
        llvm::FunctionType *cell_get_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_cell_get_func = llvm::Function::Create(cell_get_type, llvm::Function::ExternalLinkage, "PyCell_Get", module);

        // PyObject* PyTuple_GetItem(PyObject* tuple, Py_ssize_t index)
        // Returns borrowed reference
        llvm::FunctionType *tuple_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, i64_type}, false);
        py_tuple_getitem_func = llvm::Function::Create(tuple_getitem_type, llvm::Function::ExternalLinkage, "PyTuple_GetItem", module);

        // Py_ssize_t PyTuple_Size(PyObject* tuple)
        llvm::FunctionType *tuple_size_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
        py_tuple_size_func = llvm::Function::Create(tuple_size_type, llvm::Function::ExternalLinkage, "PyTuple_Size", module);

        // PyObject* PySlice_New(PyObject* start, PyObject* stop, PyObject* step)
        // Creates a new slice object - any can be NULL (means default)
        llvm::FunctionType *slice_new_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type, ptr_type}, false);
        py_slice_new_func = llvm::Function::Create(slice_new_type, llvm::Function::ExternalLinkage, "PySlice_New", module);

        // PyObject* PySequence_GetSlice(PyObject* o, Py_ssize_t i1, Py_ssize_t i2)
        // Returns o[i1:i2] - new reference
        llvm::FunctionType *sequence_getslice_type = llvm::FunctionType::get(ptr_type, {ptr_type, i64_type, i64_type}, false);
        py_sequence_getslice_func = llvm::Function::Create(sequence_getslice_type, llvm::Function::ExternalLinkage, "PySequence_GetSlice", module);

        // int PySequence_SetSlice(PyObject* o, Py_ssize_t i1, Py_ssize_t i2, PyObject* v)
        // Sets o[i1:i2] = v - returns 0 on success, -1 on failure
        llvm::FunctionType *sequence_setslice_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, i64_type, i64_type, ptr_type}, false);
        py_sequence_setslice_func = llvm::Function::Create(sequence_setslice_type, llvm::Function::ExternalLinkage, "PySequence_SetSlice", module);

        // Py_ssize_t PySequence_Size(PyObject* o)
        // Returns length of sequence, or -1 on error
        llvm::FunctionType *sequence_size_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
        py_sequence_size_func = llvm::Function::Create(sequence_size_type, llvm::Function::ExternalLinkage, "PySequence_Size", module);

        // PyObject* PySequence_Tuple(PyObject* o)
        // Convert any sequence to a tuple - returns new reference
        llvm::FunctionType *sequence_tuple_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_sequence_tuple_func = llvm::Function::Create(sequence_tuple_type, llvm::Function::ExternalLinkage, "PySequence_Tuple", module);

        // PyObject* PySequence_GetItem(PyObject* o, Py_ssize_t i)
        // Returns o[i], new reference. Supports negative indices.
        llvm::FunctionType *sequence_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, i64_type}, false);
        py_sequence_getitem_func = llvm::Function::Create(sequence_getitem_type, llvm::Function::ExternalLinkage, "PySequence_GetItem", module);

        // int PyObject_DelItem(PyObject* o, PyObject* key)
        // Delete o[key] - returns 0 on success, -1 on failure
        llvm::FunctionType *object_delitem_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_object_delitem_func = llvm::Function::Create(object_delitem_type, llvm::Function::ExternalLinkage, "PyObject_DelItem", module);

        // int _PySet_Update(PyObject* set, PyObject* iterable)
        // Update set with items from iterable - returns 0 on success, -1 on failure
        llvm::FunctionType *set_update_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_set_update_func = llvm::Function::Create(set_update_type, llvm::Function::ExternalLinkage, "_PySet_Update", module);

        // int PyDict_Update(PyObject* a, PyObject* b)
        // Update dict a with items from dict b - returns 0 on success, -1 on failure
        llvm::FunctionType *dict_update_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_dict_update_func = llvm::Function::Create(dict_update_type, llvm::Function::ExternalLinkage, "PyDict_Update", module);

        // int PyDict_Merge(PyObject* a, PyObject* b, int override)
        // Merge dict b into dict a - returns 0 on success, -1 on failure
        llvm::FunctionType *dict_merge_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type, builder->getInt32Ty()}, false);
        py_dict_merge_func = llvm::Function::Create(dict_merge_type, llvm::Function::ExternalLinkage, "PyDict_Merge", module);

        // PyObject* PyDict_GetItem(PyObject* p, PyObject* key)
        // Returns borrowed reference or NULL if not found (does NOT set exception)
        llvm::FunctionType *dict_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_dict_getitem_func = llvm::Function::Create(dict_getitem_type, llvm::Function::ExternalLinkage, "PyDict_GetItem", module);

        // ========== Exception Handling API (Bug #3 fix) ==========

        // PyObject* PyErr_Occurred(void)
        // Returns NULL if no error, otherwise returns the exception type (borrowed ref)
        llvm::FunctionType *err_occurred_type = llvm::FunctionType::get(ptr_type, {}, false);
        py_err_occurred_func = llvm::Function::Create(err_occurred_type, llvm::Function::ExternalLinkage, "PyErr_Occurred", module);

        // void PyErr_Fetch(PyObject** ptype, PyObject** pvalue, PyObject** ptraceback)
        // Retrieve exception info and clear error indicator
        llvm::FunctionType *err_fetch_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type, ptr_type}, false);
        py_err_fetch_func = llvm::Function::Create(err_fetch_type, llvm::Function::ExternalLinkage, "PyErr_Fetch", module);

        // void PyErr_Restore(PyObject* type, PyObject* value, PyObject* traceback)
        // Set exception info (steals references)
        llvm::FunctionType *err_restore_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type, ptr_type}, false);
        py_err_restore_func = llvm::Function::Create(err_restore_type, llvm::Function::ExternalLinkage, "PyErr_Restore", module);

        // void PyErr_SetObject(PyObject* type, PyObject* value)
        // Set exception (does not steal references)
        llvm::FunctionType *err_set_object_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type}, false);
        py_err_set_object_func = llvm::Function::Create(err_set_object_type, llvm::Function::ExternalLinkage, "PyErr_SetObject", module);

        // void PyErr_SetString(PyObject* type, const char* message)
        llvm::FunctionType *err_set_string_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type}, false);
        py_err_set_string_func = llvm::Function::Create(err_set_string_type, llvm::Function::ExternalLinkage, "PyErr_SetString", module);

        // void PyErr_Clear(void)
        // Clear current error indicator
        llvm::FunctionType *err_clear_type = llvm::FunctionType::get(void_type, {}, false);
        py_err_clear_func = llvm::Function::Create(err_clear_type, llvm::Function::ExternalLinkage, "PyErr_Clear", module);

        // int PyErr_GivenExceptionMatches(PyObject* given, PyObject* exc)
        // Returns 1 if given matches exc, 0 otherwise
        llvm::FunctionType *exception_matches_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_exception_matches_func = llvm::Function::Create(exception_matches_type, llvm::Function::ExternalLinkage, "PyErr_GivenExceptionMatches", module);

        // PyObject* PyObject_Type(PyObject* o)
        // Get the type of an object (new reference)
        llvm::FunctionType *object_type_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_object_type_func = llvm::Function::Create(object_type_type, llvm::Function::ExternalLinkage, "PyObject_Type", module);

        // void PyException_SetCause(PyObject* exc, PyObject* cause)
        // Set __cause__ attribute (steals reference to cause)
        llvm::FunctionType *set_cause_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type}, false);
        py_exception_set_cause_func = llvm::Function::Create(set_cause_type, llvm::Function::ExternalLinkage, "PyException_SetCause", module);

        // ========== Attribute/Name Deletion API ==========

        // int PyObject_DelAttr(PyObject* o, PyObject* attr_name)
        // Delete attribute named attr_name from object o
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *delattr_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_object_delattr_func = llvm::Function::Create(delattr_type, llvm::Function::ExternalLinkage, "PyObject_DelAttr", module);

        // int PyDict_DelItem(PyObject* p, PyObject* key)
        // Delete key from dictionary p
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *dict_delitem_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_dict_delitem_func = llvm::Function::Create(dict_delitem_type, llvm::Function::ExternalLinkage, "PyDict_DelItem", module);

        // int PyCell_Set(PyObject* cell, PyObject* value)
        // Set the contents of cell to value (NULL to clear)
        // Returns 0 on success, -1 on failure with exception set
        llvm::FunctionType *cell_set_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_cell_set_func = llvm::Function::Create(cell_set_type, llvm::Function::ExternalLinkage, "PyCell_Set", module);

        // ========== Format/String API (f-string support) ==========

        // PyObject* PyObject_Format(PyObject* obj, PyObject* format_spec)
        // Format obj using format_spec. format_spec can be NULL for default formatting
        // Returns formatted string on success, NULL on failure
        llvm::FunctionType *object_format_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_object_format_func = llvm::Function::Create(object_format_type, llvm::Function::ExternalLinkage, "PyObject_Format", module);

        // PyObject* PyObject_Repr(PyObject* o)
        // Compute string representation (repr) of object
        // Returns new reference on success, NULL on failure
        llvm::FunctionType *object_repr_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_object_repr_func = llvm::Function::Create(object_repr_type, llvm::Function::ExternalLinkage, "PyObject_Repr", module);

        // PyObject* PyObject_ASCII(PyObject* o)
        // Like repr() but escapes non-ASCII characters
        // Returns new reference on success, NULL on failure
        llvm::FunctionType *object_ascii_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        py_object_ascii_func = llvm::Function::Create(object_ascii_type, llvm::Function::ExternalLinkage, "PyObject_ASCII", module);

        // ========== Import API ==========

        // PyObject* PyImport_ImportModuleLevelObject(PyObject* name, PyObject* globals,
        //                                             PyObject* locals, PyObject* fromlist, int level)
        // Import a module with level (0 = absolute, >0 = relative)
        // Returns new reference to module on success, NULL on failure
        llvm::FunctionType *import_module_type = llvm::FunctionType::get(
            ptr_type, {ptr_type, ptr_type, ptr_type, ptr_type, builder->getInt32Ty()}, false);
        py_import_importmodule_func = llvm::Function::Create(import_module_type, llvm::Function::ExternalLinkage, "PyImport_ImportModuleLevelObject", module);

        // ========== Function Creation API (MAKE_FUNCTION / SET_FUNCTION_ATTRIBUTE) ==========

        // PyObject* PyFunction_New(PyObject* code, PyObject* globals)
        // Create a new function object from code object and globals dict
        // Returns new reference on success, NULL on failure
        llvm::FunctionType *function_new_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
        py_function_new_func = llvm::Function::Create(function_new_type, llvm::Function::ExternalLinkage, "PyFunction_New", module);

        // int PyFunction_SetDefaults(PyObject* op, PyObject* defaults)
        // Set tuple of default argument values for positional parameters
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *function_set_defaults_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_function_set_defaults_func = llvm::Function::Create(function_set_defaults_type, llvm::Function::ExternalLinkage, "PyFunction_SetDefaults", module);

        // int PyFunction_SetKwDefaults(PyObject* op, PyObject* kwdefaults)
        // Set dict of keyword-only parameter defaults
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *function_set_kwdefaults_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_function_set_kwdefaults_func = llvm::Function::Create(function_set_kwdefaults_type, llvm::Function::ExternalLinkage, "PyFunction_SetKwDefaults", module);

        // int PyFunction_SetAnnotations(PyObject* op, PyObject* annotations)
        // Set tuple of strings for parameter/return annotations
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *function_set_annotations_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_function_set_annotations_func = llvm::Function::Create(function_set_annotations_type, llvm::Function::ExternalLinkage, "PyFunction_SetAnnotations", module);

        // int PyFunction_SetClosure(PyObject* op, PyObject* closure)
        // Set tuple of cell objects for free variable closure
        // Returns 0 on success, -1 on failure
        llvm::FunctionType *function_set_closure_type = llvm::FunctionType::get(
            builder->getInt32Ty(), {ptr_type, ptr_type}, false);
        py_function_set_closure_func = llvm::Function::Create(function_set_closure_type, llvm::Function::ExternalLinkage, "PyFunction_SetClosure", module);

        // PyObject* jit_call_with_kwargs(PyObject* callable, PyObject** args, Py_ssize_t nargs, PyObject* kwnames)
        // Our C helper for CALL_KW opcode - splits args and builds kwargs dict at runtime
        llvm::FunctionType *call_with_kwargs_type = llvm::FunctionType::get(
            ptr_type, {ptr_type, ptr_type, i64_type, ptr_type}, false);
        jit_call_with_kwargs_func = llvm::Function::Create(call_with_kwargs_type, llvm::Function::ExternalLinkage, "jit_call_with_kwargs", module);
    }

    bool JITCore::compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::object py_globals_dict, nb::object py_builtins_dict, nb::list py_closure_cells, nb::list py_exception_table, const std::string &name, int param_count, int total_locals, int nlocals)
    {
        if (!jit)
        {
            return false;
        }

        // Check if already compiled to prevent duplicate symbol errors
        if (compiled_functions.count(name) > 0)
        {
            return true; // Already compiled, return success
        }

        // Bug #4 Fix: Store globals and builtins dicts for runtime lookup
        // These are dictionaries, not pre-resolved values
        globals_dict_ptr = py_globals_dict.ptr();
        Py_INCREF(globals_dict_ptr);

        builtins_dict_ptr = py_builtins_dict.ptr();
        Py_INCREF(builtins_dict_ptr);

        // Convert Python instructions list to C++ vector
        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i)
        {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]); // Get actual jump target from Python (can be negative)
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse exception table for try/except handling (Bug #3 fix)
        std::vector<ExceptionTableEntry> exception_table;
        for (size_t i = 0; i < py_exception_table.size(); ++i)
        {
            nb::dict entry_dict = nb::cast<nb::dict>(py_exception_table[i]);
            ExceptionTableEntry entry;
            entry.start = nb::cast<int32_t>(entry_dict["start"]);
            entry.end = nb::cast<int32_t>(entry_dict["end"]);
            entry.target = nb::cast<int32_t>(entry_dict["target"]);
            entry.depth = nb::cast<int32_t>(entry_dict["depth"]);
            entry.lasti = nb::cast<bool>(entry_dict["lasti"]);
            exception_table.push_back(entry);
        }

        // Convert Python constants list - support both int64 and PyObject*
        std::vector<int64_t> int_constants;
        std::vector<PyObject *> obj_constants;
        for (size_t i = 0; i < py_constants.size(); ++i)
        {
            nb::object const_obj = py_constants[i];
            PyObject *py_obj = const_obj.ptr();

            // Bug #2 Fix: Check for bool BEFORE int, since bool is a subclass of int
            // Python's True/False need to be stored as PyObject* to preserve identity
            if (py_obj == Py_True || py_obj == Py_False)
            {
                // Store bools as PyObject* to preserve True/False identity
                int_constants.push_back(0);
                Py_INCREF(py_obj); // Keep reference alive
                obj_constants.push_back(py_obj);
                stored_constants.push_back(py_obj); // Track for cleanup in destructor
            }
            // Try to convert to int64 for regular integers
            else if (PyLong_Check(py_obj))
            {
                try
                {
                    int64_t int_val = nb::cast<int64_t>(const_obj);
                    int_constants.push_back(int_val);
                    obj_constants.push_back(nullptr); // Mark as int constant
                }
                catch (...)
                {
                    // Integer too large for int64, store as PyObject*
                    int_constants.push_back(0);
                    Py_INCREF(py_obj); // Keep reference alive
                    obj_constants.push_back(py_obj);
                    stored_constants.push_back(py_obj); // Track for cleanup in destructor
                }
            }
            else
            {
                // All other types stored as PyObject*
                int_constants.push_back(0);
                Py_INCREF(py_obj); // Keep reference alive
                obj_constants.push_back(py_obj);
                stored_constants.push_back(py_obj); // Track for cleanup in destructor
            }
        }

        // Extract names (used by LOAD_ATTR, LOAD_GLOBAL, etc)
        std::vector<PyObject *> name_objects;
        for (size_t i = 0; i < py_names.size(); ++i)
        {
            nb::object name_obj = py_names[i];
            PyObject *py_name = name_obj.ptr();
            Py_INCREF(py_name); // Keep reference alive
            name_objects.push_back(py_name);
            stored_names.push_back(py_name); // Track for cleanup in destructor
        }

        // Bug #4 Fix: No longer extract global VALUES here.
        // globals_dict_ptr and builtins_dict_ptr are stored at the start of this function.
        // LOAD_GLOBAL will do runtime lookup using PyDict_GetItem.

        // Extract closure cells (used by COPY_FREE_VARS / LOAD_DEREF)
        std::vector<PyObject *> closure_cells;
        for (size_t i = 0; i < py_closure_cells.size(); ++i)
        {
            nb::object cell_obj = py_closure_cells[i];
            if (cell_obj.is_none())
            {
                closure_cells.push_back(nullptr);
            }
            else
            {
                PyObject *py_cell = cell_obj.ptr();
                Py_INCREF(py_cell); // Keep reference alive
                closure_cells.push_back(py_cell);
                stored_closure_cells.push_back(py_cell); // Track for cleanup in destructor
            }
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Declare Python C API functions
        declare_python_api_functions(module.get(), &builder);

        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);
        llvm::Type *ptr_type = builder.getPtrTy();

        // Create function type - return PyObject* (ptr) to support both int and object returns
        // In object mode, all values are PyObject*, ints are boxed as PyLong
        std::vector<llvm::Type *> param_types(param_count, ptr_type); // Parameters are PyObject*
        llvm::FunctionType *func_type = llvm::FunctionType::get(
            ptr_type, // Return PyObject*
            param_types,
            false);

        llvm::Function *func = llvm::Function::Create(
            func_type,
            llvm::Function::ExternalLinkage,
            name,
            module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        std::unordered_map<int, llvm::BasicBlock *> jump_targets;
        std::unordered_map<int, size_t> stack_depth_at_offset; // Track stack depth at each offset for loops

        // Bug #1 Fix: Track incoming stack states per block for PHI node insertion
        struct BlockStackState
        {
            std::vector<llvm::Value *> stack;
            llvm::BasicBlock *predecessor;
        };
        std::unordered_map<int, std::vector<BlockStackState>> block_incoming_stacks;
        std::unordered_map<int, bool> block_needs_phi; // Blocks that need PHI nodes

        // Create allocas only for actual locals needed (not 256!)
        // In object mode, all locals are PyObject* (ptr type)
        llvm::IRBuilder<> alloca_builder(entry, entry->begin());
        llvm::Value *null_ptr_init = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
        for (int i = 0; i < total_locals; ++i)
        {
            local_allocas[i] = alloca_builder.CreateAlloca(
                ptr_type, nullptr, "local_" + std::to_string(i));
            // CRITICAL: Initialize to NULL to avoid SEGFAULT on LOAD_FAST before STORE_FAST
            alloca_builder.CreateStore(null_ptr_init, local_allocas[i]);
        }

        // Store function parameters into allocas
        auto args = func->arg_begin();
        for (int i = 0; i < param_count; ++i)
        {
            builder.CreateStore(&*args++, local_allocas[i]);
        }

        // First pass: Create basic blocks for all jump targets
        jump_targets[0] = entry;
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];

            if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE ||
                instr.opcode == op::POP_JUMP_IF_NONE || instr.opcode == op::POP_JUMP_IF_NOT_NONE)
            {
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(target_offset), func);
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_" + std::to_string(target_offset), func);
                }
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
                // Pre-create blocks for forward jump targets
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "forward_" + std::to_string(target_offset), func);
                }
            }
        }

        // Bug #3 Fix: Create blocks for exception handler targets from exception table
        std::unordered_map<int, llvm::BasicBlock *> exception_handlers;
        std::unordered_map<int, int> exception_handler_depth; // Stack depth at handler entry
        for (const auto &exc_entry : exception_table)
        {
            if (!jump_targets.count(exc_entry.target))
            {
                jump_targets[exc_entry.target] = llvm::BasicBlock::Create(
                    *local_context, "exc_handler_" + std::to_string(exc_entry.target), func);
            }
            exception_handlers[exc_entry.target] = jump_targets[exc_entry.target];
            exception_handler_depth[exc_entry.target] = exc_entry.depth;
        }

        // Build a map from instruction offset to exception handler (if any)
        // This tells us where to jump when an error occurs at a given offset
        std::unordered_map<int, int> offset_to_handler;
        for (const auto &exc_entry : exception_table)
        {
            for (int off = exc_entry.start; off < exc_entry.end; off += 2)
            {
                // If multiple handlers cover the same offset, use the innermost (first in table)
                if (!offset_to_handler.count(off))
                {
                    offset_to_handler[off] = exc_entry.target;
                }
            }
        }

        // Bug #3 Fix: Helper lambda to generate error checking code after API calls
        // If an error occurred (PyErr_Occurred is non-NULL), branch to exception handler or return NULL
        auto check_error_and_branch = [&](int current_offset, llvm::Value *result, const char *call_name)
        {
            // Check if this offset has an exception handler
            if (offset_to_handler.count(current_offset))
            {
                int handler_offset = offset_to_handler[current_offset];

                // Create blocks for error path and continue path
                llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_error_" + std::to_string(current_offset), func);
                llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_continue_" + std::to_string(current_offset), func);

                // Check if result is NULL (error occurred)
                llvm::Value *is_error = builder.CreateICmpEQ(
                    result,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                    "is_error");
                builder.CreateCondBr(is_error, error_block, continue_block);

                // Error path: branch to exception handler
                builder.SetInsertPoint(error_block);

                // Stack unwinding: decref all values on the stack that are PyObject*
                // The exception handler expects a specific stack depth (exception_handler_depth)
                int target_depth = exception_handler_depth.count(handler_offset) ? exception_handler_depth[handler_offset] : 0;

                // Decref stack values above target depth
                for (size_t s = stack.size(); s > static_cast<size_t>(target_depth); --s)
                {
                    llvm::Value *val = stack[s - 1];
                    if (val->getType()->isPointerTy())
                    {
                        // Check not NULL before decref
                        llvm::Value *is_null = builder.CreateICmpEQ(
                            val,
                            llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                            "is_null");
                        llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(
                            *local_context, "decref_unwind", func);
                        llvm::BasicBlock *after_decref = llvm::BasicBlock::Create(
                            *local_context, "after_decref_unwind", func);
                        builder.CreateCondBr(is_null, after_decref, decref_block);
                        builder.SetInsertPoint(decref_block);
                        builder.CreateCall(py_decref_func, {val});
                        builder.CreateBr(after_decref);
                        builder.SetInsertPoint(after_decref);
                    }
                }

                // Branch to handler
                builder.CreateBr(jump_targets[handler_offset]);

                // Continue on success path
                builder.SetInsertPoint(continue_block);
            }
            else
            {
                // No exception handler: if error, just return NULL
                // Only check if result could be NULL
                llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_error_ret_" + std::to_string(current_offset), func);
                llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_continue_ret_" + std::to_string(current_offset), func);

                llvm::Value *is_error = builder.CreateICmpEQ(
                    result,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                    "is_error");
                builder.CreateCondBr(is_error, error_block, continue_block);

                builder.SetInsertPoint(error_block);
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                builder.SetInsertPoint(continue_block);
            }
        };

        // Helper to switch to a dead block after generating a terminator
        // This prevents generating invalid IR with code after ret/unreachable
        int dead_block_counter = 0;
        auto switch_to_dead_block = [&]()
        {
            llvm::BasicBlock *dead_block = llvm::BasicBlock::Create(
                *local_context, "dead_" + std::to_string(dead_block_counter++), func);
            builder.SetInsertPoint(dead_block);
            stack.clear();
        };

        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            int current_offset = instructions[i].offset;

            // If this offset is a jump target, switch to that block and handle PHI nodes
            if (jump_targets.count(current_offset) && jump_targets[current_offset] != builder.GetInsertBlock())
            {
                // Record current stack state before branching (for fall-through)
                llvm::BasicBlock *current_block = builder.GetInsertBlock();
                if (!current_block->getTerminator())
                {
                    // Record stack state for this predecessor
                    BlockStackState state;
                    state.stack = stack;
                    state.predecessor = current_block;
                    block_incoming_stacks[current_offset].push_back(state);

                    builder.CreateBr(jump_targets[current_offset]);
                }
                builder.SetInsertPoint(jump_targets[current_offset]);

                // Bug #1 Fix: Check if this block has recorded incoming stacks
                if (block_incoming_stacks.count(current_offset) &&
                    !block_incoming_stacks[current_offset].empty())
                {

                    auto &incoming = block_incoming_stacks[current_offset];

                    if (incoming.size() == 1)
                    {
                        // Single predecessor with recorded stack - just use it directly
                        stack = incoming[0].stack;
                    }
                    else
                    {
                        // Multiple predecessors - may need PHI nodes
                        size_t stack_size = incoming[0].stack.size();

                        // Verify all predecessors have same stack depth
                        bool valid = true;
                        for (const auto &s : incoming)
                        {
                            if (s.stack.size() != stack_size)
                            {
                                valid = false;
                                break;
                            }
                        }

                        if (valid && stack_size > 0)
                        {
                            // Create PHI nodes for each stack slot that differs
                            std::vector<llvm::Value *> merged_stack;
                            for (size_t slot = 0; slot < stack_size; ++slot)
                            {
                                // Check if all incoming values are the same (no PHI needed)
                                llvm::Value *first_val = incoming[0].stack[slot];
                                bool all_same = true;
                                for (size_t j = 1; j < incoming.size(); ++j)
                                {
                                    if (incoming[j].stack[slot] != first_val)
                                    {
                                        all_same = false;
                                        break;
                                    }
                                }

                                if (all_same)
                                {
                                    // No PHI needed, all paths have same value
                                    merged_stack.push_back(first_val);
                                }
                                else
                                {
                                    // Create PHI node at the start of this block
                                    llvm::Type *value_type = first_val->getType();
                                    llvm::PHINode *phi = builder.CreatePHI(
                                        value_type,
                                        incoming.size(),
                                        "stack_phi_" + std::to_string(slot));

                                    // Add incoming values from each predecessor
                                    for (const auto &s : incoming)
                                    {
                                        phi->addIncoming(s.stack[slot], s.predecessor);
                                    }

                                    merged_stack.push_back(phi);
                                }
                            }

                            // Replace stack with merged version
                            stack = merged_stack;
                        }
                        else if (stack_depth_at_offset.count(current_offset))
                        {
                            // Fallback: just restore stack depth
                            size_t expected_depth = stack_depth_at_offset[current_offset];
                            while (stack.size() > expected_depth)
                            {
                                stack.pop_back();
                            }
                        }
                    }
                }
                else if (stack_depth_at_offset.count(current_offset))
                {
                    // Single predecessor or no recorded stacks - restore stack depth
                    size_t expected_depth = stack_depth_at_offset[current_offset];
                    while (stack.size() > expected_depth)
                    {
                        stack.pop_back();
                    }
                }
            }

            // Record stack depth at this offset (only if not already recorded)
            // This preserves the FIRST time we see this offset's stack state
            if (!stack_depth_at_offset.count(current_offset))
            {
                stack_depth_at_offset[current_offset] = stack.size();
            }

            const auto &instr = instructions[i];

            // Python 3.13 opcodes
            if (instr.opcode == op::RESUME || instr.opcode == op::CACHE)
            {
                // RESUME is function preamble, CACHE is placeholder for adaptive interpreter
                continue;
            }
            else if (instr.opcode == op::COPY_FREE_VARS)
            {
                // Copy closure cells from __closure__ tuple into local slots
                // Slots for free vars start at nlocals (after local variables)
                // The cells themselves are stored at compile time in closure_cells vector
                int num_free_vars = instr.arg;
                for (int j = 0; j < num_free_vars && j < static_cast<int>(closure_cells.size()); ++j)
                {
                    if (closure_cells[j] != nullptr)
                    {
                        // Store the cell pointer in local slot nlocals + j
                        int slot = nlocals + j;
                        if (local_allocas.count(slot))
                        {
                            llvm::Value *cell_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(closure_cells[j]));
                            llvm::Value *cell_obj = builder.CreateIntToPtr(cell_ptr, ptr_type);
                            builder.CreateStore(cell_obj, local_allocas[slot]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::LOAD_DEREF)
            {
                // Load from a cell/free variable slot
                // The slot contains a PyCellObject, we need to get its contents
                int slot = instr.arg;
                if (local_allocas.count(slot))
                {
                    llvm::Value *cell = builder.CreateLoad(ptr_type, local_allocas[slot], "load_cell_" + std::to_string(slot));
                    // PyCell_Get returns new reference to cell contents
                    llvm::Value *contents = builder.CreateCall(py_cell_get_func, {cell}, "cell_contents");
                    stack.push_back(contents);
                }
            }
            else if (instr.opcode == op::STORE_DEREF)
            {
                // Store to a cell/free variable slot
                // Stack: TOS = value to store
                // The slot contains a PyCellObject, we store value into the cell
                int slot = instr.arg;
                if (!stack.empty() && local_allocas.count(slot))
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    // Box i64 to PyLong if needed - PyCell_Set expects PyObject*
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                    }

                    llvm::Value *cell = builder.CreateLoad(ptr_type, local_allocas[slot], "store_cell_" + std::to_string(slot));

                    // PyCell_Set(cell, value) - steals reference to value
                    // Returns 0 on success, -1 on failure
                    builder.CreateCall(py_cell_set_func, {cell, value});
                    // Note: PyCell_Set steals reference, so no decref needed
                }
            }
            else if (instr.opcode == op::LOAD_FAST)
            {
                if (local_allocas.count(instr.arg))
                {
                    // In object mode, load PyObject* from local
                    llvm::Value *loaded = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "load_local_" + std::to_string(instr.arg));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded});
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                // Python 3.13: Pushes co_varnames[arg>>4] then co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                if (local_allocas.count(first_local))
                {
                    llvm::Value *loaded1 = builder.CreateLoad(ptr_type, local_allocas[first_local], "load_local_" + std::to_string(first_local));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded1});
                    stack.push_back(loaded1);
                }
                if (local_allocas.count(second_local))
                {
                    llvm::Value *loaded2 = builder.CreateLoad(ptr_type, local_allocas[second_local], "load_local_" + std::to_string(second_local));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded2});
                    stack.push_back(loaded2);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_AND_CLEAR)
            {
                // Python 3.13: Load local variable and set it to NULL (for comprehensions)
                // Pushes the value of the local variable at index arg, then sets the local to NULL.
                // This is used in comprehensions to temporarily clear the iteration variable.
                if (local_allocas.count(instr.arg))
                {
                    llvm::Value *loaded = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "load_local_" + std::to_string(instr.arg));
                    // Incref since we're taking it to the stack (but no decref on clear since we're keeping the ref)
                    builder.CreateCall(py_xincref_func, {loaded}); // Use XINCREF since it might be NULL
                    stack.push_back(loaded);
                    // Clear the local variable (set to NULL)
                    llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    builder.CreateStore(null_ptr, local_allocas[instr.arg]);
                }
                else
                {
                    // Local not allocated yet, push NULL
                    llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    stack.push_back(null_ptr);
                }
            }
            else if (instr.opcode == op::STORE_FAST_LOAD_FAST)
            {
                // Python 3.13: Store TOS into local arg>>4, then load local arg&15
                // This is an optimized combined opcode used in comprehensions
                int store_local = instr.arg >> 4;
                int load_local = instr.arg & 0xF;

                // First: STORE_FAST for store_local
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // Box i64 values if needed
                    if (val->getType()->isIntegerTy(64))
                    {
                        val = builder.CreateCall(py_long_fromlonglong_func, {val});
                    }

                    // Decref old value before storing new one
                    if (local_allocas.count(store_local))
                    {
                        llvm::Value *old_val = builder.CreateLoad(ptr_type, local_allocas[store_local], "old_local");
                        llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        llvm::Value *is_not_null = builder.CreateICmpNE(old_val, null_check, "is_not_null");

                        llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(*local_context, "decref_old", func);
                        llvm::BasicBlock *store_block = llvm::BasicBlock::Create(*local_context, "store_new", func);
                        builder.CreateCondBr(is_not_null, decref_block, store_block);

                        builder.SetInsertPoint(decref_block);
                        builder.CreateCall(py_decref_func, {old_val});
                        builder.CreateBr(store_block);

                        builder.SetInsertPoint(store_block);
                        builder.CreateStore(val, local_allocas[store_local]);
                    }
                }

                // Second: LOAD_FAST for load_local
                if (local_allocas.count(load_local))
                {
                    llvm::Value *loaded = builder.CreateLoad(ptr_type, local_allocas[load_local], "load_local_" + std::to_string(load_local));
                    builder.CreateCall(py_incref_func, {loaded});
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                // arg is index into constants table
                if (instr.arg < int_constants.size())
                {
                    if (obj_constants[instr.arg] != nullptr)
                    {
                        // PyObject* constant - load as pointer
                        llvm::Value *const_ptr = llvm::ConstantInt::get(
                            i64_type,
                            reinterpret_cast<uint64_t>(obj_constants[instr.arg]));
                        llvm::Value *py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                        // Increment reference count since we're putting it on stack
                        builder.CreateCall(py_incref_func, {py_obj});
                        stack.push_back(py_obj);
                    }
                    else
                    {
                        // int64 constant - keep as i64 to allow fast native integer operations
                        // Will be boxed to PyLong only when needed (e.g., when mixed with PyObject*)
                        llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                        stack.push_back(const_val);
                    }
                }
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // In object mode, locals are PyObject* typed, so box i64 values
                    if (val->getType()->isIntegerTy(64))
                    {
                        val = builder.CreateCall(py_long_fromlonglong_func, {val});
                    }

                    // CRITICAL: Decref old value before storing new one to prevent memory leak
                    llvm::Value *old_val = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "old_local");
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *is_not_null = builder.CreateICmpNE(old_val, null_check, "is_not_null");

                    llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(*local_context, "decref_old", func);
                    llvm::BasicBlock *store_block = llvm::BasicBlock::Create(*local_context, "store_new", func);

                    builder.CreateCondBr(is_not_null, decref_block, store_block);

                    builder.SetInsertPoint(decref_block);
                    builder.CreateCall(py_decref_func, {old_val});
                    builder.CreateBr(store_block);

                    builder.SetInsertPoint(store_block);
                    builder.CreateStore(val, local_allocas[instr.arg]);
                }
            }
            else if (instr.opcode == op::STORE_FAST_STORE_FAST)
            {
                // Python 3.13: Stores STACK[-1] into co_varnames[arg>>4] and STACK[-2] into co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;

                if (stack.size() >= 2)
                {
                    llvm::Value *first_val = stack.back();
                    stack.pop_back(); // STACK[-1]
                    llvm::Value *second_val = stack.back();
                    stack.pop_back(); // STACK[-2]

                    // Box i64 values if needed
                    if (first_val->getType()->isIntegerTy(64))
                    {
                        first_val = builder.CreateCall(py_long_fromlonglong_func, {first_val});
                    }
                    if (second_val->getType()->isIntegerTy(64))
                    {
                        second_val = builder.CreateCall(py_long_fromlonglong_func, {second_val});
                    }

                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));

                    // Store first to first_local (with decref of old value)
                    if (local_allocas.count(first_local))
                    {
                        llvm::Value *old_val1 = builder.CreateLoad(ptr_type, local_allocas[first_local], "old_local1");
                        llvm::Value *is_not_null1 = builder.CreateICmpNE(old_val1, null_check, "is_not_null1");

                        llvm::BasicBlock *decref_block1 = llvm::BasicBlock::Create(*local_context, "decref_old1", func);
                        llvm::BasicBlock *store_block1 = llvm::BasicBlock::Create(*local_context, "store_new1", func);

                        builder.CreateCondBr(is_not_null1, decref_block1, store_block1);

                        builder.SetInsertPoint(decref_block1);
                        builder.CreateCall(py_decref_func, {old_val1});
                        builder.CreateBr(store_block1);

                        builder.SetInsertPoint(store_block1);
                        builder.CreateStore(first_val, local_allocas[first_local]);
                    }

                    // Store second to second_local (with decref of old value)
                    if (local_allocas.count(second_local))
                    {
                        llvm::Value *old_val2 = builder.CreateLoad(ptr_type, local_allocas[second_local], "old_local2");
                        llvm::Value *is_not_null2 = builder.CreateICmpNE(old_val2, null_check, "is_not_null2");

                        llvm::BasicBlock *decref_block2 = llvm::BasicBlock::Create(*local_context, "decref_old2", func);
                        llvm::BasicBlock *store_block2 = llvm::BasicBlock::Create(*local_context, "store_new2", func);

                        builder.CreateCondBr(is_not_null2, decref_block2, store_block2);

                        builder.SetInsertPoint(decref_block2);
                        builder.CreateCall(py_decref_func, {old_val2});
                        builder.CreateBr(store_block2);

                        builder.SetInsertPoint(store_block2);
                        builder.CreateStore(second_val, local_allocas[second_local]);
                    }
                }
            }
            else if (instr.opcode == op::UNPACK_SEQUENCE)
            {
                // Unpack TOS into count individual values
                // Stack order after unpack: [..., last_value, ..., first_value] (first value on TOS)
                int count = instr.arg;

                if (!stack.empty())
                {
                    llvm::Value *sequence = stack.back();
                    stack.pop_back();

                    // Unpack the sequence by calling PySequence_GetItem for each index
                    // Push in reverse order so that first item ends up on top
                    std::vector<llvm::Value *> unpacked;
                    for (int i = 0; i < count; ++i)
                    {
                        llvm::Value *idx = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value *idx_obj = builder.CreateCall(py_long_fromlonglong_func, {idx});
                        llvm::Value *item = builder.CreateCall(py_object_getitem_func, {sequence, idx_obj});
                        builder.CreateCall(py_decref_func, {idx_obj}); // Free temp index
                        unpacked.push_back(item);
                    }

                    // Push in reverse order (last item first, so first item is on top)
                    for (int i = count - 1; i >= 0; --i)
                    {
                        stack.push_back(unpacked[i]);
                    }

                    // Decref the original sequence (we're done with it)
                    if (sequence->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {sequence});
                    }
                }
            }
            else if (instr.opcode == op::UNPACK_EX)
            {
                // UNPACK_EX: Unpack sequence with starred target
                // arg = (count_after << 8) | count_before
                // Example: a, *b, c = [1,2,3,4,5] -> count_before=1, count_after=1
                // Stack before: sequence
                // Stack after: values in reverse order (last to first), with starred list in middle
                // For a, *b, c with [1,2,3,4,5]: push c=5, then b=[2,3,4], then a=1
                // So store order (popping) gives: a=1, b=[2,3,4], c=5
                int count_before = instr.arg & 0xFF;
                int count_after = (instr.arg >> 8) & 0xFF;

                if (!stack.empty())
                {
                    llvm::Value *sequence = stack.back();
                    stack.pop_back();

                    // Get sequence length using PySequence_Size (works on lists, tuples, etc.)
                    llvm::Value *seq_len = builder.CreateCall(py_sequence_size_func, {sequence}, "seq_len");

                    // Unpack first 'count_before' items using PySequence_GetItem
                    std::vector<llvm::Value *> before_items;
                    for (int i = 0; i < count_before; ++i)
                    {
                        llvm::Value *idx = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value *item = builder.CreateCall(py_sequence_getitem_func, {sequence, idx}, "before_item");
                        check_error_and_branch(current_offset, item, "unpack_ex_before");
                        before_items.push_back(item);
                    }

                    // Unpack last 'count_after' items (negative indices)
                    std::vector<llvm::Value *> after_items;
                    for (int i = count_after; i > 0; --i)
                    {
                        // Negative index from end: -(i)
                        llvm::Value *neg_idx = llvm::ConstantInt::get(i64_type, -static_cast<int64_t>(i));
                        llvm::Value *item = builder.CreateCall(py_sequence_getitem_func, {sequence, neg_idx}, "after_item");
                        check_error_and_branch(current_offset, item, "unpack_ex_after");
                        after_items.push_back(item);
                    }

                    // Middle items go into a list using PySequence_GetSlice
                    // middle = sequence[count_before : len-count_after]
                    llvm::Value *middle_start = llvm::ConstantInt::get(i64_type, count_before);
                    llvm::Value *after_count_val = llvm::ConstantInt::get(i64_type, count_after);
                    llvm::Value *middle_end = builder.CreateSub(seq_len, after_count_val, "middle_end");

                    // Get the middle slice as a list
                    llvm::Value *middle_list = builder.CreateCall(py_sequence_getslice_func,
                                                                  {sequence, middle_start, middle_end}, "middle_list");
                    check_error_and_branch(current_offset, middle_list, "unpack_ex_middle");

                    // Push in reverse order for STORE_FAST to pop correctly
                    // STORE_FAST will pop: first before_items, then middle_list, then after_items
                    // So we push: after_items first (in reverse), then middle, then before (in reverse)
                    for (int i = static_cast<int>(after_items.size()) - 1; i >= 0; --i)
                    {
                        stack.push_back(after_items[i]);
                    }
                    stack.push_back(middle_list);
                    for (int i = static_cast<int>(before_items.size()) - 1; i >= 0; --i)
                    {
                        stack.push_back(before_items[i]);
                    }

                    // Decref original sequence
                    builder.CreateCall(py_decref_func, {sequence});
                }
            }
            else if (instr.opcode == op::BINARY_OP)
            {
                if (stack.size() >= 2)
                {
                    // Stack: [first_pushed, second_pushed] with second on top
                    llvm::Value *second = stack.back();
                    stack.pop_back();
                    llvm::Value *first = stack.back();
                    stack.pop_back();

                    llvm::Value *result = nullptr;

                    // Check if either operand is PyObject* (ptr type)
                    bool first_is_pyobject = first->getType()->isPointerTy() && !first->getType()->isIntegerTy(64);
                    bool second_is_pyobject = second->getType()->isPointerTy() && !second->getType()->isIntegerTy(64);
                    bool is_pyobject = first_is_pyobject || second_is_pyobject;

                    // Bug #3 Fix: Operations that can raise exceptions (division, modulo, power)
                    // MUST go through Python API for proper exception handling
                    bool can_raise = (instr.arg == 11 || // TRUE_DIV (a / b)
                                      instr.arg == 2 ||  // FLOOR_DIV (a // b)
                                      instr.arg == 6 ||  // MOD (a % b)
                                      instr.arg == 8);   // POW (a ** b)

                    if (is_pyobject || can_raise)
                    {
                        // Object mode: use Python C API
                        // Box both operands to PyObject* if needed, track if we created temps
                        llvm::Value *orig_first = first;
                        llvm::Value *orig_second = second;
                        bool first_boxed = false;
                        bool second_boxed = false;

                        if (first->getType()->isIntegerTy(64))
                        {
                            first = builder.CreateCall(py_long_fromlonglong_func, {first});
                            first_boxed = true;
                        }
                        if (second->getType()->isIntegerTy(64))
                        {
                            second = builder.CreateCall(py_long_fromlonglong_func, {second});
                            second_boxed = true;
                        }

                        switch (instr.arg)
                        {
                        case 0:  // ADD (a + b)
                        case 13: // INPLACE_ADD (a += b)
                            result = builder.CreateCall(py_number_add_func, {first, second});
                            break;
                        case 10: // SUB (a - b)
                        case 23: // INPLACE_SUB (a -= b)
                            result = builder.CreateCall(py_number_subtract_func, {first, second});
                            break;
                        case 5:  // MUL (a * b)
                        case 18: // INPLACE_MUL (a *= b)
                            result = builder.CreateCall(py_number_multiply_func, {first, second});
                            break;
                        case 11: // TRUE_DIV (a / b)
                        case 24: // INPLACE_TRUE_DIV (a /= b)
                            result = builder.CreateCall(py_number_truedivide_func, {first, second});
                            break;
                        case 2:  // FLOOR_DIV (a // b)
                        case 15: // INPLACE_FLOOR_DIV (a //= b)
                            result = builder.CreateCall(py_number_floordivide_func, {first, second});
                            break;
                        case 6:  // MOD (a % b)
                        case 19: // INPLACE_MOD (a %= b)
                            result = builder.CreateCall(py_number_remainder_func, {first, second});
                            break;
                        case 8:  // POW (a ** b)
                        case 21: // INPLACE_POW (a **= b)
                        { 
                            // PyNumber_Power(base, exp, Py_None) - Py_None for no modular arithmetic
                            llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                                i64_type,
                                reinterpret_cast<uint64_t>(Py_None));
                            llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                            result = builder.CreateCall(py_number_power_func, {first, second, py_none});
                            break;
                        }
                        case 1:  // AND (a & b) - bitwise
                        case 14: // INPLACE_AND (a &= b)
                            result = builder.CreateCall(py_number_and_func, {first, second});
                            break;
                        case 7:  // OR (a | b) - bitwise
                        case 20: // INPLACE_OR (a |= b)
                            result = builder.CreateCall(py_number_or_func, {first, second});
                            break;
                        case 12: // XOR (a ^ b) - bitwise
                        case 25: // INPLACE_XOR (a ^= b)
                            result = builder.CreateCall(py_number_xor_func, {first, second});
                            break;
                        case 3:  // LSHIFT (a << b)
                        case 16: // INPLACE_LSHIFT (a <<= b)
                            result = builder.CreateCall(py_number_lshift_func, {first, second});
                            break;
                        case 9:  // RSHIFT (a >> b)
                        case 22: // INPLACE_RSHIFT (a >>= b)
                            result = builder.CreateCall(py_number_rshift_func, {first, second});
                            break;
                        case 4:  // MATMUL (a @ b)
                        case 17: // INPLACE_MATMUL (a @= b)
                            result = builder.CreateCall(py_number_matrixmultiply_func, {first, second});
                            break;
                        default:
                            // Unsupported binary op - set error and return NULL
                            {
                                llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                                    llvm::Type::getVoidTy(*local_context),
                                    {ptr_type, ptr_type}, false);
                                llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                                    "PyErr_SetString", py_err_set_str_type);
                                llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                                    i64_type, reinterpret_cast<uint64_t>(PyExc_TypeError));
                                llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                                llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported binary operation");
                                builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                                result = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                            }
                            break;
                        }

                        // Decref boxed temporaries (not the originals)
                        if (first_boxed)
                        {
                            builder.CreateCall(py_decref_func, {first});
                        }
                        else if (first_is_pyobject)
                        {
                            // Decref original PyObject* operand we consumed
                            builder.CreateCall(py_decref_func, {first});
                        }
                        if (second_boxed)
                        {
                            builder.CreateCall(py_decref_func, {second});
                        }
                        else if (second_is_pyobject)
                        {
                            // Decref original PyObject* operand we consumed
                            builder.CreateCall(py_decref_func, {second});
                        }

                        // Bug #3 Fix: Check for error (division by zero, type errors, etc.)
                        // result can be NULL if the operation raised an exception
                        if (result && result->getType()->isPointerTy())
                        {
                            check_error_and_branch(current_offset, result, "binary_op");
                        }
                    }
                    else
                    {
                        // Native int64 mode
                        switch (instr.arg)
                        {
                        case 0:  // ADD (a + b)
                        case 13: // INPLACE_ADD (a += b)
                            result = builder.CreateAdd(first, second, "add");
                            break;
                        case 10: // SUB (a - b)
                        case 23: // INPLACE_SUB (a -= b)
                            result = builder.CreateSub(first, second, "sub");
                            break;
                        case 5:  // MUL (a * b)
                        case 18: // INPLACE_MUL (a *= b)
                            result = builder.CreateMul(first, second, "mul");
                            break;
                        case 11: // TRUE_DIV (a / b) - returns float, but we only support int for now
                        case 24: // INPLACE_TRUE_DIV (a /= b)
                        {
                            // Check for division by zero - fall back to Python API which raises properly
                            llvm::Value *is_zero = builder.CreateICmpEQ(second, llvm::ConstantInt::get(i64_type, 0), "divisor_is_zero");
                            llvm::Function *current_fn = builder.GetInsertBlock()->getParent();
                            llvm::BasicBlock *div_ok = llvm::BasicBlock::Create(*local_context, "div_ok", current_fn);
                            llvm::BasicBlock *div_zero = llvm::BasicBlock::Create(*local_context, "div_zero", current_fn);
                            llvm::BasicBlock *div_cont = llvm::BasicBlock::Create(*local_context, "div_cont", current_fn);
                            
                            builder.CreateCondBr(is_zero, div_zero, div_ok);
                            
                            // Division by zero path - box operands and use Python API to raise error
                            builder.SetInsertPoint(div_zero);
                            llvm::Value *lhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {first});
                            llvm::Value *rhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {second});
                            llvm::Value *div_err_result = builder.CreateCall(py_number_truedivide_func, {lhs_boxed, rhs_boxed});
                            builder.CreateCall(py_decref_func, {lhs_boxed});
                            builder.CreateCall(py_decref_func, {rhs_boxed});
                            // div_err_result is NULL - check and branch to error
                            check_error_and_branch(current_offset, div_err_result, "div_by_zero");
                            builder.CreateBr(div_cont);  // Unreachable but needed for IR validity
                            
                            // Normal division path
                            builder.SetInsertPoint(div_ok);
                            llvm::Value *div_result = builder.CreateSDiv(first, second, "div");
                            builder.CreateBr(div_cont);
                            
                            // Continuation - use PHI to merge results
                            builder.SetInsertPoint(div_cont);
                            llvm::PHINode *div_phi = builder.CreatePHI(i64_type, 2, "div_phi");
                            div_phi->addIncoming(div_result, div_ok);
                            div_phi->addIncoming(llvm::ConstantInt::get(i64_type, 0), div_zero); // Dummy, never used
                            result = div_phi;
                            break;
                        }
                        case 2:  // FLOOR_DIV (a // b)
                        case 15: // INPLACE_FLOOR_DIV (a //= b)
                        {
                            // Check for division by zero
                            llvm::Value *is_zero = builder.CreateICmpEQ(second, llvm::ConstantInt::get(i64_type, 0), "divisor_is_zero");
                            llvm::Function *current_fn = builder.GetInsertBlock()->getParent();
                            llvm::BasicBlock *div_ok = llvm::BasicBlock::Create(*local_context, "floordiv_ok", current_fn);
                            llvm::BasicBlock *div_zero = llvm::BasicBlock::Create(*local_context, "floordiv_zero", current_fn);
                            llvm::BasicBlock *div_cont = llvm::BasicBlock::Create(*local_context, "floordiv_cont", current_fn);
                            
                            builder.CreateCondBr(is_zero, div_zero, div_ok);
                            
                            // Division by zero path
                            builder.SetInsertPoint(div_zero);
                            llvm::Value *lhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {first});
                            llvm::Value *rhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {second});
                            llvm::Value *div_err_result = builder.CreateCall(py_number_floordivide_func, {lhs_boxed, rhs_boxed});
                            builder.CreateCall(py_decref_func, {lhs_boxed});
                            builder.CreateCall(py_decref_func, {rhs_boxed});
                            check_error_and_branch(current_offset, div_err_result, "floordiv_by_zero");
                            builder.CreateBr(div_cont);
                            
                            // Normal division path
                            builder.SetInsertPoint(div_ok);
                            llvm::Value *div_result = builder.CreateSDiv(first, second, "floordiv");
                            builder.CreateBr(div_cont);
                            
                            // Continuation
                            builder.SetInsertPoint(div_cont);
                            llvm::PHINode *div_phi = builder.CreatePHI(i64_type, 2, "floordiv_phi");
                            div_phi->addIncoming(div_result, div_ok);
                            div_phi->addIncoming(llvm::ConstantInt::get(i64_type, 0), div_zero);
                            result = div_phi;
                            break;
                        }
                        case 6:  // MOD (a % b)
                        case 19: // INPLACE_MOD (a %= b)
                        {
                            // Check for modulo by zero
                            llvm::Value *is_zero = builder.CreateICmpEQ(second, llvm::ConstantInt::get(i64_type, 0), "divisor_is_zero");
                            llvm::Function *current_fn = builder.GetInsertBlock()->getParent();
                            llvm::BasicBlock *mod_ok = llvm::BasicBlock::Create(*local_context, "mod_ok", current_fn);
                            llvm::BasicBlock *mod_zero = llvm::BasicBlock::Create(*local_context, "mod_zero", current_fn);
                            llvm::BasicBlock *mod_cont = llvm::BasicBlock::Create(*local_context, "mod_cont", current_fn);
                            
                            builder.CreateCondBr(is_zero, mod_zero, mod_ok);
                            
                            // Modulo by zero path
                            builder.SetInsertPoint(mod_zero);
                            llvm::Value *lhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {first});
                            llvm::Value *rhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {second});
                            llvm::Value *mod_err_result = builder.CreateCall(py_number_remainder_func, {lhs_boxed, rhs_boxed});
                            builder.CreateCall(py_decref_func, {lhs_boxed});
                            builder.CreateCall(py_decref_func, {rhs_boxed});
                            check_error_and_branch(current_offset, mod_err_result, "mod_by_zero");
                            builder.CreateBr(mod_cont);
                            
                            // Normal modulo path
                            builder.SetInsertPoint(mod_ok);
                            llvm::Value *mod_result = builder.CreateSRem(first, second, "mod");
                            builder.CreateBr(mod_cont);
                            
                            // Continuation
                            builder.SetInsertPoint(mod_cont);
                            llvm::PHINode *mod_phi = builder.CreatePHI(i64_type, 2, "mod_phi");
                            mod_phi->addIncoming(mod_result, mod_ok);
                            mod_phi->addIncoming(llvm::ConstantInt::get(i64_type, 0), mod_zero);
                            result = mod_phi;
                            break;
                        }
                        case 1:  // AND (a & b) - bitwise
                        case 14: // INPLACE_AND (a &= b)
                            result = builder.CreateAnd(first, second, "and");
                            break;
                        case 7:  // OR (a | b) - bitwise
                        case 20: // INPLACE_OR (a |= b)
                            result = builder.CreateOr(first, second, "or");
                            break;
                        case 12: // XOR (a ^ b) - bitwise
                        case 25: // INPLACE_XOR (a ^= b)
                            result = builder.CreateXor(first, second, "xor");
                            break;
                        case 3:  // LSHIFT (a << b)
                        case 16: // INPLACE_LSHIFT (a <<= b)
                            result = builder.CreateShl(first, second, "shl");
                            break;
                        case 9:  // RSHIFT (a >> b)
                        case 22: // INPLACE_RSHIFT (a >>= b)
                            result = builder.CreateAShr(first, second, "shr");
                            break;
                        case 8:  // POW (a ** b) - Binary exponentiation O(log n)
                        case 21: // INPLACE_POW (a **= b)
                        {
                            // Implement iterative binary exponentiation in LLVM IR
                            // result = 1; while (exp > 0) { if (exp & 1) result *= base; base *= base; exp >>= 1; }

                            llvm::Function *current_func = builder.GetInsertBlock()->getParent();

                            // Create basic blocks for the power loop
                            llvm::BasicBlock *pow_entry = builder.GetInsertBlock();
                            llvm::BasicBlock *pow_loop = llvm::BasicBlock::Create(*local_context, "pow_loop", current_func);
                            llvm::BasicBlock *pow_odd = llvm::BasicBlock::Create(*local_context, "pow_odd", current_func);
                            llvm::BasicBlock *pow_cont = llvm::BasicBlock::Create(*local_context, "pow_cont", current_func);
                            llvm::BasicBlock *pow_done = llvm::BasicBlock::Create(*local_context, "pow_done", current_func);

                            // Entry: initialize and jump to loop
                            llvm::Value *init_result = llvm::ConstantInt::get(i64_type, 1);
                            builder.CreateBr(pow_loop);

                            // Loop header with phi nodes
                            builder.SetInsertPoint(pow_loop);
                            llvm::PHINode *phi_result = builder.CreatePHI(i64_type, 2, "pow_result");
                            llvm::PHINode *phi_base = builder.CreatePHI(i64_type, 2, "pow_base");
                            llvm::PHINode *phi_exp = builder.CreatePHI(i64_type, 2, "pow_exp");

                            phi_result->addIncoming(init_result, pow_entry);
                            phi_base->addIncoming(first, pow_entry);
                            phi_exp->addIncoming(second, pow_entry);

                            // Check if exp > 0
                            llvm::Value *exp_gt_zero = builder.CreateICmpSGT(
                                phi_exp,
                                llvm::ConstantInt::get(i64_type, 0),
                                "exp_gt_zero");
                            builder.CreateCondBr(exp_gt_zero, pow_odd, pow_done);

                            // Check if exp is odd (exp & 1)
                            builder.SetInsertPoint(pow_odd);
                            llvm::Value *exp_is_odd = builder.CreateAnd(
                                phi_exp,
                                llvm::ConstantInt::get(i64_type, 1),
                                "exp_is_odd");
                            llvm::Value *is_odd = builder.CreateICmpNE(
                                exp_is_odd,
                                llvm::ConstantInt::get(i64_type, 0),
                                "is_odd");

                            // If odd: new_result = result * base, else: new_result = result
                            llvm::Value *result_times_base = builder.CreateMul(phi_result, phi_base, "result_times_base");
                            llvm::Value *new_result = builder.CreateSelect(is_odd, result_times_base, phi_result, "new_result");

                            // base = base * base
                            llvm::Value *new_base = builder.CreateMul(phi_base, phi_base, "base_squared");

                            // exp = exp >> 1
                            llvm::Value *new_exp = builder.CreateAShr(
                                phi_exp,
                                llvm::ConstantInt::get(i64_type, 1),
                                "exp_halved");

                            builder.CreateBr(pow_cont);

                            // Continue block - update phi nodes and loop back
                            builder.SetInsertPoint(pow_cont);
                            phi_result->addIncoming(new_result, pow_cont);
                            phi_base->addIncoming(new_base, pow_cont);
                            phi_exp->addIncoming(new_exp, pow_cont);
                            builder.CreateBr(pow_loop);

                            // Done block - result is in phi_result
                            builder.SetInsertPoint(pow_done);
                            result = phi_result;
                            break;
                        }
                        default:
                            // Unsupported binary op in integer mode - box and use Python API with error
                            {
                                llvm::Value *lhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {first});
                                llvm::Value *rhs_boxed = builder.CreateCall(py_long_fromlonglong_func, {second});
                                llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                                    llvm::Type::getVoidTy(*local_context),
                                    {ptr_type, ptr_type}, false);
                                llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                                    "PyErr_SetString", py_err_set_str_type);
                                llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                                    i64_type, reinterpret_cast<uint64_t>(PyExc_TypeError));
                                llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                                llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported binary operation");
                                builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                                builder.CreateCall(py_decref_func, {lhs_boxed});
                                builder.CreateCall(py_decref_func, {rhs_boxed});
                                // Return NULL to signal error
                                result = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                            }
                            break;
                        }
                    }

                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_NEGATIVE)
            {
                // Implements STACK[-1] = -STACK[-1]
                if (!stack.empty())
                {
                    llvm::Value *operand = stack.back();
                    stack.pop_back();
                    llvm::Value *result = nullptr;

                    if (operand->getType()->isIntegerTy(64))
                    {
                        // Native int64: just negate
                        result = builder.CreateNeg(operand, "neg");
                    }
                    else
                    {
                        // PyObject*: use PyNumber_Negative
                        result = builder.CreateCall(py_number_negative_func, {operand});
                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }

                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_INVERT)
            {
                // Implements STACK[-1] = ~STACK[-1] (bitwise NOT)
                if (!stack.empty())
                {
                    llvm::Value *operand = stack.back();
                    stack.pop_back();
                    llvm::Value *result = nullptr;

                    if (operand->getType()->isIntegerTy(64))
                    {
                        // Native int64: use XOR with -1 for bitwise NOT
                        result = builder.CreateXor(operand, llvm::ConstantInt::get(i64_type, -1), "invert");
                    }
                    else
                    {
                        // PyObject*: use PyNumber_Invert
                        result = builder.CreateCall(py_number_invert_func, {operand});
                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }

                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_NOT)
            {
                // Implements STACK[-1] = not STACK[-1] (logical NOT, returns True/False)
                if (!stack.empty())
                {
                    llvm::Value *operand = stack.back();
                    stack.pop_back();
                    llvm::Value *result = nullptr;

                    if (operand->getType()->isIntegerTy(64))
                    {
                        // Native int64: compare to zero, invert result
                        llvm::Value *is_zero = builder.CreateICmpEQ(operand, llvm::ConstantInt::get(i64_type, 0), "iszero");
                        result = builder.CreateZExt(is_zero, i64_type, "not");
                    }
                    else
                    {
                        // PyObject*: use PyObject_Not
                        // Returns 0 if object is truthy, 1 if falsy, -1 on error
                        llvm::Value *not_result = builder.CreateCall(py_object_not_func, {operand}, "not");

                        // Convert result to Py_True or Py_False
                        // If not_result == 1, return Py_True; else return Py_False
                        llvm::Value *is_true = builder.CreateICmpEQ(not_result, llvm::ConstantInt::get(builder.getInt32Ty(), 1), "is_true");

                        llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                        result = builder.CreateSelect(is_true, py_true, py_false, "not_result");

                        // Incref the result (Py_True/Py_False are immortal in 3.12+, but be safe)
                        builder.CreateCall(py_incref_func, {result});

                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }

                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::TO_BOOL)
            {
                // Convert TOS to a boolean value - used before conditionals
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    llvm::Value *result = nullptr;

                    if (val->getType()->isIntegerTy(64))
                    {
                        // Native int64: compare != 0 to get boolean, then convert to Py_True/Py_False
                        llvm::Value *is_nonzero = builder.CreateICmpNE(val, llvm::ConstantInt::get(i64_type, 0), "nonzero");

                        llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                        result = builder.CreateSelect(is_nonzero, py_true, py_false, "tobool_result");
                        builder.CreateCall(py_incref_func, {result});
                    }
                    else
                    {
                        // PyObject*: use PyObject_IsTrue to get boolean, then return Py_True/Py_False
                        llvm::Value *is_true = builder.CreateCall(py_object_istrue_func, {val}, "istrue");
                        llvm::Value *is_nonzero = builder.CreateICmpNE(is_true, llvm::ConstantInt::get(builder.getInt32Ty(), 0), "nonzero");

                        llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                        result = builder.CreateSelect(is_nonzero, py_true, py_false, "tobool_result");
                        builder.CreateCall(py_incref_func, {result});

                        // Decref the original value
                        builder.CreateCall(py_decref_func, {val});
                    }

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::NOP)
            {
                // Do nothing - this is a no-operation instruction
            }
            else if (instr.opcode == op::EXTENDED_ARG)
            {
                // EXTENDED_ARG is handled by Python's dis module transparently
                // The combined argument value is already in the following instruction's arg field
                // This opcode is a no-op in our JIT compiler
            }
            else if (instr.opcode == op::LOAD_ASSERTION_ERROR)
            {
                // LOAD_ASSERTION_ERROR: Push AssertionError exception class onto stack
                // Used by assert statements
                llvm::Value *assertion_error = builder.CreateIntToPtr(
                    llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(PyExc_AssertionError)),
                    ptr_type);
                stack.push_back(assertion_error);
            }
            else if (instr.opcode == op::CALL_INTRINSIC_1)
            {
                // CALL_INTRINSIC_1: Calls intrinsic function with one argument
                // Intrinsic function codes from Python's pycore_intrinsics.h:
                // 1: PRINT (internal debug)
                // 2: IMPORT_STAR (from x import *)
                // 3: STOPITERATION_ERROR
                // 4: ASYNC_GEN_WRAP
                // 5: UNARY_POSITIVE (+x)
                // 6: LIST_TO_TUPLE
                // 7-11: Type-related intrinsics (TypeVar, ParamSpec, etc.)
                if (stack.size() >= 1)
                {
                    llvm::Value *operand = stack.back();
                    stack.pop_back();

                    llvm::Value *result = nullptr;
                    
                    switch (instr.arg)
                    {
                    case 1: // INTRINSIC_PRINT
                    {
                        // Debug print - call PyObject_Print(obj, stdout, 0)
                        // This is used for debugging, just print and return None
                        llvm::FunctionType *print_type = llvm::FunctionType::get(
                            builder.getInt32Ty(), {ptr_type, ptr_type, builder.getInt32Ty()}, false);
                        llvm::FunctionCallee print_func = module->getOrInsertFunction(
                            "PyObject_Print", print_type);
                        // Get stdout - use __acrt_iob_func(1) on Windows, or stdout symbol on Unix
                        // For simplicity, we'll just use PyObject_Repr and print to stderr
                        // Actually, let's call sys.stdout.write which is more portable
                        // For now, just decref and return None - print intrinsic is rarely critical
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {result});
                        break;
                    }
                    case 3: // INTRINSIC_STOPITERATION_ERROR
                    {
                        // Used to handle StopIteration in generators
                        // Just consume the value - error already raised
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        // Push None as placeholder
                        llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {result});
                        break;
                    }
                    case 4: // INTRINSIC_ASYNC_GEN_WRAP
                    {
                        // Wrap value for async generator - calls _PyAsyncGenValueWrapperNew
                        // This is an internal CPython function, so we use PyObject_Call approach
                        // For now, just return the value as-is since we're not fully supporting async gen
                        result = operand;  // Transfer ownership
                        break;
                    }
                    case 5: // INTRINSIC_UNARY_POSITIVE
                    {
                        // Implements unary + operator
                        result = builder.CreateCall(py_number_positive_func, {operand});
                        // Check for NULL (error)
                        llvm::Value *is_null = builder.CreateIsNull(result);
                        llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                            *local_context, "intrinsic_error_" + std::to_string(i), func);
                        llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                            *local_context, "intrinsic_continue_" + std::to_string(i), func);
                        builder.CreateCondBr(is_null, error_block, continue_block);

                        builder.SetInsertPoint(error_block);
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                        builder.SetInsertPoint(continue_block);
                        // Decref input operand
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        break;
                    }
                    case 6: // INTRINSIC_LIST_TO_TUPLE
                    {
                        // Convert list to tuple - PyList_AsTuple
                        llvm::FunctionType *list_as_tuple_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee list_as_tuple_func = module->getOrInsertFunction(
                            "PyList_AsTuple", list_as_tuple_type);
                        result = builder.CreateCall(list_as_tuple_func, {operand});
                        // Decref the list
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        // Check for error
                        check_error_and_branch(current_offset, result, "list_to_tuple");
                        break;
                    }
                    case 10: // INTRINSIC_SUBSCRIPT_GENERIC
                    {
                        // Implements Generic[T] type subscripting for type hints
                        // Pops (class, args) - args is the subscript, class is the type being subscripted
                        // Actually: operand is a tuple (class, item) that was built before the intrinsic
                        // Needs to call __class_getitem__ on the class with the item
                        // Uses PyObject_GetItem to implement X[Y] for type subscripting
                        llvm::FunctionType *get_item_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee get_item_func = module->getOrInsertFunction(
                            "PyObject_GetItem", get_item_type);
                        
                        // operand is a tuple of (origin, args)
                        // We need to unpack and call origin[args]
                        llvm::FunctionType *tuple_get_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, i64_type}, false);
                        llvm::FunctionCallee tuple_get_func = module->getOrInsertFunction(
                            "PyTuple_GetItem", tuple_get_type);  // Borrowed reference!
                        
                        // Get origin (index 0)
                        llvm::Value *origin = builder.CreateCall(tuple_get_func, 
                            {operand, llvm::ConstantInt::get(i64_type, 0)});
                        // Get args (index 1)
                        llvm::Value *args = builder.CreateCall(tuple_get_func,
                            {operand, llvm::ConstantInt::get(i64_type, 1)});
                        
                        // Call origin[args]
                        result = builder.CreateCall(get_item_func, {origin, args});
                        
                        // Decref the tuple (PyTuple_GetItem returns borrowed refs)
                        builder.CreateCall(py_decref_func, {operand});
                        
                        // Check for error
                        check_error_and_branch(current_offset, result, "subscript_generic");
                        break;
                    }
                    case 7: // INTRINSIC_TYPEVAR
                    {
                        // Creates a TypeVar from its arguments
                        // operand is a tuple: (name, *constraints, bound=None, covariant=False, contravariant=False, infer_variance=False)
                        // In Python 3.12+, this calls typing.TypeVar with unpacked args
                        // We import typing.TypeVar and call it
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        // Import typing module
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        // Get TypeVar class
                        llvm::Value *typevar_name = builder.CreateGlobalStringPtr("TypeVar");
                        llvm::Value *typevar_class = builder.CreateCall(getattr_func, {typing_mod, typevar_name});
                        
                        // Call TypeVar(*args) where args is the operand tuple
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typevar_class, operand, kwargs});
                        
                        // Cleanup
                        builder.CreateCall(py_decref_func, {typevar_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {operand});
                        
                        check_error_and_branch(current_offset, result, "typevar");
                        break;
                    }
                    case 8: // INTRINSIC_PARAMSPEC
                    {
                        // Creates a ParamSpec - operand is tuple (name, bound=None, covariant=False, contravariant=False)
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        // Import typing module
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        // Get ParamSpec class
                        llvm::Value *paramspec_name = builder.CreateGlobalStringPtr("ParamSpec");
                        llvm::Value *paramspec_class = builder.CreateCall(getattr_func, {typing_mod, paramspec_name});
                        
                        // Call ParamSpec(*args)
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {paramspec_class, operand, kwargs});
                        
                        // Cleanup
                        builder.CreateCall(py_decref_func, {paramspec_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {operand});
                        
                        check_error_and_branch(current_offset, result, "paramspec");
                        break;
                    }
                    case 9: // INTRINSIC_TYPEVARTUPLE
                    {
                        // Creates a TypeVarTuple - operand is tuple (name,)
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        // Import typing module
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        // Get TypeVarTuple class
                        llvm::Value *typevartuple_name = builder.CreateGlobalStringPtr("TypeVarTuple");
                        llvm::Value *typevartuple_class = builder.CreateCall(getattr_func, {typing_mod, typevartuple_name});
                        
                        // Call TypeVarTuple(*args)
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typevartuple_class, operand, kwargs});
                        
                        // Cleanup
                        builder.CreateCall(py_decref_func, {typevartuple_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {operand});
                        
                        check_error_and_branch(current_offset, result, "typevartuple");
                        break;
                    }
                    case 11: // INTRINSIC_TYPEALIAS
                    {
                        // Creates a TypeAlias - operand is tuple (name, type_params, value)
                        // In Python 3.12+, this creates a TypeAliasType
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        // Import typing module
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        // Get TypeAliasType class
                        llvm::Value *typealias_name = builder.CreateGlobalStringPtr("TypeAliasType");
                        llvm::Value *typealias_class = builder.CreateCall(getattr_func, {typing_mod, typealias_name});
                        
                        // Call TypeAliasType(*args)
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typealias_class, operand, kwargs});
                        
                        // Cleanup
                        builder.CreateCall(py_decref_func, {typealias_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {operand});
                        
                        check_error_and_branch(current_offset, result, "typealias");
                        break;
                    }
                    case 2: // INTRINSIC_IMPORT_STAR
                    {
                        // from module import * - most complex intrinsic
                        // operand is the module object to import from
                        // We need to:
                        // 1. Get the module's __dict__ (or __all__ if present)
                        // 2. Get the current locals dictionary
                        // 3. Merge items from module into locals
                        
                        // Get frame and locals
                        llvm::FunctionType *get_frame_type = llvm::FunctionType::get(
                            ptr_type, {}, false);
                        llvm::FunctionCallee get_frame_func = module->getOrInsertFunction(
                            "PyEval_GetFrame", get_frame_type);
                        
                        llvm::FunctionType *frame_get_locals_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee frame_get_locals_func = module->getOrInsertFunction(
                            "PyFrame_GetLocals", frame_get_locals_type);
                        
                        // PyImport_ImportStar helper does the actual work
                        // But it's internal, so we need to implement manually
                        // Use PyObject_Dir to get names, then getattr/setitem
                        
                        // Get module's __dict__
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        // Check for __all__ first
                        llvm::Value *all_name = builder.CreateGlobalStringPtr("__all__");
                        llvm::Value *all_list = builder.CreateCall(getattr_func, {operand, all_name});
                        
                        // Get frame locals
                        llvm::Value *frame = builder.CreateCall(get_frame_func, {});
                        llvm::Value *locals = builder.CreateCall(frame_get_locals_func, {frame});
                        
                        // Use PyDict_Merge to merge module dict into locals
                        llvm::FunctionType *dict_merge_type = llvm::FunctionType::get(
                            builder.getInt32Ty(), {ptr_type, ptr_type, builder.getInt32Ty()}, false);
                        llvm::FunctionCallee dict_merge_func = module->getOrInsertFunction(
                            "PyDict_Merge", dict_merge_type);
                        
                        // Get module __dict__
                        llvm::Value *dict_name = builder.CreateGlobalStringPtr("__dict__");
                        llvm::Value *mod_dict = builder.CreateCall(getattr_func, {operand, dict_name});
                        
                        // Merge (override=1 means replace existing keys)
                        builder.CreateCall(dict_merge_func, 
                            {locals, mod_dict, llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
                        
                        // Cleanup
                        builder.CreateCall(py_decref_func, {mod_dict});
                        builder.CreateCall(py_decref_func, {locals});
                        builder.CreateCall(py_decref_func, {operand});
                        
                        // Clear any error from __all__ not existing
                        llvm::FunctionType *err_clear_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context), {}, false);
                        llvm::FunctionCallee err_clear_func = module->getOrInsertFunction(
                            "PyErr_Clear", err_clear_type);
                        builder.CreateCall(err_clear_func, {});
                        
                        // Return None
                        llvm::Value *py_none_ptr_ret = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(py_none_ptr_ret, ptr_type);
                        builder.CreateCall(py_incref_func, {result});
                        break;
                    }
                    default:
                    {
                        // Unknown intrinsic - raise error
                        if (operand->getType()->isPointerTy())
                        {
                            builder.CreateCall(py_decref_func, {operand});
                        }
                        llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context),
                            {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                            "PyErr_SetString", py_err_set_str_type);
                        llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(PyExc_SystemError));
                        llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                        llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported intrinsic function");
                        builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                        // Return false to indicate JIT failed
                        return false;
                    }
                    }
                    
                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::COMPARE_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back();
                    stack.pop_back();
                    llvm::Value *lhs = stack.back();
                    stack.pop_back();

                    // Python 3.13 encoding: (op_code << 5) | flags
                    // Extraction: op_code = arg >> 5
                    // Compare operations: 0=<, 1=<=, 2===, 3=!=, 4=>, 5=>=
                    int op_code = instr.arg >> 5;
                    llvm::Value *cmp_result = nullptr;

                    // Check if either operand is a pointer (PyObject*)
                    bool lhs_is_ptr = lhs->getType()->isPointerTy();
                    bool rhs_is_ptr = rhs->getType()->isPointerTy();

                    // Prepare Py_True and Py_False pointers for result
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                    if (lhs_is_ptr || rhs_is_ptr)
                    {
                        // At least one operand is PyObject* - use PyObject_RichCompareBool
                        // Track if we boxed operands
                        bool lhs_boxed = false;
                        bool rhs_boxed = false;

                        // First, ensure both are PyObject*
                        if (!lhs_is_ptr)
                        {
                            // Convert i64 to PyObject*
                            lhs = builder.CreateCall(py_long_fromlonglong_func, {lhs});
                            lhs_boxed = true;
                        }
                        if (!rhs_is_ptr)
                        {
                            // Convert i64 to PyObject*
                            rhs = builder.CreateCall(py_long_fromlonglong_func, {rhs});
                            rhs_boxed = true;
                        }

                        // Map our op_code to Python's comparison opid
                        // Our encoding: 0=<, 1=<=, 2===, 3=!=, 4=>, 5=>=
                        // Python opid: Py_LT=0, Py_LE=1, Py_EQ=2, Py_NE=3, Py_GT=4, Py_GE=5
                        // They match directly
                        llvm::Value *opid = llvm::ConstantInt::get(builder.getInt32Ty(), op_code);

                        // Call PyObject_RichCompareBool - returns int (0=false, 1=true, -1=error)
                        llvm::Value *result = builder.CreateCall(py_object_richcompare_bool_func, {lhs, rhs, opid});

                        // Decref boxed temporaries and consumed PyObject* operands
                        if (lhs_boxed)
                        {
                            builder.CreateCall(py_decref_func, {lhs});
                        }
                        else
                        {
                            builder.CreateCall(py_decref_func, {lhs});
                        }
                        if (rhs_boxed)
                        {
                            builder.CreateCall(py_decref_func, {rhs});
                        }
                        else
                        {
                            builder.CreateCall(py_decref_func, {rhs});
                        }

                        // Convert to Py_True/Py_False (Bug #2 fix)
                        llvm::Value *is_true = builder.CreateICmpSGT(result, llvm::ConstantInt::get(builder.getInt32Ty(), 0));
                        cmp_result = builder.CreateSelect(is_true, py_true, py_false);
                        builder.CreateCall(py_incref_func, {cmp_result});
                    }
                    else
                    {
                        // Both are i64 - use native integer comparison
                        llvm::Value *bool_result = nullptr;
                        switch (op_code)
                        {
                        case 0: // <
                            bool_result = builder.CreateICmpSLT(lhs, rhs, "lt");
                            break;
                        case 1: // <=
                            bool_result = builder.CreateICmpSLE(lhs, rhs, "le");
                            break;
                        case 2: // ==
                            bool_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                            break;
                        case 3: // !=
                            bool_result = builder.CreateICmpNE(lhs, rhs, "ne");
                            break;
                        case 4: // >
                            bool_result = builder.CreateICmpSGT(lhs, rhs, "gt");
                            break;
                        case 5: // >=
                            bool_result = builder.CreateICmpSGE(lhs, rhs, "ge");
                            break;
                        default:
                            bool_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                            break;
                        }
                        // Convert to Py_True/Py_False (Bug #2 fix)
                        cmp_result = builder.CreateSelect(bool_result, py_true, py_false);
                        builder.CreateCall(py_incref_func, {cmp_result});
                    }

                    if (cmp_result)
                    {
                        stack.push_back(cmp_result);
                    }
                }
            }
            else if (instr.opcode == op::CONTAINS_OP)
            {
                // Implements 'in' / 'not in' test
                // Stack: TOS=container, TOS1=value
                // arg & 1: 0 = 'in', 1 = 'not in'
                if (stack.size() >= 2)
                {
                    llvm::Value *container = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    bool invert = (instr.arg & 1) != 0;

                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool container_is_ptr = container->getType()->isPointerTy();

                    // Convert int64 value to PyObject* if needed
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    if (value_was_boxed)
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                    }

                    // PySequence_Contains returns 1 if contains, 0 if not, -1 on error
                    llvm::Value *result = builder.CreateCall(py_sequence_contains_func, {container, value}, "contains");

                    if (invert)
                    {
                        // 'not in': invert the result (1->0, 0->1)
                        result = builder.CreateXor(result, llvm::ConstantInt::get(result->getType(), 1), "not_in");
                    }

                    // Decref consumed operands
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    if (container_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {container});
                    }

                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *is_true = builder.CreateICmpSGT(result, llvm::ConstantInt::get(result->getType(), 0));
                    llvm::Value *bool_result = builder.CreateSelect(is_true, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::IS_OP)
            {
                // Implements 'is' / 'is not' identity test
                // Stack: TOS=rhs, TOS1=lhs
                // arg & 1: 0 = 'is', 1 = 'is not'
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back();
                    stack.pop_back();
                    llvm::Value *lhs = stack.back();
                    stack.pop_back();
                    bool invert = (instr.arg & 1) != 0;

                    bool lhs_is_ptr = lhs->getType()->isPointerTy();
                    bool rhs_is_ptr = rhs->getType()->isPointerTy();
                    bool lhs_was_boxed = false;
                    bool rhs_was_boxed = false;

                    // Ensure both are pointers for identity comparison
                    if (lhs->getType()->isIntegerTy(64))
                    {
                        lhs = builder.CreateCall(py_long_fromlonglong_func, {lhs});
                        lhs_was_boxed = true;
                    }
                    if (rhs->getType()->isIntegerTy(64))
                    {
                        rhs = builder.CreateCall(py_long_fromlonglong_func, {rhs});
                        rhs_was_boxed = true;
                    }

                    // Pointer identity comparison
                    llvm::Value *is_same = builder.CreateICmpEQ(lhs, rhs, "is");

                    if (invert)
                    {
                        is_same = builder.CreateNot(is_same, "is_not");
                    }

                    // Decref consumed operands
                    if (lhs_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {lhs});
                    }
                    else if (lhs_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {lhs});
                    }
                    if (rhs_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {rhs});
                    }
                    else if (rhs_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {rhs});
                    }

                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *bool_result = builder.CreateSelect(is_same, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== Pattern Matching Opcodes ==========
            else if (instr.opcode == op::MATCH_MAPPING)
            {
                // MATCH_MAPPING: Test if TOS is a mapping (dict-like)
                // Pushes True if isinstance(TOS, collections.abc.Mapping), False otherwise
                // TOS remains on stack, result is pushed on top
                if (!stack.empty())
                {
                    llvm::Value *subject = stack.back();
                    // Don't pop - MATCH_MAPPING leaves subject on stack

                    // Box if needed
                    bool subject_was_boxed = false;
                    if (subject->getType()->isIntegerTy(64))
                    {
                        subject = builder.CreateCall(py_long_fromlonglong_func, {subject});
                        subject_was_boxed = true;
                        // Replace top of stack with boxed version
                        stack.back() = subject;
                        builder.CreateCall(py_incref_func, {subject});
                    }

                    // PyMapping_Check returns 1 if object has mapping protocol
                    llvm::FunctionType *py_mapping_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_mapping_check_func = module->getOrInsertFunction(
                        "PyMapping_Check", py_mapping_check_type);
                    llvm::Value *is_mapping = builder.CreateCall(py_mapping_check_func, {subject}, "is_mapping");

                    // Convert to Py_True/Py_False
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *is_true = builder.CreateICmpNE(is_mapping, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*local_context), 0));
                    llvm::Value *bool_result = builder.CreateSelect(is_true, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::MATCH_SEQUENCE)
            {
                // MATCH_SEQUENCE: Test if TOS is a sequence (list/tuple-like, but NOT str/bytes/bytearray)
                // Pushes True if it's a sequence, False otherwise
                // TOS remains on stack, result is pushed on top
                if (!stack.empty())
                {
                    llvm::Value *subject = stack.back();
                    // Don't pop - MATCH_SEQUENCE leaves subject on stack

                    // Box if needed
                    bool subject_was_boxed = false;
                    if (subject->getType()->isIntegerTy(64))
                    {
                        subject = builder.CreateCall(py_long_fromlonglong_func, {subject});
                        subject_was_boxed = true;
                        stack.back() = subject;
                        builder.CreateCall(py_incref_func, {subject});
                    }

                    // For pattern matching, we need to check:
                    // 1. PySequence_Check(obj) is true
                    // 2. NOT isinstance(obj, (str, bytes, bytearray))
                    // Use _PySequence_IterSearch internal API or manual check

                    // Check if it's a sequence
                    llvm::FunctionType *py_sequence_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_sequence_check_func = module->getOrInsertFunction(
                        "PySequence_Check", py_sequence_check_type);
                    llvm::Value *is_sequence = builder.CreateCall(py_sequence_check_func, {subject}, "is_sequence");

                    // Check if it's a string (must exclude)
                    llvm::FunctionType *py_unicode_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_unicode_check_func = module->getOrInsertFunction(
                        "PyUnicode_Check", py_unicode_check_type);
                    llvm::Value *is_unicode = builder.CreateCall(py_unicode_check_func, {subject}, "is_unicode");

                    // Check if it's bytes
                    llvm::FunctionType *py_bytes_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_bytes_check_func = module->getOrInsertFunction(
                        "PyBytes_Check", py_bytes_check_type);
                    llvm::Value *is_bytes = builder.CreateCall(py_bytes_check_func, {subject}, "is_bytes");

                    // Check if it's bytearray
                    llvm::FunctionType *py_bytearray_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_bytearray_check_func = module->getOrInsertFunction(
                        "PyByteArray_Check", py_bytearray_check_type);
                    llvm::Value *is_bytearray = builder.CreateCall(py_bytearray_check_func, {subject}, "is_bytearray");

                    // Result = is_sequence && !is_unicode && !is_bytes && !is_bytearray
                    llvm::Value *zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*local_context), 0);
                    llvm::Value *seq_ok = builder.CreateICmpNE(is_sequence, zero, "seq_ok");
                    llvm::Value *not_unicode = builder.CreateICmpEQ(is_unicode, zero, "not_unicode");
                    llvm::Value *not_bytes = builder.CreateICmpEQ(is_bytes, zero, "not_bytes");
                    llvm::Value *not_bytearray = builder.CreateICmpEQ(is_bytearray, zero, "not_bytearray");
                    llvm::Value *result = builder.CreateAnd(seq_ok, not_unicode);
                    result = builder.CreateAnd(result, not_bytes);
                    result = builder.CreateAnd(result, not_bytearray);

                    // Convert to Py_True/Py_False
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *bool_result = builder.CreateSelect(result, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::MATCH_KEYS)
            {
                // MATCH_KEYS: Extract values from a mapping for given keys
                // Stack: TOS=keys (tuple), TOS1=subject (mapping)
                // Result: push values tuple (or None if key missing)
                // Subject and keys remain on stack below the result
                // Note: Python 3.11+ no longer pushes a separate boolean
                if (stack.size() >= 2)
                {
                    llvm::Value *keys = stack.back();
                    // Don't pop - keys stay on stack
                    llvm::Value *subject = stack[stack.size() - 2];
                    // Don't pop - subject stays on stack

                    // Box if needed
                    if (keys->getType()->isIntegerTy(64))
                    {
                        keys = builder.CreateCall(py_long_fromlonglong_func, {keys});
                        builder.CreateCall(py_incref_func, {keys});
                        stack.back() = keys;
                    }
                    if (subject->getType()->isIntegerTy(64))
                    {
                        subject = builder.CreateCall(py_long_fromlonglong_func, {subject});
                        builder.CreateCall(py_incref_func, {subject});
                        stack[stack.size() - 2] = subject;
                    }

                    // Call helper: PyObject* _PyJIT_MatchKeys(PyObject* subject, PyObject* keys)
                    // Returns tuple of values if all keys found, or None if any key missing
                    // The helper handles incref on the result
                    llvm::FunctionType *match_keys_helper_type = llvm::FunctionType::get(
                        ptr_type, {ptr_type, ptr_type}, false);
                    llvm::FunctionCallee match_keys_helper = module->getOrInsertFunction(
                        "_PyJIT_MatchKeys", match_keys_helper_type);
                    llvm::Value *result = builder.CreateCall(match_keys_helper, {subject, keys}, "match_keys_result");

                    // The helper returns either a tuple (success) or None (failure)
                    // Both are valid PyObject* with proper refcount, so just push it
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::MATCH_CLASS)
            {
                // MATCH_CLASS: Match against a class pattern
                // TOS = tuple of keyword attribute names
                // TOS1 = class to match against  
                // TOS2 = subject
                // arg = number of positional sub-patterns
                // Result: pops names, cls; pushes attrs tuple (or None if no match)
                // Subject remains on stack below the result
                if (stack.size() >= 3)
                {
                    llvm::Value *names = stack.back();
                    stack.pop_back();
                    llvm::Value *cls = stack.back();
                    stack.pop_back();
                    llvm::Value *subject = stack.back();
                    // Don't pop subject - it stays on stack

                    int nargs = instr.arg; // Number of positional patterns

                    // Box if needed
                    if (names->getType()->isIntegerTy(64))
                    {
                        names = builder.CreateCall(py_long_fromlonglong_func, {names});
                        builder.CreateCall(py_incref_func, {names});
                    }
                    if (cls->getType()->isIntegerTy(64))
                    {
                        cls = builder.CreateCall(py_long_fromlonglong_func, {cls});
                        builder.CreateCall(py_incref_func, {cls});
                    }
                    if (subject->getType()->isIntegerTy(64))
                    {
                        subject = builder.CreateCall(py_long_fromlonglong_func, {subject});
                        builder.CreateCall(py_incref_func, {subject});
                        stack.back() = subject;
                    }

                    // Call helper: PyObject* _PyJIT_MatchClass(subject, cls, nargs, names)
                    // Returns tuple of matched attributes if successful, Py_None (incref'd) otherwise
                    llvm::FunctionType *match_class_helper_type = llvm::FunctionType::get(
                        ptr_type, {ptr_type, ptr_type, llvm::Type::getInt32Ty(*local_context), ptr_type}, false);
                    llvm::FunctionCallee match_class_helper = module->getOrInsertFunction(
                        "_PyJIT_MatchClass", match_class_helper_type);
                    llvm::Value *nargs_val = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*local_context), nargs);
                    llvm::Value *result = builder.CreateCall(match_class_helper, {subject, cls, nargs_val, names}, "match_class_result");

                    // Push the result (either tuple or None)
                    stack.push_back(result);

                    // Decref the consumed values (names, cls were popped)
                    builder.CreateCall(py_decref_func, {names});
                    builder.CreateCall(py_decref_func, {cls});
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                if (!stack.empty() && i + 1 < instructions.size())
                {
                    llvm::Value *cond = stack.back();
                    stack.pop_back();

                    llvm::Value *bool_cond = nullptr;

                    // Handle different condition types
                    if (cond->getType()->isPointerTy())
                    {
                        // PyObject* - use PyObject_IsTrue for proper Python truthiness
                        // Returns 1 if true, 0 if false, -1 on error
                        llvm::Value *istrue_result = builder.CreateCall(py_object_istrue_func, {cond}, "istrue");
                        // Compare result > 0 (1 means true, 0 means false, -1 means error)
                        bool_cond = builder.CreateICmpSGT(
                            istrue_result,
                            llvm::ConstantInt::get(istrue_result->getType(), 0),
                            "tobool_obj");
                    }
                    else
                    {
                        // Integer - compare to zero
                        bool_cond = builder.CreateICmpNE(
                            cond,
                            llvm::ConstantInt::get(cond->getType(), 0),
                            "tobool");
                    }

                    int target_offset = instr.argval; // Use Python's calculated target
                    int next_offset = instructions[i + 1].offset;

                    if (!jump_targets.count(target_offset))
                    {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(target_offset), func);
                    }

                    // Create block for fall-through only if next instruction is also a jump target
                    if (!jump_targets.count(next_offset))
                    {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(next_offset), func);
                    }

                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        llvm::BasicBlock *current_block = builder.GetInsertBlock();

                        // Bug #1 Fix: Record stack state for BOTH branch targets
                        // This enables PHI node insertion at merge points
                        BlockStackState state;
                        state.stack = stack;
                        state.predecessor = current_block;
                        block_incoming_stacks[target_offset].push_back(state);
                        block_incoming_stacks[next_offset].push_back(state);

                        // POP_JUMP_IF_FALSE: jump if condition is FALSE (0), continue if TRUE (non-zero)
                        // POP_JUMP_IF_TRUE: jump if condition is TRUE (non-zero), continue if FALSE (0)
                        if (instr.opcode == op::POP_JUMP_IF_FALSE)
                        {
                            // Jump to target when condition is false (0), continue to next when true (non-zero)
                            builder.CreateCondBr(bool_cond, jump_targets[next_offset], jump_targets[target_offset]);
                        }
                        else
                        { // POP_JUMP_IF_TRUE (opcode 100)
                            // Jump to target when condition is true (non-zero), continue to next when false (0)
                            builder.CreateCondBr(bool_cond, jump_targets[target_offset], jump_targets[next_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_NONE || instr.opcode == op::POP_JUMP_IF_NOT_NONE)
            {
                // Jump based on whether value is None
                if (!stack.empty() && i + 1 < instructions.size())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // Get Python's Py_None singleton address
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);

                    // Compare pointer to Py_None
                    llvm::Value *is_none = builder.CreateICmpEQ(val, py_none, "is_none");

                    // Decref the value we popped (it's consumed)
                    if (val->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {val});
                    }

                    int target_offset = instr.argval;
                    int next_offset = instructions[i + 1].offset;

                    if (!jump_targets.count(target_offset))
                    {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(target_offset), func);
                    }
                    if (!jump_targets.count(next_offset))
                    {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(next_offset), func);
                    }

                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        llvm::BasicBlock *current_block = builder.GetInsertBlock();

                        // Bug #1 Fix: Record stack state for BOTH branch targets
                        BlockStackState state;
                        state.stack = stack;
                        state.predecessor = current_block;
                        block_incoming_stacks[target_offset].push_back(state);
                        block_incoming_stacks[next_offset].push_back(state);

                        if (instr.opcode == op::POP_JUMP_IF_NONE)
                        {
                            // Jump if is_none is true
                            builder.CreateCondBr(is_none, jump_targets[target_offset], jump_targets[next_offset]);
                        }
                        else
                        { // POP_JUMP_IF_NOT_NONE
                            // Jump if is_none is false (i.e., not None)
                            builder.CreateCondBr(is_none, jump_targets[next_offset], jump_targets[target_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                // For loops: jump backward to loop header
                int target_offset = instr.argval; // Use Python's calculated target
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_header_" + std::to_string(target_offset), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }

                // CRITICAL: Reset compile-time stack to match the target's expected depth
                // This is necessary because the loop body may have pushed values that
                // don't exist at the loop header. The runtime stack is correct (bytecode
                // is verified), but our compile-time tracking gets out of sync.
                if (stack_depth_at_offset.count(target_offset))
                {
                    size_t target_depth = stack_depth_at_offset[target_offset];
                    while (stack.size() > target_depth)
                    {
                        stack.pop_back();
                    }
                }

                // CRITICAL FIX: After JUMP_BACKWARD, the following instructions
                // (typically END_FOR, POP_TOP) are unreachable via this path.
                // Create an unreachable block WITH A TERMINATOR to prevent
                // fall-through stack recording from corrupting after_loop blocks.
                llvm::BasicBlock *unreachable_block = llvm::BasicBlock::Create(
                    *local_context, "unreachable_after_jump_" + std::to_string(i), func);
                builder.SetInsertPoint(unreachable_block);
                // Add unreachable terminator so we don't record fall-through
                builder.CreateUnreachable();
                // Clear stack since code here is truly unreachable
                stack.clear();
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
                // Unconditional forward jump
                int target_offset = instr.argval; // Use Python's calculated target
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "jump_target_" + std::to_string(target_offset), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }
                // Create a new block for any code after the jump (unreachable but prevents issues)
                llvm::BasicBlock *after_jump = llvm::BasicBlock::Create(
                    *local_context, "after_jump_" + std::to_string(i), func);
                builder.SetInsertPoint(after_jump);
            }
            else if (instr.opcode == op::RETURN_CONST)
            {
                // Return a constant from co_consts without using stack
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    // Get the constant value and return as PyObject*
                    if (instr.arg < int_constants.size())
                    {
                        if (obj_constants[instr.arg] != nullptr)
                        {
                            // PyObject* constant
                            llvm::Value *const_ptr = llvm::ConstantInt::get(
                                i64_type,
                                reinterpret_cast<uint64_t>(obj_constants[instr.arg]));
                            llvm::Value *py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                            builder.CreateCall(py_incref_func, {py_obj});
                            builder.CreateRet(py_obj);
                        }
                        else
                        {
                            // int64 constant - convert to PyObject*
                            llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                            llvm::Value *py_obj = builder.CreateCall(py_long_fromlonglong_func, {const_val});
                            builder.CreateRet(py_obj);
                        }
                    }
                    else
                    {
                        // Fallback: return None
                        llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {py_none});
                        builder.CreateRet(py_none);
                    }
                }
            }
            else if (instr.opcode == op::RETURN_VALUE)
            {
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    if (!stack.empty())
                    {
                        llvm::Value *ret_val = stack.back();
                        // If returning i64, convert to PyObject*
                        if (ret_val->getType()->isIntegerTy(64))
                        {
                            ret_val = builder.CreateCall(py_long_fromlonglong_func, {ret_val});
                        }
                        builder.CreateRet(ret_val);
                    }
                    else
                    {
                        // Return None
                        llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {py_none});
                        builder.CreateRet(py_none);
                    }
                }
            }
            else if (instr.opcode == op::BUILD_LIST)
            {
                // arg is the number of items to pop from stack
                int count = instr.arg;

                // Create new list with PyList_New(count)
                llvm::Value *count_val = llvm::ConstantInt::get(i64_type, count);
                llvm::Value *new_list = builder.CreateCall(py_list_new_func, {count_val});

                // Pop items from stack and add to list (in reverse order)
                std::vector<llvm::Value *> items;
                std::vector<bool> item_is_ptr;
                for (int i = 0; i < count; ++i)
                {
                    if (!stack.empty())
                    {
                        llvm::Value *item = stack.back();
                        item_is_ptr.push_back(item->getType()->isPointerTy());
                        items.push_back(item);
                        stack.pop_back();
                    }
                }

                // Add items to list in correct order
                for (int i = count - 1; i >= 0; --i)
                {
                    llvm::Value *index_val = llvm::ConstantInt::get(i64_type, count - 1 - i);
                    llvm::Value *item = items[i];
                    bool was_ptr = item_is_ptr[i];

                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64))
                    {
                        item = builder.CreateCall(py_long_fromlonglong_func, {item});
                        // PyList_SetItem steals reference, so new PyLong is transferred
                    }
                    else
                    {
                        // PyList_SetItem steals reference, but stack values own their refs
                        // We need to incref so SetItem can steal, and stack value is released
                        // But since we're consuming the stack value, we just transfer ownership
                        // No incref needed - SetItem steals the ref we got from stack
                    }

                    // PyList_SetItem steals reference (transfers ownership)
                    builder.CreateCall(py_list_setitem_func, {new_list, index_val, item});
                }

                stack.push_back(new_list);
            }
            else if (instr.opcode == op::BUILD_TUPLE)
            {
                // arg is the number of items to pop from stack
                int count = instr.arg;

                // Create new tuple with PyTuple_New(count)
                llvm::Value *count_val = llvm::ConstantInt::get(i64_type, count);
                llvm::Value *new_tuple = builder.CreateCall(py_tuple_new_func, {count_val});

                // Pop items from stack (in reverse order)
                std::vector<llvm::Value *> items;
                for (int i = 0; i < count; ++i)
                {
                    if (!stack.empty())
                    {
                        items.push_back(stack.back());
                        stack.pop_back();
                    }
                }

                // Add items to tuple in correct order
                for (int i = count - 1; i >= 0; --i)
                {
                    llvm::Value *index_val = llvm::ConstantInt::get(i64_type, count - 1 - i);
                    llvm::Value *item = items[i];

                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64))
                    {
                        item = builder.CreateCall(py_long_fromlonglong_func, {item});
                        // PyTuple_SetItem steals reference, new PyLong is transferred
                    }
                    // For PyObject* from stack: PyTuple_SetItem steals the reference
                    // Since we consumed it from stack, we transfer ownership directly
                    // No incref needed - SetItem steals the ref we got from stack

                    // PyTuple_SetItem steals reference (transfers ownership)
                    builder.CreateCall(py_tuple_setitem_func, {new_tuple, index_val, item});
                }

                stack.push_back(new_tuple);
            }
            else if (instr.opcode == op::BUILD_MAP)
            {
                // Build a dictionary from arg key-value pairs
                // arg = number of key-value pairs (stack has 2*arg items)
                int count = instr.arg;

                // Create new empty dict
                llvm::Value *new_dict = builder.CreateCall(py_dict_new_func, {}, "new_dict");

                // Pop key-value pairs from stack (in reverse order)
                // Stack order: ... key1 value1 key2 value2 ... (TOS is last value)
                std::vector<std::pair<llvm::Value *, llvm::Value *>> pairs;
                std::vector<std::pair<bool, bool>> pairs_are_ptr;
                for (int i = 0; i < count; ++i)
                {
                    if (stack.size() >= 2)
                    {
                        llvm::Value *value = stack.back();
                        stack.pop_back();
                        llvm::Value *key = stack.back();
                        stack.pop_back();
                        pairs_are_ptr.push_back({key->getType()->isPointerTy(), value->getType()->isPointerTy()});
                        pairs.push_back({key, value});
                    }
                }

                // Add pairs to dict in correct order (reverse of how we popped)
                for (int i = count - 1; i >= 0; --i)
                {
                    llvm::Value *key = pairs[i].first;
                    llvm::Value *value = pairs[i].second;
                    bool key_is_ptr = pairs_are_ptr[i].first;
                    bool value_is_ptr = pairs_are_ptr[i].second;
                    bool key_was_boxed = false;
                    bool value_was_boxed = false;

                    // Convert int64 to PyObject* if needed
                    if (key->getType()->isIntegerTy(64))
                    {
                        key = builder.CreateCall(py_long_fromlonglong_func, {key});
                        key_was_boxed = true;
                    }
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        value_was_boxed = true;
                    }

                    // PyDict_SetItem does NOT steal references (it increfs both)
                    builder.CreateCall(py_dict_setitem_func, {new_dict, key, value});

                    // Decref our copies (SetItem already incref'd its own)
                    if (key_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    else if (key_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                }

                stack.push_back(new_dict);
            }
            else if (instr.opcode == op::BUILD_CONST_KEY_MAP)
            {
                // Build a dictionary from const key tuple and count values
                // Stack: value1, value2, ..., valueN, key_tuple (TOS)
                // arg = number of key-value pairs (count)
                int count = instr.arg;

                if (!stack.empty())
                {
                    // Pop the keys tuple from TOS
                    llvm::Value *keys_tuple = stack.back();
                    stack.pop_back();

                    // Pop count values from stack
                    std::vector<llvm::Value *> values;
                    for (int i = 0; i < count; ++i)
                    {
                        if (!stack.empty())
                        {
                            values.push_back(stack.back());
                            stack.pop_back();
                        }
                    }

                    // Create new empty dict
                    llvm::Value *new_dict = builder.CreateCall(py_dict_new_func, {}, "new_dict");

                    // Add pairs to dict - values are in reverse order of keys
                    for (int i = 0; i < count; ++i)
                    {
                        // Get key from tuple at index i
                        llvm::Value *idx = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value *idx_obj = builder.CreateCall(py_long_fromlonglong_func, {idx});
                        llvm::Value *key = builder.CreateCall(py_object_getitem_func, {keys_tuple, idx_obj});
                        builder.CreateCall(py_decref_func, {idx_obj}); // Free the temp index object

                        // Get corresponding value (values are in reverse order)
                        llvm::Value *value = values[count - 1 - i];

                        // Convert int64 to PyObject* if needed
                        if (value->getType()->isIntegerTy(64))
                        {
                            value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        }

                        // PyDict_SetItem does NOT steal references
                        builder.CreateCall(py_dict_setitem_func, {new_dict, key, value});

                        // Decref the key (PyObject_GetItem returns new reference)
                        builder.CreateCall(py_decref_func, {key});
                    }

                    // Decref the keys tuple (we're done with it)
                    builder.CreateCall(py_decref_func, {keys_tuple});

                    stack.push_back(new_dict);
                }
            }
            else if (instr.opcode == op::BUILD_SET)
            {
                // Build a set from arg items on stack
                int count = instr.arg;

                // Create new empty set (pass NULL for empty)
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                llvm::Value *new_set = builder.CreateCall(py_set_new_func, {null_ptr}, "new_set");

                // Pop items from stack (in reverse order)
                std::vector<llvm::Value *> items;
                std::vector<bool> items_are_ptr;
                for (int i = 0; i < count; ++i)
                {
                    if (!stack.empty())
                    {
                        llvm::Value *item = stack.back();
                        items_are_ptr.push_back(item->getType()->isPointerTy());
                        items.push_back(item);
                        stack.pop_back();
                    }
                }

                // Add items to set in correct order
                for (int i = count - 1; i >= 0; --i)
                {
                    llvm::Value *item = items[i];
                    bool item_is_ptr = items_are_ptr[i];
                    bool item_was_boxed = false;

                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64))
                    {
                        item = builder.CreateCall(py_long_fromlonglong_func, {item});
                        item_was_boxed = true;
                    }

                    // PySet_Add does NOT steal references (it increfs)
                    builder.CreateCall(py_set_add_func, {new_set, item});

                    // Decref our copy (SetAdd already incref'd its own)
                    if (item_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                    else if (item_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }

                stack.push_back(new_set);
            }
            else if (instr.opcode == op::LIST_EXTEND)
            {
                // Extends the list STACK[-i] with the iterable STACK.pop()
                // Used for list literals like [1, 2, 3, 4, 5] in Python 3.9+
                if (!stack.empty())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();

                    // arg tells us where the list is: STACK[-i]
                    int list_index = instr.arg;
                    if (list_index > 0 && static_cast<size_t>(list_index) <= stack.size())
                    {
                        // Get the list from stack position (0-indexed from end)
                        llvm::Value *list = stack[stack.size() - list_index];

                        // Call PyList_Extend(list, iterable) - returns 0 on success
                        builder.CreateCall(py_list_extend_func, {list, iterable});

                        // Decref the iterable (we consumed it)
                        if (!iterable->getType()->isIntegerTy(64))
                        {
                            builder.CreateCall(py_decref_func, {iterable});
                        }
                    }
                }
            }
            else if (instr.opcode == op::BINARY_SUBSCR)
            {
                // Implements container[key]
                if (stack.size() >= 2)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();

                    // Track if we need to decref the key (if we box it)
                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    bool key_is_ptr = key->getType()->isPointerTy();

                    // Convert int64 key to PyObject* if needed
                    if (key_was_boxed)
                    {
                        key = builder.CreateCall(py_long_fromlonglong_func, {key});
                    }

                    // PyObject_GetItem returns new reference
                    llvm::Value *result = builder.CreateCall(py_object_getitem_func, {container, key});

                    // Decrement key refcount - if we boxed it or if it was a PyObject* from stack
                    if (key_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    else if (key_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }

                    // CRITICAL: Decref the container we consumed from the stack
                    if (container->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {container});
                    }

                    // Bug #3 Fix: Check for index error or key error
                    check_error_and_branch(current_offset, result, "binary_subscr");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::BUILD_SLICE)
            {
                // Build a slice object
                // arg=2: slice(start, stop), arg=3: slice(start, stop, step)
                int argc = instr.arg;
                if (argc == 2 && stack.size() >= 2)
                {
                    llvm::Value *stop = stack.back();
                    stack.pop_back();
                    llvm::Value *start = stack.back();
                    stack.pop_back();

                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);

                    if (start_boxed)
                    {
                        start = builder.CreateCall(py_long_fromlonglong_func, {start});
                    }
                    if (stop_boxed)
                    {
                        stop = builder.CreateCall(py_long_fromlonglong_func, {stop});
                    }

                    // PySlice_New(start, stop, NULL)
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);

                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});

                    // Decref temporaries
                    if (start_boxed)
                        builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed)
                        builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {stop});

                    stack.push_back(slice);
                }
                else if (argc == 3 && stack.size() >= 3)
                {
                    llvm::Value *step = stack.back();
                    stack.pop_back();
                    llvm::Value *stop = stack.back();
                    stack.pop_back();
                    llvm::Value *start = stack.back();
                    stack.pop_back();

                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);
                    bool step_boxed = step->getType()->isIntegerTy(64);

                    if (start_boxed)
                    {
                        start = builder.CreateCall(py_long_fromlonglong_func, {start});
                    }
                    if (stop_boxed)
                    {
                        stop = builder.CreateCall(py_long_fromlonglong_func, {stop});
                    }
                    if (step_boxed)
                    {
                        step = builder.CreateCall(py_long_fromlonglong_func, {step});
                    }

                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, step});

                    // Decref temporaries
                    if (start_boxed)
                        builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed)
                        builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {stop});
                    if (step_boxed)
                        builder.CreateCall(py_decref_func, {step});
                    else if (step->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {step});

                    stack.push_back(slice);
                }
            }
            else if (instr.opcode == op::BINARY_SLICE)
            {
                // Implements TOS = TOS1[TOS2:TOS]  (container[start:stop])
                // Stack: TOS=stop, TOS1=start, TOS2=container
                if (stack.size() >= 3)
                {
                    llvm::Value *stop = stack.back();
                    stack.pop_back();
                    llvm::Value *start = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();

                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);

                    if (start_boxed)
                    {
                        start = builder.CreateCall(py_long_fromlonglong_func, {start});
                    }
                    if (stop_boxed)
                    {
                        stop = builder.CreateCall(py_long_fromlonglong_func, {stop});
                    }

                    // Build a slice object
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});

                    // Use PyObject_GetItem with the slice
                    llvm::Value *result = builder.CreateCall(py_object_getitem_func, {container, slice});

                    // Decref slice (consumed)
                    builder.CreateCall(py_decref_func, {slice});

                    // Decref temporaries and consumed values
                    if (start_boxed)
                        builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed)
                        builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {stop});
                    if (container->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {container});
                    }

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::STORE_SUBSCR)
            {
                // Implements container[key] = value
                // Per Python docs: key = STACK.pop(); container = STACK.pop(); value = STACK.pop()
                // Stack order: TOS=key, TOS1=container, TOS2=value
                if (stack.size() >= 3)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back(); // TOS
                    llvm::Value *container = stack.back();
                    stack.pop_back(); // TOS1
                    llvm::Value *value = stack.back();
                    stack.pop_back(); // TOS2

                    // Track if we need to decref (if we box values)
                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    bool key_is_ptr = key->getType()->isPointerTy();
                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool container_is_ptr = container->getType()->isPointerTy();

                    // Convert int64 key to PyObject* if needed
                    if (key_was_boxed)
                    {
                        key = builder.CreateCall(py_long_fromlonglong_func, {key});
                    }

                    // Convert int64 value to PyObject* if needed
                    if (value_was_boxed)
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                    }

                    // PyObject_SetItem(container, key, value) - returns 0 on success
                    builder.CreateCall(py_object_setitem_func, {container, key, value});

                    // Decrement temp refs if we created them
                    if (key_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    else if (key_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }

                    // CRITICAL: Decref container since we consumed it from stack
                    if (container_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {container});
                    }
                }
            }
            else if (instr.opcode == op::STORE_SLICE)
            {
                // Implements container[start:stop] = value
                // Stack: TOS=stop, TOS1=start, TOS2=container, TOS3=value
                if (stack.size() >= 4)
                {
                    llvm::Value *stop = stack.back();
                    stack.pop_back();
                    llvm::Value *start = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);

                    if (start_boxed)
                    {
                        start = builder.CreateCall(py_long_fromlonglong_func, {start});
                    }
                    if (stop_boxed)
                    {
                        stop = builder.CreateCall(py_long_fromlonglong_func, {stop});
                    }

                    // Build a slice object
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});

                    // PyObject_SetItem(container, slice, value)
                    builder.CreateCall(py_object_setitem_func, {container, slice, value});

                    // Decref slice
                    builder.CreateCall(py_decref_func, {slice});

                    // Decref temporaries
                    if (start_boxed)
                        builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed)
                        builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {stop});
                    if (container->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {container});
                    if (value->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {value});
                }
            }
            else if (instr.opcode == op::DELETE_SUBSCR)
            {
                // Implements del container[key]
                // Stack: TOS=key, TOS1=container
                if (stack.size() >= 2)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();

                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    if (key_was_boxed)
                    {
                        key = builder.CreateCall(py_long_fromlonglong_func, {key});
                    }

                    // PyObject_DelItem(container, key)
                    builder.CreateCall(py_object_delitem_func, {container, key});

                    // Decref
                    if (key_was_boxed)
                        builder.CreateCall(py_decref_func, {key});
                    else if (key->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {key});
                    if (container->getType()->isPointerTy())
                        builder.CreateCall(py_decref_func, {container});
                }
            }
            else if (instr.opcode == op::STORE_ATTR)
            {
                // Implements obj.attr = value
                // Stack order: TOS=obj, TOS1=value
                // Python 3.13: STORE_ATTR uses oparg directly as index into co_names
                int name_idx = instr.arg;

                if (stack.size() >= 2 && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back(); // TOS
                    llvm::Value *value = stack.back();
                    stack.pop_back(); // TOS1
                    bool value_is_ptr = value->getType()->isPointerTy();

                    // Get attribute name from names (PyUnicode string)
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    // Convert int64 value to PyObject* if needed
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    if (value_was_boxed)
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                    }

                    // PyObject_SetAttr(obj, attr_name, value) - returns 0 on success
                    builder.CreateCall(py_object_setattr_func, {obj, attr_name, value});

                    // Decref the value if we boxed it or it was a PyObject* from stack
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }

                    // CRITICAL: Decref the object we consumed from the stack
                    if (obj->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {obj});
                    }
                }
            }
            else if (instr.opcode == op::LIST_APPEND)
            {
                // Append TOS to the list at STACK[-i]
                // arg = i (distance from TOS, AFTER popping item)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    // Calculate list index BEFORE popping (stack.size() - 1 - i_val + 1 = stack.size() - i_val)
                    // But after popping item, list is at stack[stack.size() - i_val]
                    // Actually: TOS is item, list is at STACK[-(i+1)] before pop = STACK[-i] after pop
                    llvm::Value *item = stack.back();
                    stack.pop_back();
                    bool item_is_ptr = item->getType()->isPointerTy();
                    bool item_was_boxed = false;
                    // After pop, list is at distance i from new TOS, which is index (size - i)
                    llvm::Value *list = stack[stack.size() - i_val]; // List stays on stack

                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64))
                    {
                        item = builder.CreateCall(py_long_fromlonglong_func, {item});
                        item_was_boxed = true;
                    }

                    // PyList_Append does NOT steal references (it increfs)
                    builder.CreateCall(py_list_append_func, {list, item});

                    // Decref our copy since Append already incref'd its own
                    if (item_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                    else if (item_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }
            }
            else if (instr.opcode == op::LIST_EXTEND)
            {
                // Extend list at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();
                    bool iter_is_ptr = iterable->getType()->isPointerTy();
                    llvm::Value *list = stack[stack.size() - i_val]; // List stays on stack

                    // _PyList_Extend(list, iterable)
                    builder.CreateCall(py_list_extend_func, {list, iterable});

                    // Decref iterable since we consumed it
                    if (iter_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {iterable});
                    }
                }
            }
            else if (instr.opcode == op::SET_UPDATE)
            {
                // Update set at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();
                    bool iter_is_ptr = iterable->getType()->isPointerTy();
                    llvm::Value *set = stack[stack.size() - i_val]; // Set stays on stack

                    // _PySet_Update(set, iterable)
                    builder.CreateCall(py_set_update_func, {set, iterable});

                    // Decref iterable since we consumed it
                    if (iter_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {iterable});
                    }
                }
            }
            else if (instr.opcode == op::DICT_UPDATE)
            {
                // Update dict at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    llvm::Value *update_dict = stack.back();
                    stack.pop_back();
                    bool update_is_ptr = update_dict->getType()->isPointerTy();
                    llvm::Value *dict = stack[stack.size() - i_val]; // Dict stays on stack

                    // PyDict_Update(dict, update_dict)
                    builder.CreateCall(py_dict_update_func, {dict, update_dict});

                    // Decref update_dict since we consumed it
                    if (update_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {update_dict});
                    }
                }
            }
            else if (instr.opcode == op::DICT_MERGE)
            {
                // Merge dict at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    llvm::Value *update_dict = stack.back();
                    stack.pop_back();
                    bool update_is_ptr = update_dict->getType()->isPointerTy();
                    llvm::Value *dict = stack[stack.size() - i_val]; // Dict stays on stack

                    // PyDict_Merge(dict, update_dict, 1) - override=1 for merge
                    llvm::Value *override_flag = llvm::ConstantInt::get(builder.getInt32Ty(), 1);
                    builder.CreateCall(py_dict_merge_func, {dict, update_dict, override_flag});

                    // Decref update_dict since we consumed it
                    if (update_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {update_dict});
                    }
                }
            }
            else if (instr.opcode == op::SET_ADD)
            {
                // Add TOS to the set at STACK[-i]
                // arg = i (distance from TOS)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size())
                {
                    llvm::Value *item = stack.back();
                    stack.pop_back();
                    bool item_is_ptr = item->getType()->isPointerTy();
                    bool item_was_boxed = false;
                    llvm::Value *set = stack[stack.size() - i_val]; // Set stays on stack

                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64))
                    {
                        item = builder.CreateCall(py_long_fromlonglong_func, {item});
                        item_was_boxed = true;
                    }

                    // PySet_Add does NOT steal references (it increfs)
                    builder.CreateCall(py_set_add_func, {set, item});

                    // Decref our copy since Add already incref'd its own
                    if (item_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                    else if (item_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }
            }
            else if (instr.opcode == op::MAP_ADD)
            {
                // Add key-value pair to the dict at STACK[-i]
                // Stack: TOS=value, TOS1=key
                // arg = i (distance from TOS, after popping key and value)
                int i_val = instr.arg;
                if (stack.size() >= 2 && static_cast<size_t>(i_val) <= stack.size() - 2)
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    bool key_is_ptr = key->getType()->isPointerTy();
                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool key_was_boxed = false;
                    bool value_was_boxed = false;
                    llvm::Value *dict = stack[stack.size() - i_val]; // Dict stays on stack

                    // Convert int64 to PyObject* if needed
                    if (key->getType()->isIntegerTy(64))
                    {
                        key = builder.CreateCall(py_long_fromlonglong_func, {key});
                        key_was_boxed = true;
                    }
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        value_was_boxed = true;
                    }

                    // PyDict_SetItem does NOT steal references (it increfs both)
                    builder.CreateCall(py_dict_setitem_func, {dict, key, value});

                    // Decref our copies since SetItem already incref'd its own
                    if (key_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    else if (key_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                }
            }
            // ========== DELETE Operations ==========
            else if (instr.opcode == op::DELETE_FAST)
            {
                // DELETE_FAST: Delete local variable co_varnames[arg]
                // Sets the local slot to NULL (unbound)
                int var_idx = instr.arg;
                if (local_allocas.count(var_idx))
                {
                    // Load and decref the old value
                    llvm::Value *old_val = builder.CreateLoad(ptr_type, local_allocas[var_idx], "old_local");
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *is_not_null = builder.CreateICmpNE(old_val, null_check, "is_not_null");

                    llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(*local_context, "decref_del", func);
                    llvm::BasicBlock *store_block = llvm::BasicBlock::Create(*local_context, "store_null", func);

                    builder.CreateCondBr(is_not_null, decref_block, store_block);

                    builder.SetInsertPoint(decref_block);
                    builder.CreateCall(py_decref_func, {old_val});
                    builder.CreateBr(store_block);

                    builder.SetInsertPoint(store_block);
                    // Set slot to NULL (represents unbound local)
                    builder.CreateStore(null_check, local_allocas[var_idx]);
                }
            }
            else if (instr.opcode == op::DELETE_ATTR)
            {
                // DELETE_ATTR: Implements del obj.attr
                // Stack: TOS=obj
                // Python 3.13: DELETE_ATTR uses namei directly (unlike LOAD_ATTR/STORE_ATTR which use arg >> 1)
                int name_idx = instr.arg;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();

                    // Get attribute name from names (PyUnicode string)
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    // PyObject_DelAttr(obj, attr_name) - returns 0 on success, -1 on failure
                    builder.CreateCall(py_object_delattr_func, {obj, attr_name});

                    // Decref the object we consumed from the stack
                    if (obj->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {obj});
                    }
                }
            }
            else if (instr.opcode == op::DELETE_GLOBAL)
            {
                // DELETE_GLOBAL: Implements del global_name
                // arg = index into co_names
                int name_idx = instr.arg;

                if (name_idx < static_cast<int>(name_objects.size()))
                {
                    // Get the name object for deletion
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "del_name");

                    // Get globals dict pointer
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    // PyDict_DelItem(globals_dict, name) - returns 0 on success, -1 on failure
                    builder.CreateCall(py_dict_delitem_func, {globals_dict, name_obj});
                }
            }
            else if (instr.opcode == op::DELETE_NAME)
            {
                // DELETE_NAME: Implements del name in module/class scope
                // Similar to DELETE_GLOBAL but for local namespace
                // In practice, this often uses the same globals dict at module level
                int name_idx = instr.arg;

                if (name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "del_name");

                    // For now, use globals dict (correct for module-level code)
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    builder.CreateCall(py_dict_delitem_func, {globals_dict, name_obj});
                }
            }
            else if (instr.opcode == op::DELETE_DEREF)
            {
                // DELETE_DEREF: Delete a name from closure cell
                // arg = index into "fast locals" storage (after nlocals, in cellvars/freevars area)
                int cell_idx = instr.arg;

                // Cell index is relative to the closure cells we received
                // For functions with closures, cells are at indices >= nlocals
                if (cell_idx < static_cast<int>(closure_cells.size()) && closure_cells[cell_idx] != nullptr)
                {
                    llvm::Value *cell_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(closure_cells[cell_idx]));
                    llvm::Value *cell = builder.CreateIntToPtr(cell_ptr, ptr_type, "cell");

                    // PyCell_Set(cell, NULL) to clear the cell
                    llvm::Value *null_value = llvm::ConstantPointerNull::get(
                        llvm::PointerType::get(*local_context, 0));
                    builder.CreateCall(py_cell_set_func, {cell, null_value});
                }
            }
            // ========== Module/Class Namespace Operations ==========
            else if (instr.opcode == op::STORE_NAME)
            {
                // STORE_NAME: Store TOS into local namespace (module/class level)
                // arg = index into co_names
                // At module level, local namespace = globals
                int name_idx = instr.arg;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    // Get the name object
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "store_name");

                    // Get globals dict (at module level, locals = globals)
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    // PyDict_SetItem(globals_dict, name, value)
                    builder.CreateCall(py_dict_setitem_func, {globals_dict, name_obj, value});

                    // Decref value (PyDict_SetItem does NOT steal reference)
                    if (value->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                }
            }
            else if (instr.opcode == op::LOAD_NAME)
            {
                // LOAD_NAME: Load from local namespace (module/class level)
                // arg = index into co_names
                // Lookup order: locals (globals at module level) -> globals -> builtins
                int name_idx = instr.arg;

                if (name_idx < static_cast<int>(name_objects.size()))
                {
                    // Get the name object
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "load_name");

                    // Get globals dict
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    // Try globals first (at module level, locals = globals)
                    llvm::Value *result = builder.CreateCall(py_dict_getitem_func, {globals_dict, name_obj}, "name_lookup");

                    // Check if found
                    llvm::Value *is_null = builder.CreateICmpEQ(
                        result,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "not_found");

                    llvm::BasicBlock *found_block = llvm::BasicBlock::Create(*local_context, "name_found", func);
                    llvm::BasicBlock *try_builtins_block = llvm::BasicBlock::Create(*local_context, "name_try_builtins", func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(*local_context, "name_continue", func);

                    builder.CreateCondBr(is_null, try_builtins_block, found_block);

                    // Try builtins
                    builder.SetInsertPoint(try_builtins_block);
                    llvm::Value *builtins_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(builtins_dict_ptr));
                    llvm::Value *builtins_dict = builder.CreateIntToPtr(builtins_ptr, ptr_type, "builtins_dict");
                    llvm::Value *builtin_result = builder.CreateCall(py_dict_getitem_func, {builtins_dict, name_obj}, "builtin_lookup");
                    builder.CreateBr(continue_block);

                    // Found in globals
                    builder.SetInsertPoint(found_block);
                    builder.CreateBr(continue_block);

                    // Continue with PHI node
                    builder.SetInsertPoint(continue_block);
                    llvm::PHINode *result_phi = builder.CreatePHI(ptr_type, 2, "name_result");
                    result_phi->addIncoming(builtin_result, try_builtins_block);
                    result_phi->addIncoming(result, found_block);

                    // Incref (PyDict_GetItem returns borrowed reference)
                    builder.CreateCall(py_incref_func, {result_phi});

                    stack.push_back(result_phi);
                }
            }
            else if (instr.opcode == op::STORE_GLOBAL)
            {
                // STORE_GLOBAL: Store TOS into global namespace
                // arg = index into co_names
                int name_idx = instr.arg;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    // Get the name object
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "store_global_name");

                    // Get globals dict
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    // Box i64 values to PyLong before storing in dict
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                    }

                    // PyDict_SetItem(globals_dict, name, value)
                    // Returns 0 on success, -1 on failure
                    llvm::Value *result = builder.CreateCall(py_dict_setitem_func, {globals_dict, name_obj, value});

                    // Decref value (PyDict_SetItem does NOT steal reference, it increfs internally)
                    builder.CreateCall(py_decref_func, {value});

                    // Check for error
                    llvm::Value *is_error = builder.CreateICmpSLT(
                        result,
                        llvm::ConstantInt::get(builder.getInt32Ty(), 0),
                        "store_global_error");

                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                        *local_context, "store_global_error_" + std::to_string(current_offset), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "store_global_continue_" + std::to_string(current_offset), func);

                    builder.CreateCondBr(is_error, error_block, continue_block);

                    // Error block - return NULL
                    builder.SetInsertPoint(error_block);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                    // Continue block
                    builder.SetInsertPoint(continue_block);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_CHECK)
            {
                // LOAD_FAST_CHECK: Like LOAD_FAST but raises UnboundLocalError if uninitialized
                // arg = local variable index
                // In our implementation, we treat it like LOAD_FAST since we initialize all locals
                int slot = instr.arg;
                if (local_allocas.count(slot))
                {
                    llvm::Value *loaded = builder.CreateLoad(ptr_type, local_allocas[slot], "load_fast_check_" + std::to_string(slot));

                    // Check if value is NULL (uninitialized)
                    llvm::Value *is_null = builder.CreateICmpEQ(
                        loaded,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "is_unbound");

                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(*local_context, "unbound_error", func);
                    llvm::BasicBlock *ok_block = llvm::BasicBlock::Create(*local_context, "load_ok", func);

                    builder.CreateCondBr(is_null, error_block, ok_block);

                    // Error block - return NULL to signal error
                    builder.SetInsertPoint(error_block);
                    // In a real impl, we'd raise UnboundLocalError here
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                    // OK block
                    builder.SetInsertPoint(ok_block);
                    builder.CreateCall(py_incref_func, {loaded});
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::MAKE_CELL)
            {
                // MAKE_CELL: Create a cell object for the local at arg
                // This is used to create cells for variables captured by nested functions
                // arg = local variable index
                // 
                // CPython's MAKE_CELL takes the existing value at the local slot 
                // (which may be a function parameter), creates a cell containing that value,
                // and stores the cell back at the same slot.
                int slot = instr.arg;

                // Get the current value from the local slot (may be the parameter value)
                llvm::FunctionType *py_cell_new_type = llvm::FunctionType::get(
                    ptr_type, {ptr_type}, false);
                llvm::FunctionCallee py_cell_new_func = module->getOrInsertFunction(
                    "PyCell_New", py_cell_new_type);
                
                // Load the existing value from the local slot
                llvm::Value *initial_value = nullptr;
                if (local_allocas.count(slot))
                {
                    initial_value = builder.CreateLoad(ptr_type, local_allocas[slot], "initial_cell_value");
                }
                else
                {
                    // If no local exists yet, use NULL
                    initial_value = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                }
                
                // Create a cell containing the initial value
                llvm::Value *cell = builder.CreateCall(py_cell_new_func, {initial_value}, "new_cell");
                
                // Store the cell in the local slot (replacing the original value)
                if (local_allocas.count(slot))
                {
                    builder.CreateStore(cell, local_allocas[slot]);
                }
            }
            else if (instr.opcode == op::LOAD_CLOSURE)
            {
                // LOAD_CLOSURE: Push a reference to the cell object at slot arg
                // arg = index into co_cellvars + co_freevars
                // Used when creating nested functions to capture variables
                int slot = instr.arg;

                if (local_allocas.count(slot))
                {
                    // Load the cell object itself (not its contents like LOAD_DEREF)
                    llvm::Value *cell = builder.CreateLoad(ptr_type, local_allocas[slot], "load_closure_" + std::to_string(slot));
                    // Incref since we're pushing to stack
                    builder.CreateCall(py_incref_func, {cell});
                    stack.push_back(cell);
                }
            }
            // ========== FORMAT Operations (f-string support) ==========
            else if (instr.opcode == op::FORMAT_SIMPLE)
            {
                // FORMAT_SIMPLE: Format TOS with empty format spec
                // Implements: value.__format__("")
                // Stack: TOS=value -> TOS=formatted_string
                if (!stack.empty())
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    bool value_is_ptr = value->getType()->isPointerTy();

                    // Box int64 to PyObject* if needed
                    bool value_was_boxed = false;
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        value_was_boxed = true;
                    }

                    // PyObject_Format(value, NULL) - NULL means empty format spec ""
                    llvm::Value *null_spec = llvm::ConstantPointerNull::get(
                        llvm::PointerType::get(*local_context, 0));
                    llvm::Value *result = builder.CreateCall(py_object_format_func, {value, null_spec}, "formatted");

                    // Decref the value we consumed
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::FORMAT_WITH_SPEC)
            {
                // FORMAT_WITH_SPEC: Format TOS1 using format spec at TOS
                // Implements: value.__format__(spec)
                // Stack: TOS=spec, TOS1=value -> TOS=formatted_string
                if (stack.size() >= 2)
                {
                    llvm::Value *spec = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    bool spec_is_ptr = spec->getType()->isPointerTy();
                    bool value_is_ptr = value->getType()->isPointerTy();

                    // Box int64 values to PyObject* if needed
                    bool value_was_boxed = false;
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        value_was_boxed = true;
                    }
                    // spec should be a string, but handle int just in case
                    bool spec_was_boxed = false;
                    if (spec->getType()->isIntegerTy(64))
                    {
                        spec = builder.CreateCall(py_long_fromlonglong_func, {spec});
                        spec_was_boxed = true;
                    }

                    // PyObject_Format(value, spec)
                    llvm::Value *result = builder.CreateCall(py_object_format_func, {value, spec}, "formatted");

                    // Decref consumed values
                    if (spec_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {spec});
                    }
                    else if (spec_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {spec});
                    }
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CONVERT_VALUE)
            {
                // CONVERT_VALUE: Convert TOS to string using specified conversion
                // arg == 1: str(value)
                // arg == 2: repr(value)
                // arg == 3: ascii(value)
                // Stack: TOS=value -> TOS=converted_string
                if (!stack.empty())
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    bool value_is_ptr = value->getType()->isPointerTy();

                    // Box int64 to PyObject* if needed
                    bool value_was_boxed = false;
                    if (value->getType()->isIntegerTy(64))
                    {
                        value = builder.CreateCall(py_long_fromlonglong_func, {value});
                        value_was_boxed = true;
                    }

                    llvm::Value *result = nullptr;
                    int conversion = instr.arg;

                    if (conversion == 1)
                    {
                        // str(value)
                        result = builder.CreateCall(py_object_str_func, {value}, "str_conv");
                    }
                    else if (conversion == 2)
                    {
                        // repr(value)
                        result = builder.CreateCall(py_object_repr_func, {value}, "repr_conv");
                    }
                    else if (conversion == 3)
                    {
                        // ascii(value)
                        result = builder.CreateCall(py_object_ascii_func, {value}, "ascii_conv");
                    }
                    else
                    {
                        // Unknown conversion - raise error
                        llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context),
                            {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                            "PyErr_SetString", py_err_set_str_type);
                        llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(PyExc_ValueError));
                        llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                        llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported conversion type");
                        builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                        result = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    }

                    // Decref the value we consumed
                    if (value_was_boxed)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    else if (value_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {value});
                    }

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::BUILD_STRING)
            {
                // BUILD_STRING: Concatenate 'arg' strings from the stack
                // Stack: ..., str0, str1, ..., strN-1 -> TOS=concatenated_string
                // Strings are pushed in order, so str0 was pushed first (deepest), strN-1 is TOS
                int count = instr.arg;

                if (count > 0 && static_cast<int>(stack.size()) >= count)
                {
                    // Pop 'count' strings from stack (reverse order - TOS is last pushed)
                    std::vector<llvm::Value *> strings;
                    for (int i = 0; i < count; i++)
                    {
                        strings.push_back(stack.back());
                        stack.pop_back();
                    }
                    // Now strings[0] = TOS (last pushed), strings[count-1] = deepest (first pushed)

                    // Concatenate in correct order: start with first pushed (strings[count-1])
                    llvm::Value *result = strings[count - 1];

                    for (int i = count - 2; i >= 0; i--)
                    {
                        llvm::Value *next_str = strings[i];
                        llvm::Value *new_result = builder.CreateCall(py_unicode_concat_func, {result, next_str}, "concat_str");

                        // Check for NULL (concat failure - e.g., out of memory)
                        llvm::Value *is_null = builder.CreateIsNull(new_result);
                        llvm::BasicBlock *concat_error_block = llvm::BasicBlock::Create(
                            *local_context, "concat_error_" + std::to_string(i), func);
                        llvm::BasicBlock *concat_ok_block = llvm::BasicBlock::Create(
                            *local_context, "concat_ok_" + std::to_string(i), func);
                        builder.CreateCondBr(is_null, concat_error_block, concat_ok_block);

                        // Error path: decref what we have and return NULL
                        builder.SetInsertPoint(concat_error_block);
                        if (i < count - 2)
                        {
                            builder.CreateCall(py_decref_func, {result});
                        }
                        builder.CreateCall(py_decref_func, {next_str});
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                        // OK path: continue concatenation
                        builder.SetInsertPoint(concat_ok_block);

                        // Decref previous result (but not the original first string on first iteration)
                        if (i < count - 2)
                        {
                            // This was an intermediate result from previous concat
                            builder.CreateCall(py_decref_func, {result});
                        }
                        // Decref the string we just concatenated (next_str)
                        builder.CreateCall(py_decref_func, {next_str});

                        result = new_result;
                    }

                    // Decref the first string (strings[count-1]) since PyUnicode_Concat returns new reference
                    builder.CreateCall(py_decref_func, {strings[count - 1]});

                    stack.push_back(result);
                }
                else if (count == 0)
                {
                    // Empty string case
                    llvm::Value *empty_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(PyUnicode_FromString("")));
                    llvm::Value *empty_str = builder.CreateIntToPtr(empty_ptr, ptr_type, "empty_str");
                    stack.push_back(empty_str);
                }
            }
            // ========== IMPORT Operations ==========
            else if (instr.opcode == op::IMPORT_NAME)
            {
                // IMPORT_NAME: Import a module
                // Stack: TOS=fromlist, TOS1=level -> TOS=module
                // arg = index into co_names for module name
                int name_idx = instr.arg;

                if (stack.size() >= 2 && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *fromlist = stack.back();
                    stack.pop_back();
                    llvm::Value *level_obj = stack.back();
                    stack.pop_back();

                    // Get module name from names
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name = builder.CreateIntToPtr(name_ptr, ptr_type, "module_name");

                    // Get globals dict for context
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals");

                    // locals can be NULL for import
                    llvm::Value *locals_null = llvm::ConstantPointerNull::get(
                        llvm::PointerType::get(*local_context, 0));

                    // Extract level as integer
                    // level_obj is either int64 or PyLong
                    llvm::Value *level_int;
                    if (level_obj->getType()->isIntegerTy(64))
                    {
                        level_int = builder.CreateTrunc(level_obj, builder.getInt32Ty());
                    }
                    else
                    {
                        // PyLong_AsLong then truncate
                        llvm::Value *level_long = builder.CreateCall(py_long_aslong_func, {level_obj});
                        level_int = builder.CreateTrunc(level_long, builder.getInt32Ty());
                        // Decref the level PyLong
                        builder.CreateCall(py_decref_func, {level_obj});
                    }

                    // PyImport_ImportModuleLevelObject(name, globals, locals, fromlist, level)
                    llvm::Value *module = builder.CreateCall(
                        py_import_importmodule_func,
                        {name, globals, locals_null, fromlist, level_int},
                        "imported_module");

                    // Decref fromlist
                    if (fromlist->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {fromlist});
                    }

                    stack.push_back(module);
                }
            }
            else if (instr.opcode == op::IMPORT_FROM)
            {
                // IMPORT_FROM: Load attribute from module
                // Stack: TOS=module (stays on stack) -> pushes attribute
                // arg = index into co_names for attribute name
                int name_idx = instr.arg;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *module = stack.back(); // Don't pop - module stays on stack

                    // Get attribute name from names
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type, "attr_name");

                    // PyObject_GetAttr(module, attr_name) - returns new reference
                    llvm::Value *attr = builder.CreateCall(py_object_getattr_func, {module, attr_name}, "imported_attr");

                    stack.push_back(attr);
                }
            }
            else if (instr.opcode == op::LOAD_ATTR)
            {
                // Implements obj.attr
                // Python 3.13: arg >> 1 = index into co_names, arg & 1 = method load flag
                int name_idx = instr.arg >> 1;
                bool is_method = (instr.arg & 1) != 0;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();

                    // Get attribute name from names (PyUnicode string)
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    // PyObject_GetAttr returns new reference (bound method for methods)
                    llvm::Value *result = builder.CreateCall(py_object_getattr_func, {obj, attr_name});

                    // CRITICAL: Decref the object we consumed from the stack
                    if (obj->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {obj});
                    }

                    // Bug #3 Fix: Check for attribute error
                    check_error_and_branch(current_offset, result, "load_attr");

                    if (is_method)
                    {
                        // Method loading for CALL opcode
                        // CALL expects stack layout: [callable, self_or_null, args...]
                        // For bound methods from GetAttr, self is already bound in the method
                        // Push callable (bound method) first, then NULL for self_or_null
                        // Stack order: push method, then push NULL
                        // Result: [..., method, NULL] so that after LOAD_FAST arg:
                        //         [..., method, NULL, arg]
                        // CALL 1 sees: callable=stack[-3]=method, self_or_null=stack[-2]=NULL
                        llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
                            llvm::PointerType::get(*local_context, 0));
                        stack.push_back(result);   // callable = bound method
                        stack.push_back(null_ptr); // self_or_null = NULL
                    }
                    else
                    {
                        // Normal attribute access
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::LOAD_SUPER_ATTR)
            {
                // LOAD_SUPER_ATTR: Implements super().attr
                // Stack before: global_super (TOS), class, self
                // Stack after: attr_value or bound_method
                // arg >> 2 = index into co_names (attribute name)
                // arg & 1 = if set, load as method (push NULL after)
                // arg & 2 = if set, super() is being called without arguments
                int name_idx = instr.arg >> 2;
                bool load_method = (instr.arg & 1) != 0;

                if (stack.size() >= 3 && name_idx < static_cast<int>(name_objects.size()))
                {
                    // Pop in order: global_super, class, self
                    llvm::Value *global_super = stack.back();
                    stack.pop_back();
                    llvm::Value *cls = stack.back();
                    stack.pop_back();
                    llvm::Value *self = stack.back();
                    stack.pop_back();

                    // Get attribute name
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    // Call super(cls, self) to create super object
                    // Build args tuple: (cls, self)
                    llvm::Value *args_tuple = builder.CreateCall(py_tuple_new_func, {llvm::ConstantInt::get(i64_type, 2)}, "super_args");

                    // PyTuple_SetItem steals references, so incref first
                    builder.CreateCall(py_incref_func, {cls});
                    builder.CreateCall(py_incref_func, {self});
                    builder.CreateCall(py_tuple_setitem_func, {args_tuple, llvm::ConstantInt::get(i64_type, 0), cls});
                    builder.CreateCall(py_tuple_setitem_func, {args_tuple, llvm::ConstantInt::get(i64_type, 1), self});

                    // Call global_super(cls, self)
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *super_obj = builder.CreateCall(py_object_call_func, {global_super, args_tuple, null_kwargs}, "super_obj");

                    builder.CreateCall(py_decref_func, {args_tuple});

                    // Get attribute from super object
                    llvm::Value *result = builder.CreateCall(py_object_getattr_func, {super_obj, attr_name}, "super_attr");

                    // Decref intermediate values
                    builder.CreateCall(py_decref_func, {super_obj});
                    builder.CreateCall(py_decref_func, {global_super});
                    builder.CreateCall(py_decref_func, {cls});
                    builder.CreateCall(py_decref_func, {self});

                    // Check for error
                    check_error_and_branch(current_offset, result, "load_super_attr");

                    if (load_method)
                    {
                        // For method calls, push callable then NULL
                        llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        stack.push_back(result);
                        stack.push_back(null_ptr);
                    }
                    else
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::LOAD_GLOBAL)
            {
                // Python 3.13: LOAD_GLOBAL loads global variable
                // arg >> 1 = index into co_names
                // arg & 1 = if set, push NULL after global (for calling convention)
                int name_idx = instr.arg >> 1;
                bool push_null = (instr.arg & 1) != 0;

                if (name_idx < name_objects.size())
                {
                    // Bug #4 Fix: Runtime lookup instead of compile-time resolved value
                    // Get the name object for lookup
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "name_obj");

                    // Get globals dict pointer
                    llvm::Value *globals_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr, ptr_type, "globals_dict");

                    // PyDict_GetItem(globals_dict, name) - returns borrowed reference or NULL
                    llvm::Value *global_obj = builder.CreateCall(
                        py_dict_getitem_func,
                        {globals_dict, name_obj},
                        "global_lookup");

                    // Check if found in globals, if not try builtins
                    llvm::Value *is_null = builder.CreateICmpEQ(
                        global_obj,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "not_found_in_globals");

                    llvm::BasicBlock *found_block = llvm::BasicBlock::Create(*local_context, "global_found", func);
                    llvm::BasicBlock *try_builtins_block = llvm::BasicBlock::Create(*local_context, "try_builtins", func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(*local_context, "global_continue", func);

                    builder.CreateCondBr(is_null, try_builtins_block, found_block);

                    // Try builtins
                    builder.SetInsertPoint(try_builtins_block);
                    llvm::Value *builtins_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(builtins_dict_ptr));
                    llvm::Value *builtins_dict = builder.CreateIntToPtr(builtins_ptr, ptr_type, "builtins_dict");
                    llvm::Value *builtin_obj = builder.CreateCall(
                        py_dict_getitem_func,
                        {builtins_dict, name_obj},
                        "builtin_lookup");
                    builder.CreateBr(continue_block);

                    // Found in globals
                    builder.SetInsertPoint(found_block);
                    builder.CreateBr(continue_block);

                    // Continue with PHI node to select result
                    builder.SetInsertPoint(continue_block);
                    llvm::PHINode *result_phi = builder.CreatePHI(ptr_type, 2, "global_result");
                    result_phi->addIncoming(builtin_obj, try_builtins_block);
                    result_phi->addIncoming(global_obj, found_block);

                    // Incref the result (PyDict_GetItem returns borrowed reference)
                    builder.CreateCall(py_incref_func, {result_phi});

                    stack.push_back(result_phi);

                    // Push NULL after global if needed (Python 3.13 calling convention)
                    if (push_null)
                    {
                        llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        stack.push_back(null_ptr);
                    }
                }
            }
            else if (instr.opcode == op::CALL)
            {
                // Python 3.13: CALL opcode, arg = number of arguments (excluding self/NULL)
                // Stack layout (CPython uses indices from bottom):
                //   callable = stack[-2-oparg]
                //   self_or_null = stack[-1-oparg]
                //   args = &stack[-oparg] (oparg elements)
                int num_args = instr.arg;

                if (stack.size() >= static_cast<size_t>(num_args + 2))
                {
                    // Access stack by index (matches CPython implementation)
                    size_t base = stack.size() - num_args - 2;

                    llvm::Value *callable = stack[base];         // stack[-2-oparg]
                    llvm::Value *self_or_null = stack[base + 1]; // stack[-1-oparg]

                    // Track if operands are pointers for decref
                    bool callable_is_ptr = callable->getType()->isPointerTy();
                    std::vector<bool> args_are_ptr;

                    // Collect arguments in order
                    std::vector<llvm::Value *> args;
                    for (int i = 0; i < num_args; ++i)
                    {
                        llvm::Value *arg = stack[base + 2 + i];
                        args_are_ptr.push_back(arg->getType()->isPointerTy());
                        args.push_back(arg); // stack[-oparg+i]
                    }

                    // Remove all CALL operands from stack
                    stack.erase(stack.begin() + base, stack.end());

                    // Create args tuple - PyTuple_SetItem steals references so we transfer ownership
                    llvm::Value *args_count = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value *args_tuple = builder.CreateCall(py_tuple_new_func, {args_count});

                    // Fill tuple with args in correct order
                    for (int i = 0; i < num_args; ++i)
                    {
                        llvm::Value *index_val = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value *arg = args[i];

                        // Box int64 to PyObject* if needed
                        if (arg->getType()->isIntegerTy(64))
                        {
                            arg = builder.CreateCall(py_long_fromlonglong_func, {arg});
                            // PyTuple_SetItem steals reference - new PyLong is transferred
                        }
                        // For PyObject*: PyTuple_SetItem steals reference
                        // We consume the stack value, so transfer ownership directly
                        // No incref needed

                        // PyTuple_SetItem steals reference (transfers ownership)
                        builder.CreateCall(py_tuple_setitem_func, {args_tuple, index_val, arg});
                    }

                    // Call PyObject_Call(callable, args_tuple, NULL)
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *result = builder.CreateCall(py_object_call_func, {callable, args_tuple, null_kwargs});

                    // Decrement args_tuple refcount (we're done with it)
                    builder.CreateCall(py_decref_func, {args_tuple});

                    // Decref callable (we consumed it from the stack)
                    if (callable_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {callable});
                    }

                    // Note: self_or_null is either NULL or a reference we need to decref
                    // The NULL check is needed at runtime
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);

                    // Bug #3 Fix: Check for exception from called function
                    check_error_and_branch(current_offset, result, "call");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CALL_KW)
            {
                // Python 3.13: CALL_KW opcode - call with keyword arguments
                // Stack layout:
                //   callable = stack[-3-oparg]
                //   self_or_null = stack[-2-oparg]
                //   args = stack[-1-oparg : -1] (oparg elements = positional + keyword args)
                //   kwnames = stack[-1] (tuple of keyword names)
                // oparg = total number of arguments (positional + keyword)
                int num_args = instr.arg;

                if (stack.size() >= static_cast<size_t>(num_args + 3))
                {
                    // Pop kwnames tuple first (TOS)
                    llvm::Value *kwnames = stack.back();
                    stack.pop_back();

                    // Now stack layout is like CALL: callable, self_or_null, args...
                    size_t base = stack.size() - num_args - 2;

                    llvm::Value *callable = stack[base];
                    llvm::Value *self_or_null = stack[base + 1];

                    bool callable_is_ptr = callable->getType()->isPointerTy();

                    // Collect all arguments
                    std::vector<llvm::Value *> args;
                    for (int i = 0; i < num_args; ++i)
                    {
                        args.push_back(stack[base + 2 + i]);
                    }

                    // Remove all operands from stack
                    stack.erase(stack.begin() + base, stack.end());

                    // Allocate stack array to hold args as PyObject* pointers
                    llvm::Type *ptr_type_local = llvm::PointerType::get(*local_context, 0);
                    llvm::ArrayType *args_array_type = llvm::ArrayType::get(ptr_type_local, num_args);
                    llvm::Value *args_array = builder.CreateAlloca(args_array_type, nullptr, "args_array");

                    // Store each arg into the array, converting int64 to PyLong if needed
                    for (int i = 0; i < num_args; ++i)
                    {
                        llvm::Value *arg = args[i];

                        // Convert int64 to PyObject* if necessary
                        if (arg->getType()->isIntegerTy(64))
                        {
                            arg = builder.CreateCall(py_long_fromlonglong_func, {arg});
                        }

                        // Get pointer to array element and store
                        llvm::Value *indices[] = {
                            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0),
                            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), i)};
                        llvm::Value *elem_ptr = builder.CreateGEP(args_array_type, args_array, indices, "arg_ptr");
                        builder.CreateStore(arg, elem_ptr);
                    }

                    // Get pointer to first element for passing to helper
                    llvm::Value *first_indices[] = {
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0),
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0)};
                    llvm::Value *args_ptr = builder.CreateGEP(args_array_type, args_array, first_indices, "args_ptr");

                    // Call our helper: jit_call_with_kwargs(callable, args_ptr, nargs, kwnames)
                    llvm::Value *nargs_val = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value *result = builder.CreateCall(jit_call_with_kwargs_func,
                                                             {callable, args_ptr, nargs_val, kwnames}, "call_kw_result");

                    // Cleanup kwnames
                    builder.CreateCall(py_decref_func, {kwnames});

                    if (callable_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {callable});
                    }

                    // Decref self_or_null if not null
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self_kw", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self_kw", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);

                    // Bug #3 Fix: Check for exception from called function
                    check_error_and_branch(current_offset, result, "call_kw");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CALL_FUNCTION_EX)
            {
                // Python 3.13: CALL_FUNCTION_EX opcode - call with *args and **kwargs
                // Reference: https://docs.python.org/3.13/library/dis.html#opcode-CALL_FUNCTION_EX
                //
                // Stack layout (bottom to top):
                //   callable
                //   self_or_null (from LOAD_GLOBAL with push_null flag)
                //   args_tuple (iterable, will be unpacked as positional args)
                //   kwargs_dict (mapping, present if flags & 1) - optional
                //
                // flags (instr.arg):
                //   bit 0: if set, kwargs_dict is present on stack

                bool has_kwargs = (instr.arg & 1) != 0;
                // Stack has: callable, null, args, [kwargs]
                size_t required = has_kwargs ? 4 : 3;

                if (stack.size() >= required)
                {
                    llvm::Type *ptr_type_local = llvm::PointerType::get(*local_context, 0);

                    // Pop in reverse order (top to bottom)
                    llvm::Value *kwargs = nullptr;
                    if (has_kwargs)
                    {
                        kwargs = stack.back();
                        stack.pop_back();
                    }

                    llvm::Value *args_seq = stack.back();
                    stack.pop_back();

                    // Pop the self_or_null marker (from LOAD_GLOBAL's push_null)
                    llvm::Value *self_or_null = stack.back();
                    stack.pop_back();

                    llvm::Value *callable = stack.back();
                    stack.pop_back();

                    bool callable_is_ptr = callable->getType()->isPointerTy();

                    // PyObject_Call requires args to be a tuple, but we might have a list
                    // Convert args_seq to tuple using PySequence_Tuple
                    llvm::Value *args_tuple = builder.CreateCall(py_sequence_tuple_func,
                                                                  {args_seq}, "args_as_tuple");
                    
                    // Decref the original sequence (we have the tuple now)
                    builder.CreateCall(py_decref_func, {args_seq});

                    // Prepare kwargs (NULL if not present)
                    llvm::Value *kwargs_arg = kwargs;
                    if (!kwargs_arg)
                    {
                        kwargs_arg = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    }

                    // Call PyObject_Call(callable, args_tuple, kwargs)
                    llvm::Value *result = builder.CreateCall(py_object_call_func,
                                                             {callable, args_tuple, kwargs_arg}, "call_ex_result");

                    // Cleanup: decref the args tuple we created
                    builder.CreateCall(py_decref_func, {args_tuple});

                    // kwargs only if present (not the NULL we created)
                    if (has_kwargs && kwargs)
                    {
                        builder.CreateCall(py_decref_func, {kwargs});
                    }

                    // Decref callable
                    if (callable_is_ptr)
                    {
                        builder.CreateCall(py_decref_func, {callable});
                    }

                    // Decref self_or_null if not NULL (similar to CALL opcode)
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self_ex");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self_ex", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self_ex", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);

                    // Check for exception from called function
                    check_error_and_branch(current_offset, result, "call_function_ex");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::POP_TOP)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    // Decref PyObject* values being popped
                    if (val->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {val});
                    }
                }
            }
            // ========== Function/Class Creation Opcodes ==========
            else if (instr.opcode == op::MAKE_FUNCTION)
            {
                // Python 3.13: MAKE_FUNCTION
                // Reference: https://docs.python.org/3.13/library/dis.html#opcode-MAKE_FUNCTION
                //
                // Stack before: code_object (TOS)
                // Stack after: function_object (TOS)
                //
                // Creates a new function object from the code object.
                // The code object's co_qualname is used as the function's __qualname__.
                // The function's __globals__ is set to the module's globals dict.
                //
                // In Python 3.13, MAKE_FUNCTION no longer takes flags (unlike older versions).
                // Defaults, kwdefaults, annotations, and closure are set separately via
                // SET_FUNCTION_ATTRIBUTE opcode.

                if (stack.empty())
                {
                    PyErr_SetString(PyExc_RuntimeError, "MAKE_FUNCTION: stack underflow");
                    return false;
                }

                // Pop code object from stack
                llvm::Value *code_obj = stack.back();
                stack.pop_back();

                // Ensure code_obj is a pointer type (should always be, but check)
                if (code_obj->getType()->isIntegerTy(64))
                {
                    code_obj = builder.CreateCall(py_long_fromlonglong_func, {code_obj});
                }

                // Get globals dict as constant pointer
                llvm::Value *globals_ptr = llvm::ConstantInt::get(
                    builder.getInt64Ty(),
                    reinterpret_cast<uint64_t>(globals_dict_ptr));
                llvm::Value *globals = builder.CreateIntToPtr(globals_ptr, ptr_type);

                // Call PyFunction_New(code, globals)
                llvm::Value *func_obj = builder.CreateCall(py_function_new_func, {code_obj, globals});

                // Decref code object (we consumed it)
                builder.CreateCall(py_decref_func, {code_obj});

                // Check for error
                check_error_and_branch(current_offset, func_obj, "make_function");

                // Push new function object
                stack.push_back(func_obj);
            }
            else if (instr.opcode == op::SET_FUNCTION_ATTRIBUTE)
            {
                // Python 3.13: SET_FUNCTION_ATTRIBUTE(flag)
                // Reference: https://docs.python.org/3.13/library/dis.html#opcode-SET_FUNCTION_ATTRIBUTE
                //
                // From Python docs: "Expects the function at STACK[-1] and the attribute value 
                // to set at STACK[-2]; consumes both and leaves the function at STACK[-1]"
                //
                // Stack before: func (TOS), value (TOS1)
                // Stack after: func (TOS)
                //
                // Sets an attribute on the function based on the flag:
                //   0x01: defaults - tuple of default argument values for positional params
                //   0x02: kwdefaults - dict of keyword-only parameter defaults
                //   0x04: annotations - tuple of strings for annotations
                //   0x08: closure - tuple of cell objects for free variables

                if (stack.size() < 2)
                {
                    PyErr_SetString(PyExc_RuntimeError, "SET_FUNCTION_ATTRIBUTE: stack underflow");
                    return false;
                }

                // Pop function (TOS) and value (TOS1) from stack
                llvm::Value *py_func = stack.back(); // TOS = function
                stack.pop_back();
                llvm::Value *value = stack.back();   // TOS1 = value (closure tuple, defaults, etc.)
                stack.pop_back();

                // Ensure both values are pointer types (box i64 if needed)
                if (py_func->getType()->isIntegerTy(64))
                {
                    py_func = builder.CreateCall(py_long_fromlonglong_func, {py_func});
                }
                if (value->getType()->isIntegerTy(64))
                {
                    value = builder.CreateCall(py_long_fromlonglong_func, {value});
                }

                int flag = instr.arg;
                llvm::Value *result = nullptr;

                if (flag == 0x01)
                {
                    // Set defaults (tuple of default argument values)
                    result = builder.CreateCall(py_function_set_defaults_func, {py_func, value});
                }
                else if (flag == 0x02)
                {
                    // Set kwdefaults (dict of keyword-only parameter defaults)
                    result = builder.CreateCall(py_function_set_kwdefaults_func, {py_func, value});
                }
                else if (flag == 0x04)
                {
                    // Set annotations (tuple of strings)
                    result = builder.CreateCall(py_function_set_annotations_func, {py_func, value});
                }
                else if (flag == 0x08)
                {
                    // Set closure (tuple of cell objects)
                    result = builder.CreateCall(py_function_set_closure_func, {py_func, value});
                }
                else
                {
                    // Unknown flag - this shouldn't happen with valid bytecode
                    PyErr_Format(PyExc_RuntimeError, "SET_FUNCTION_ATTRIBUTE: unknown flag %d", flag);
                    return false;
                }

                // Decref value (consumed by setter which steals reference)
                // Note: PyFunction_Set* functions steal the reference to value
                // so we don't need to decref it here

                // Check for error (returns -1 on failure)
                llvm::Value *is_error = builder.CreateICmpSLT(result, builder.getInt32(0));

                // Create error handling blocks
                llvm::BasicBlock *error_bb = llvm::BasicBlock::Create(*local_context, "set_func_attr_error", func);
                llvm::BasicBlock *continue_bb = llvm::BasicBlock::Create(*local_context, "set_func_attr_continue", func);
                builder.CreateCondBr(is_error, error_bb, continue_bb);

                // Error block: decref py_func and return NULL
                builder.SetInsertPoint(error_bb);
                builder.CreateCall(py_decref_func, {py_func});
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                // Continue block
                builder.SetInsertPoint(continue_bb);

                // Push function back on stack
                stack.push_back(py_func);
            }
            else if (instr.opcode == op::LOAD_BUILD_CLASS)
            {
                // Python 3.13: LOAD_BUILD_CLASS
                // Reference: https://docs.python.org/3.13/library/dis.html#opcode-LOAD_BUILD_CLASS
                //
                // Stack before: (empty)
                // Stack after: builtins.__build_class__ (TOS)
                //
                // Pushes builtins.__build_class__() onto the stack. This is called by
                // class definitions to construct a new class.

                // Get builtins dict pointer
                llvm::Value *builtins_ptr = llvm::ConstantInt::get(
                    builder.getInt64Ty(),
                    reinterpret_cast<uint64_t>(builtins_dict_ptr));
                llvm::Value *builtins = builder.CreateIntToPtr(builtins_ptr, ptr_type);

                // Get the name "__build_class__" as a Python string constant
                // We need to create it at runtime or use a constant from co_names
                // For simplicity, use PyDict_GetItemString via a helper

                // Actually, let's use PyObject_GetAttrString since builtins might be a module
                // But builtins_dict_ptr is already the builtins dict, so use PyDict_GetItem

                // Create the string "__build_class__" as a PyObject
                PyObject *build_class_name = PyUnicode_InternFromString("__build_class__");
                if (!build_class_name)
                {
                    PyErr_SetString(PyExc_RuntimeError, "LOAD_BUILD_CLASS: failed to create __build_class__ string");
                    return false;
                }
                Py_INCREF(build_class_name);                  // Keep it alive
                stored_constants.push_back(build_class_name); // Track for cleanup

                llvm::Value *name_ptr = llvm::ConstantInt::get(
                    builder.getInt64Ty(),
                    reinterpret_cast<uint64_t>(build_class_name));
                llvm::Value *name = builder.CreateIntToPtr(name_ptr, ptr_type);

                // Call PyDict_GetItem(builtins, "__build_class__")
                // Note: PyDict_GetItem returns a borrowed reference, so we need to incref
                llvm::Value *build_class = builder.CreateCall(py_dict_getitem_func, {builtins, name});

                // Check if __build_class__ was found
                llvm::Value *is_null = builder.CreateIsNull(build_class);

                // Create error handling blocks
                llvm::BasicBlock *error_bb = llvm::BasicBlock::Create(*local_context, "load_build_class_error", func);
                llvm::BasicBlock *continue_bb = llvm::BasicBlock::Create(*local_context, "load_build_class_continue", func);
                builder.CreateCondBr(is_null, error_bb, continue_bb);

                // Error block: raise error and return NULL
                builder.SetInsertPoint(error_bb);
                // Note: if PyErr_Occurred() is not set, we should set an error
                // But typically if __build_class__ is missing, something is very wrong
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                // Continue block
                builder.SetInsertPoint(continue_bb);

                // Incref since PyDict_GetItem returns borrowed reference
                builder.CreateCall(py_incref_func, {build_class});

                // Push __build_class__ onto stack
                stack.push_back(build_class);
            }
            else if (instr.opcode == op::END_FOR)
            {
                // END_FOR: Pop the iterator from the stack (used after FOR_ITER exhausted)
                if (!stack.empty())
                {
                    llvm::Value *iterator = stack.back();
                    stack.pop_back();
                    // Decref the iterator since we're done with it
                    if (iterator->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {iterator});
                    }
                }
            }
            else if (instr.opcode == op::COPY)
            {
                // Copy the n-th item from the stack to the top
                // arg = n (1 means TOS, 2 means TOS1, etc.)
                int n = instr.arg;
                if (n > 0 && static_cast<size_t>(n) <= stack.size())
                {
                    llvm::Value *item = stack[stack.size() - n];
                    // For PyObject*, incref since we're duplicating the reference
                    if (item->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_incref_func, {item});
                    }
                    stack.push_back(item);
                }
            }
            else if (instr.opcode == op::SWAP)
            {
                // Swap TOS with the n-th item from the stack
                // arg = n (2 means swap TOS with TOS1, 3 means swap TOS with TOS2, etc.)
                int n = instr.arg;
                if (n >= 2 && static_cast<size_t>(n) <= stack.size())
                {
                    size_t tos_idx = stack.size() - 1;
                    size_t other_idx = stack.size() - n;
                    std::swap(stack[tos_idx], stack[other_idx]);
                }
            }
            else if (instr.opcode == op::PUSH_NULL)
            {
                // Push a NULL onto the stack (used for method calling convention)
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                stack.push_back(null_ptr);
            }
            else if (instr.opcode == op::GET_ITER)
            {
                // Implements iter(TOS) - get an iterator from an object
                if (!stack.empty())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();

                    // PyObject_GetIter returns a new reference
                    llvm::Value *iterator = builder.CreateCall(py_object_getiter_func, {iterable}, "iter");

                    // CRITICAL: LOAD_FAST increfs, so we own this reference - must decref
                    // The iterable was pushed with a new reference from LOAD_FAST
                    if (iterable->getType()->isPointerTy())
                    {
                        builder.CreateCall(py_decref_func, {iterable});
                    }

                    stack.push_back(iterator);
                }
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                // FOR_ITER: Get next item from iterator
                // If exhausted, jump forward by arg; otherwise push next value
                //
                // IMPORTANT: CPython's FOR_ITER is a "super-instruction". When exhausted:
                // 1. Pop and decref the iterator
                // 2. Jump past END_FOR and POP_TOP (skip them entirely)
                // The END_FOR and POP_TOP after a for loop are only fallback code.
                //
                // Stack semantics:
                // - On continue: iterator stays, next_item pushed
                // - On exhaustion: iterator is popped here, jump to code after loop
                if (!stack.empty() && i + 1 < instructions.size())
                {
                    llvm::Value *iterator = stack.back();

                    // Call PyIter_Next - returns next item or NULL
                    llvm::Value *next_item = builder.CreateCall(py_iter_next_func, {iterator}, "next");

                    // Check if next_item is NULL (iterator exhausted)
                    llvm::Value *is_null = builder.CreateICmpEQ(
                        next_item,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "iter_done");

                    // CPython: argval points to END_FOR. We need to jump past END_FOR + POP_TOP.
                    // END_FOR is 2 bytes, POP_TOP is 2 bytes, so target is argval + 4
                    int end_for_offset = instr.argval;
                    int after_loop_offset = end_for_offset + 4; // Skip END_FOR (2) + POP_TOP (2)
                    int next_offset = instructions[i + 1].offset;

                    if (!jump_targets.count(after_loop_offset))
                    {
                        jump_targets[after_loop_offset] = llvm::BasicBlock::Create(
                            *local_context, "after_loop_" + std::to_string(after_loop_offset), func);
                    }
                    if (!jump_targets.count(next_offset))
                    {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "iter_continue_" + std::to_string(next_offset), func);
                    }

                    // Create blocks for the two paths
                    llvm::BasicBlock *exhausted_block = llvm::BasicBlock::Create(
                        *local_context, "for_iter_exhausted_" + std::to_string(i), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "for_iter_continue_" + std::to_string(i), func);

                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        builder.CreateCondBr(is_null, exhausted_block, continue_block);
                    }

                    // Exhausted path: Pop and decref iterator, then jump past END_FOR + POP_TOP
                    builder.SetInsertPoint(exhausted_block);
                    builder.CreateCall(py_decref_func, {iterator});

                    // CRITICAL FIX: Record the exhaust-path stack state for after_loop_offset
                    // The stack after iterator is popped (for the after-loop code)
                    {
                        BlockStackState exhaust_state;
                        exhaust_state.predecessor = exhausted_block;
                        // Copy stack WITHOUT the iterator (it's been popped)
                        for (size_t s = 0; s < stack.size() - 1; ++s)
                        {
                            exhaust_state.stack.push_back(stack[s]);
                        }
                        block_incoming_stacks[after_loop_offset].push_back(exhaust_state);
                    }

                    builder.CreateBr(jump_targets[after_loop_offset]);

                    // Continue path: push next item, continue to next instruction
                    builder.SetInsertPoint(continue_block);
                    stack.push_back(next_item);

                    // Record continue-path stack state
                    {
                        BlockStackState continue_state;
                        continue_state.predecessor = continue_block;
                        continue_state.stack = stack; // [... iterator, next_item]
                        block_incoming_stacks[next_offset].push_back(continue_state);
                    }

                    builder.CreateBr(jump_targets[next_offset]);

                    // After FOR_ITER, we've created two branches. Code generation should
                    // NOT continue linearly here. We set insert point to the continue block's
                    // target and let the main loop handle the next instruction.
                    // The orphaned block issue is avoided by NOT creating a new block here.

                    // Set insert point to the continue target (next instruction's block)
                    builder.SetInsertPoint(jump_targets[next_offset]);

                    // Stack state for continue path is already set (with next_item pushed)
                }
            }
            // ========== Exception Handling Opcodes (Bug #3 fix) ==========
            else if (instr.opcode == op::PUSH_EXC_INFO)
            {
                // PUSH_EXC_INFO: At start of exception handler
                // The exception has already been set by Python runtime when we reach here
                // Stack effect: Pushes exc_value (the current exception value)
                //
                // In CPython, this pushes the old exception state and then the new one
                // For JIT, we fetch the current exception and push it

                // Allocate space for PyErr_Fetch outputs
                llvm::Value *type_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_type_ptr");
                llvm::Value *value_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_value_ptr");
                llvm::Value *tb_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_tb_ptr");

                // Initialize to NULL
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                builder.CreateStore(null_ptr, type_ptr);
                builder.CreateStore(null_ptr, value_ptr);
                builder.CreateStore(null_ptr, tb_ptr);

                // Fetch the current exception (clears the error indicator)
                builder.CreateCall(py_err_fetch_func, {type_ptr, value_ptr, tb_ptr});

                // Load the exception value
                llvm::Value *exc_value = builder.CreateLoad(ptr_type, value_ptr, "exc_value");
                llvm::Value *exc_type = builder.CreateLoad(ptr_type, type_ptr, "exc_type");
                llvm::Value *exc_tb = builder.CreateLoad(ptr_type, tb_ptr, "exc_tb");

                // Restore the exception so CHECK_EXC_MATCH can test it
                // PyErr_Restore steals references, so we need to incref first
                builder.CreateCall(py_xincref_func, {exc_type});
                builder.CreateCall(py_xincref_func, {exc_value});
                builder.CreateCall(py_xincref_func, {exc_tb});
                builder.CreateCall(py_err_restore_func, {exc_type, exc_value, exc_tb});

                // Push exc_value onto stack (the exception instance)
                // If exc_value is NULL, use exc_type instead (for bare "raise ExceptionClass")
                llvm::Value *to_push = builder.CreateSelect(
                    builder.CreateICmpNE(exc_value, null_ptr),
                    exc_value,
                    exc_type);

                // Incref since we're pushing a new reference
                builder.CreateCall(py_xincref_func, {to_push});
                stack.push_back(to_push);
            }
            else if (instr.opcode == op::POP_EXCEPT)
            {
                // POP_EXCEPT: End of exception handler - clear the exception state
                // Stack effect: Pops nothing in Python 3.11+
                // Just clears the exception state
                builder.CreateCall(py_err_clear_func, {});
            }
            else if (instr.opcode == op::CHECK_EXC_MATCH)
            {
                // CHECK_EXC_MATCH: Test if TOS1 exception matches TOS type
                // Stack: [..., exc_value, exc_type] -> [..., exc_value, bool_result]
                // Pops the type, pushes True/False
                if (stack.size() >= 2)
                {
                    llvm::Value *exc_type = stack.back();
                    stack.pop_back();                      // Exception type to match against
                    llvm::Value *exc_value = stack.back(); // Exception value (stays on stack)

                    // Get the actual type of the exception
                    llvm::Value *actual_type = builder.CreateCall(py_object_type_func, {exc_value}, "actual_exc_type");

                    // Call PyErr_GivenExceptionMatches(actual_type, exc_type)
                    llvm::Value *match_result = builder.CreateCall(py_exception_matches_func,
                                                                   {actual_type, exc_type}, "exc_match_result");

                    // Decref actual_type (PyObject_Type returns new reference)
                    builder.CreateCall(py_decref_func, {actual_type});

                    // Decref exc_type (we popped it)
                    builder.CreateCall(py_decref_func, {exc_type});

                    // Convert match result (int) to Python bool
                    llvm::Value *is_match = builder.CreateICmpNE(match_result,
                                                                 llvm::ConstantInt::get(builder.getInt32Ty(), 0), "is_match");

                    // Get Py_True or Py_False based on result
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                    llvm::Value *result = builder.CreateSelect(is_match, py_true, py_false, "match_bool");
                    builder.CreateCall(py_incref_func, {result});
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RAISE_VARARGS)
            {
                // RAISE_VARARGS: Raise an exception
                // arg = 0: re-raise current exception
                // arg = 1: raise TOS
                // arg = 2: raise TOS1 from TOS
                int argc = instr.arg;

                if (argc == 0)
                {
                    // Re-raise: the exception should already be set
                    // Just return NULL to propagate
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                }
                else if (argc == 1)
                {
                    // raise exc
                    if (!stack.empty())
                    {
                        llvm::Value *exc = stack.back();
                        stack.pop_back();

                        // Get the type of the exception
                        llvm::Value *exc_type = builder.CreateCall(py_object_type_func, {exc}, "exc_type");

                        // Set the exception: PyErr_SetObject(type, value)
                        builder.CreateCall(py_err_set_object_func, {exc_type, exc});

                        // Decref type and exc (we own them)
                        builder.CreateCall(py_decref_func, {exc_type});
                        builder.CreateCall(py_decref_func, {exc});

                        // Return NULL to signal exception
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    }
                }
                else if (argc == 2)
                {
                    // raise exc from cause
                    if (stack.size() >= 2)
                    {
                        llvm::Value *cause = stack.back();
                        stack.pop_back();
                        llvm::Value *exc = stack.back();
                        stack.pop_back();

                        // Get the type of the exception
                        llvm::Value *exc_type = builder.CreateCall(py_object_type_func, {exc}, "exc_type");

                        // Set the exception
                        builder.CreateCall(py_err_set_object_func, {exc_type, exc});

                        // Set the cause: PyException_SetCause steals reference to cause
                        // But first we need to normalize the exception to get the exception instance
                        // For simplicity, we'll set cause on exc directly if it's an instance
                        builder.CreateCall(py_exception_set_cause_func, {exc, cause});

                        // Decref type (we own it)
                        builder.CreateCall(py_decref_func, {exc_type});
                        // Don't decref cause - PyException_SetCause steals it
                        // Don't decref exc - it's the raised exception

                        // Return NULL to signal exception
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    }
                }
                // RAISE_VARARGS always raises, switch to dead block after all paths
                switch_to_dead_block();
            }
            else if (instr.opcode == op::BEFORE_WITH)
            {
                // BEFORE_WITH: Set up a with block
                // Stack before: context_manager
                // Stack after: __exit__ method, result of __enter__()
                // This loads __exit__ and calls __enter__

                if (!stack.empty())
                {
                    llvm::Value *mgr = stack.back();
                    stack.pop_back();

                    // Get __exit__ method from context manager
                    PyObject *exit_str = PyUnicode_InternFromString("__exit__");
                    Py_INCREF(exit_str);
                    stored_constants.push_back(exit_str);

                    llvm::Value *exit_name_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(exit_str));
                    llvm::Value *exit_name = builder.CreateIntToPtr(exit_name_ptr, ptr_type);
                    llvm::Value *exit_method = builder.CreateCall(py_object_getattr_func, {mgr, exit_name}, "exit_method");

                    // Check for error getting __exit__
                    check_error_and_branch(current_offset, exit_method, "before_with_exit");

                    // Get __enter__ method and call it
                    PyObject *enter_str = PyUnicode_InternFromString("__enter__");
                    Py_INCREF(enter_str);
                    stored_constants.push_back(enter_str);

                    llvm::Value *enter_name_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(enter_str));
                    llvm::Value *enter_name = builder.CreateIntToPtr(enter_name_ptr, ptr_type);
                    llvm::Value *enter_method = builder.CreateCall(py_object_getattr_func, {mgr, enter_name}, "enter_method");

                    // Check for error getting __enter__
                    check_error_and_branch(current_offset, enter_method, "before_with_enter");

                    // Call __enter__(mgr) - it's a bound method
                    llvm::Value *empty_args = builder.CreateCall(py_tuple_new_func, {llvm::ConstantInt::get(i64_type, 0)}, "empty_args");
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *enter_result = builder.CreateCall(py_object_call_func, {enter_method, empty_args, null_kwargs}, "enter_result");

                    builder.CreateCall(py_decref_func, {empty_args});
                    builder.CreateCall(py_decref_func, {enter_method});
                    builder.CreateCall(py_decref_func, {mgr});

                    // Check for error from __enter__
                    check_error_and_branch(current_offset, enter_result, "before_with_call");

                    // Push __exit__ and enter result
                    stack.push_back(exit_method);
                    stack.push_back(enter_result);
                }
            }
            else if (instr.opcode == op::WITH_EXCEPT_START)
            {
                // WITH_EXCEPT_START: Call __exit__ with exception info
                // Stack before: __exit__, exc_type, exc_value, exc_tb (with exc_tb on top)
                // Stack after: result of __exit__(exc_type, exc_value, exc_tb)
                // Note: In Python 3.13, stack has exc_type, exc_val, exc_tb on top after PUSH_EXC_INFO

                if (stack.size() >= 4)
                {
                    // Pop exception info (on top) and __exit__ (below exception info)
                    llvm::Value *exc_tb = stack.back();
                    stack.pop_back();
                    llvm::Value *exc_val = stack.back();
                    stack.pop_back();
                    llvm::Value *exc_type = stack.back();
                    stack.pop_back();
                    llvm::Value *exit_method = stack.back();
                    stack.pop_back();

                    // Build args tuple: (exc_type, exc_val, exc_tb)
                    llvm::Value *args_tuple = builder.CreateCall(py_tuple_new_func, {llvm::ConstantInt::get(i64_type, 3)}, "exit_args");

                    // PyTuple_SetItem steals references, so incref
                    builder.CreateCall(py_incref_func, {exc_type});
                    builder.CreateCall(py_incref_func, {exc_val});
                    builder.CreateCall(py_incref_func, {exc_tb});
                    builder.CreateCall(py_tuple_setitem_func, {args_tuple, llvm::ConstantInt::get(i64_type, 0), exc_type});
                    builder.CreateCall(py_tuple_setitem_func, {args_tuple, llvm::ConstantInt::get(i64_type, 1), exc_val});
                    builder.CreateCall(py_tuple_setitem_func, {args_tuple, llvm::ConstantInt::get(i64_type, 2), exc_tb});

                    // Call __exit__(exc_type, exc_val, exc_tb)
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *result = builder.CreateCall(py_object_call_func, {exit_method, args_tuple, null_kwargs}, "exit_result");

                    builder.CreateCall(py_decref_func, {args_tuple});
                    builder.CreateCall(py_decref_func, {exit_method});

                    // Push exception info back, and result on top
                    stack.push_back(exc_type);
                    stack.push_back(exc_val);
                    stack.push_back(exc_tb);
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RERAISE)
            {
                // RERAISE: Re-raise the current exception
                // The exception should already be set in PyErr state
                // arg determines if traceback should be modified:
                //   0 = keep original traceback
                //   1 = add current location to traceback
                // For JIT, we just return NULL to propagate the exception
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                // Switch to dead block to prevent code after terminator
                switch_to_dead_block();
            }
        }

        // Ensure current block has terminator
        if (!builder.GetInsertBlock()->getTerminator())
        {
            if (!stack.empty())
            {
                llvm::Value *ret_val = stack.back();
                // If returning i64, convert to PyObject*
                if (ret_val->getType()->isIntegerTy(64))
                {
                    ret_val = builder.CreateCall(py_long_fromlonglong_func, {ret_val});
                }
                builder.CreateRet(ret_val);
            }
            else
            {
                // Return None
                llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                builder.CreateCall(py_incref_func, {py_none});
                builder.CreateRet(py_none);
            }
        }

        // Third pass: Add terminators to any unterminated blocks
        for (auto &block : *func)
        {
            if (!block.getTerminator())
            {
                builder.SetInsertPoint(&block);
                // Return None for unterminated blocks
                llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                builder.CreateCall(py_incref_func, {py_none});
                builder.CreateRet(py_none);
            }
        }

        if (llvm::verifyFunction(*func, &llvm::errs()))
        {
            llvm::errs() << "Function verification failed\n";
            func->print(llvm::errs());
            return false;
        }

        optimize_module(*module, func);

        // Capture IR if dump_ir is enabled
        if (dump_ir)
        {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            ir_stream.flush();
            last_ir = ir_str;
        }

        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(local_context));

        auto err = jit->addIRModule(std::move(tsm));
        if (err)
        {
            llvm::errs() << "Failed to add module: " << toString(std::move(err)) << "\n";
            return false;
        }

        // Mark as compiled to prevent duplicate symbol errors on subsequent calls
        compiled_functions.insert(name);
        return true;
    }

    uint64_t JITCore::lookup_symbol(const std::string &name)
    {
        if (!jit)
        {
            return 0;
        }

        auto symbol = jit->lookup(name);
        if (!symbol)
        {
            llvm::errs() << "Failed to lookup symbol: " << toString(symbol.takeError()) << "\n";
            return 0;
        }

        return symbol->getValue();
    }

    void JITCore::optimize_module(llvm::Module &module, llvm::Function *func)
    {
        if (opt_level == 0)
        {
            return;
        }

        llvm::PassBuilder PB;
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        llvm::OptimizationLevel opt_lvl;
        switch (opt_level)
        {
        case 1:
            opt_lvl = llvm::OptimizationLevel::O1;
            break;
        case 2:
            opt_lvl = llvm::OptimizationLevel::O2;
            break;
        case 3:
            opt_lvl = llvm::OptimizationLevel::O3;
            break;
        default:
            opt_lvl = llvm::OptimizationLevel::O0;
            break;
        }

        llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_lvl);
        MPM.run(module, MAM);
    }

    // Implementation of callable creation helper methods
    // PyObject* versions for object mode functions
    nb::object JITCore::create_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<PyObject *(*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> nb::object
                                {
                                    PyObject *result = fn_ptr();
                                    if (!result)
                                    {
                                        throw std::runtime_error("JIT function returned NULL");
                                    }
                                    return nb::steal(result); // Transfer ownership to nanobind
                                });
    }

    nb::object JITCore::create_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<PyObject *(*)(PyObject *)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a) -> nb::object
                                {
                                    PyObject *result = fn_ptr(a.ptr());
                                    if (!result)
                                    {
                                        throw std::runtime_error("JIT function returned NULL");
                                    }
                                    return nb::steal(result); // Transfer ownership to nanobind
                                });
    }

    nb::object JITCore::create_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<PyObject *(*)(PyObject *, PyObject *)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a, nb::object b) -> nb::object
                                {
                                    PyObject *result = fn_ptr(a.ptr(), b.ptr());
                                    if (!result)
                                    {
                                        throw std::runtime_error("JIT function returned NULL");
                                    }
                                    return nb::steal(result); // Transfer ownership to nanobind
                                });
    }

    nb::object JITCore::create_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<PyObject *(*)(PyObject *, PyObject *, PyObject *)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a, nb::object b, nb::object c) -> nb::object
                                {
                                    PyObject *result = fn_ptr(a.ptr(), b.ptr(), c.ptr());
                                    if (!result)
                                    {
                                        throw std::runtime_error("JIT function returned NULL");
                                    }
                                    return nb::steal(result); // Transfer ownership to nanobind
                                });
    }

    nb::object JITCore::create_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<PyObject *(*)(PyObject *, PyObject *, PyObject *, PyObject *)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a, nb::object b, nb::object c, nb::object d) -> nb::object
                                {
                                    PyObject *result = fn_ptr(a.ptr(), b.ptr(), c.ptr(), d.ptr());
                                    if (!result)
                                    {
                                        throw std::runtime_error("JIT function returned NULL");
                                    }
                                    return nb::steal(result); // Transfer ownership to nanobind
                                });
    }

    // Integer-mode callable generators (native i64 -> i64 functions)
    // These bypass PyObject* entirely for maximum performance
    nb::object JITCore::create_int_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> int64_t
                                { return fn_ptr(); });
    }

    nb::object JITCore::create_int_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int64_t a) -> int64_t
                                { return fn_ptr(a); });
    }

    nb::object JITCore::create_int_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int64_t a, int64_t b) -> int64_t
                                { return fn_ptr(a, b); });
    }

    nb::object JITCore::create_int_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int64_t a, int64_t b, int64_t c) -> int64_t
                                { return fn_ptr(a, b, c); });
    }

    nb::object JITCore::create_int_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int64_t a, int64_t b, int64_t c, int64_t d) -> int64_t
                                { return fn_ptr(a, b, c, d); });
    }

    nb::object JITCore::get_int_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_int_callable_0(func_ptr);
        case 1:
            return create_int_callable_1(func_ptr);
        case 2:
            return create_int_callable_2(func_ptr);
        case 3:
            return create_int_callable_3(func_ptr);
        case 4:
            return create_int_callable_4(func_ptr);
        default:
            throw std::runtime_error("Integer mode supports up to 4 parameters");
        }
    }

    bool JITCore::compile_int_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit)
        {
            return false;
        }

        // Check if already compiled to prevent duplicate symbol errors
        if (compiled_functions.count(name) > 0)
        {
            return true; // Already compiled, return success
        }

        // Convert Python instructions list to C++ vector
        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i)
        {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Extract integer constants
        std::vector<int64_t> int_constants;
        for (size_t i = 0; i < py_constants.size(); ++i)
        {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::int_>(const_obj))
            {
                int_constants.push_back(nb::cast<int64_t>(const_obj));
            }
            else
            {
                int_constants.push_back(0); // Non-integer constants default to 0
            }
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);

        // Create function type - all i64 for integer mode
        std::vector<llvm::Type *> param_types(param_count, i64_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(
            i64_type, // Return i64
            param_types,
            false);

        llvm::Function *func = llvm::Function::Create(
            func_type,
            llvm::Function::ExternalLinkage,
            name,
            module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        std::unordered_map<int, llvm::BasicBlock *> jump_targets;

        // Create i64 allocas for all locals
        llvm::IRBuilder<> alloca_builder(entry, entry->begin());
        for (int i = 0; i < total_locals; ++i)
        {
            local_allocas[i] = alloca_builder.CreateAlloca(
                i64_type, nullptr, "local_" + std::to_string(i));
        }

        // Store function parameters into allocas (already i64)
        auto args = func->arg_begin();
        for (int i = 0; i < param_count; ++i)
        {
            builder.CreateStore(&*args++, local_allocas[i]);
        }

        // First pass: Check for unsupported opcodes and create basic blocks for jump targets
        // Integer mode doesn't support Python object operations like FOR_ITER, CALL, etc.
        static const std::unordered_set<uint8_t> supported_int_opcodes = {
            op::RESUME, op::LOAD_FAST, op::LOAD_FAST_LOAD_FAST, op::LOAD_CONST,
            op::STORE_FAST, op::BINARY_OP, op::UNARY_NEGATIVE, op::COMPARE_OP,
            op::POP_JUMP_IF_FALSE, op::POP_JUMP_IF_TRUE, op::RETURN_VALUE, op::RETURN_CONST,
            op::POP_TOP, op::JUMP_BACKWARD, op::JUMP_FORWARD, op::COPY,
            op::NOP, op::CACHE
        };
        
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            if (supported_int_opcodes.find(instr.opcode) == supported_int_opcodes.end())
            {
                // Unsupported opcode for integer mode
                llvm::errs() << "Integer mode: unsupported opcode " << static_cast<int>(instr.opcode) 
                             << " at offset " << instr.offset << ". Use mode='generic' or mode='auto'.\n";
                return false;
            }
        }
        
        jump_targets[0] = entry;
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(target_offset), func);
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_header_" + std::to_string(target_offset), func);
                }
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "forward_" + std::to_string(target_offset), func);
                }
            }
        }

        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            // Handle jump targets
            if (jump_targets.count(instructions[i].offset) && jump_targets[instructions[i].offset] != builder.GetInsertBlock())
            {
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[instructions[i].offset]);
                }
                builder.SetInsertPoint(jump_targets[instructions[i].offset]);
            }

            const auto &instr = instructions[i];

            if (instr.opcode == op::RESUME)
            {
                continue;
            }
            else if (instr.opcode == op::LOAD_FAST)
            {
                if (local_allocas.count(instr.arg))
                {
                    llvm::Value *loaded = builder.CreateLoad(i64_type, local_allocas[instr.arg], "load_" + std::to_string(instr.arg));
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                if (local_allocas.count(first_local))
                {
                    stack.push_back(builder.CreateLoad(i64_type, local_allocas[first_local], "load_" + std::to_string(first_local)));
                }
                if (local_allocas.count(second_local))
                {
                    stack.push_back(builder.CreateLoad(i64_type, local_allocas[second_local], "load_" + std::to_string(second_local)));
                }
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                if (instr.arg < int_constants.size())
                {
                    llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                    stack.push_back(const_val);
                }
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                if (!stack.empty())
                {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *second = stack.back();
                    stack.pop_back();
                    llvm::Value *first = stack.back();
                    stack.pop_back();
                    llvm::Value *result = nullptr;

                    switch (instr.arg)
                    {
                    case 0: // ADD
                        result = builder.CreateAdd(first, second, "add");
                        break;
                    case 10: // SUB
                        result = builder.CreateSub(first, second, "sub");
                        break;
                    case 5: // MUL
                        result = builder.CreateMul(first, second, "mul");
                        break;
                    case 11: // TRUE_DIV
                    case 2:  // FLOOR_DIV
                    case 6:
                    { // MOD
                        // Check for division by zero
                        llvm::Value *is_zero = builder.CreateICmpEQ(
                            second,
                            llvm::ConstantInt::get(i64_type, 0),
                            "div_by_zero_check");

                        llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                            *local_context, "div_by_zero_error_" + std::to_string(i), func);
                        llvm::BasicBlock *safe_block = llvm::BasicBlock::Create(
                            *local_context, "div_safe_" + std::to_string(i), func);

                        builder.CreateCondBr(is_zero, error_block, safe_block);

                        // Error path: return INT64_MIN to signal division by zero error
                        // The Python wrapper should check for this and raise ZeroDivisionError
                        builder.SetInsertPoint(error_block);
                        builder.CreateRet(llvm::ConstantInt::get(i64_type, INT64_MIN));

                        // Safe path: perform division
                        builder.SetInsertPoint(safe_block);
                        if (instr.arg == 11)
                        {
                            result = builder.CreateSDiv(first, second, "div");
                        }
                        else if (instr.arg == 2)
                        {
                            result = builder.CreateSDiv(first, second, "floordiv");
                        }
                        else
                        { // instr.arg == 6
                            result = builder.CreateSRem(first, second, "mod");
                        }
                        break;
                    }
                    case 1: // AND
                        result = builder.CreateAnd(first, second, "and");
                        break;
                    case 7: // OR
                        result = builder.CreateOr(first, second, "or");
                        break;
                    case 12: // XOR
                        result = builder.CreateXor(first, second, "xor");
                        break;
                    case 3: // LSHIFT
                        result = builder.CreateShl(first, second, "shl");
                        break;
                    case 9: // RSHIFT
                        result = builder.CreateAShr(first, second, "shr");
                        break;
                    case 8:  // POW
                    case 21: // INPLACE_POW
                    {
                        // Implement iterative binary exponentiation
                        llvm::Function *current_func = builder.GetInsertBlock()->getParent();
                        llvm::BasicBlock *pow_entry = builder.GetInsertBlock();
                        llvm::BasicBlock *pow_loop = llvm::BasicBlock::Create(*local_context, "pow_loop", current_func);
                        llvm::BasicBlock *pow_odd = llvm::BasicBlock::Create(*local_context, "pow_odd", current_func);
                        llvm::BasicBlock *pow_cont = llvm::BasicBlock::Create(*local_context, "pow_cont", current_func);
                        llvm::BasicBlock *pow_done = llvm::BasicBlock::Create(*local_context, "pow_done", current_func);

                        llvm::Value *init_result = llvm::ConstantInt::get(i64_type, 1);
                        builder.CreateBr(pow_loop);

                        builder.SetInsertPoint(pow_loop);
                        llvm::PHINode *phi_result = builder.CreatePHI(i64_type, 2, "pow_result");
                        llvm::PHINode *phi_base = builder.CreatePHI(i64_type, 2, "pow_base");
                        llvm::PHINode *phi_exp = builder.CreatePHI(i64_type, 2, "pow_exp");

                        phi_result->addIncoming(init_result, pow_entry);
                        phi_base->addIncoming(first, pow_entry);
                        phi_exp->addIncoming(second, pow_entry);

                        llvm::Value *exp_gt_zero = builder.CreateICmpSGT(phi_exp, llvm::ConstantInt::get(i64_type, 0));
                        builder.CreateCondBr(exp_gt_zero, pow_odd, pow_done);

                        builder.SetInsertPoint(pow_odd);
                        llvm::Value *exp_is_odd = builder.CreateAnd(phi_exp, llvm::ConstantInt::get(i64_type, 1));
                        llvm::Value *is_odd = builder.CreateICmpNE(exp_is_odd, llvm::ConstantInt::get(i64_type, 0));
                        llvm::Value *result_times_base = builder.CreateMul(phi_result, phi_base);
                        llvm::Value *new_result = builder.CreateSelect(is_odd, result_times_base, phi_result);
                        llvm::Value *new_base = builder.CreateMul(phi_base, phi_base);
                        llvm::Value *new_exp = builder.CreateAShr(phi_exp, llvm::ConstantInt::get(i64_type, 1));
                        builder.CreateBr(pow_cont);

                        builder.SetInsertPoint(pow_cont);
                        phi_result->addIncoming(new_result, pow_cont);
                        phi_base->addIncoming(new_base, pow_cont);
                        phi_exp->addIncoming(new_exp, pow_cont);
                        builder.CreateBr(pow_loop);

                        builder.SetInsertPoint(pow_done);
                        result = phi_result;
                        break;
                    }
                    default:
                        // Unsupported op - return special error value
                        // In pure int mode, we return INT64_MIN to indicate error
                        result = llvm::ConstantInt::get(i64_type, INT64_MIN);
                        break;
                    }
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::UNARY_NEGATIVE)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    llvm::Value *result = builder.CreateNeg(val, "neg");
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::COMPARE_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back();
                    stack.pop_back();
                    llvm::Value *lhs = stack.back();
                    stack.pop_back();

                    // Python 3.13 encoding: (op_code << 5) | flags
                    // Extraction: op_code = arg >> 5
                    int op_code = instr.arg >> 5;
                    llvm::Value *cmp_result = nullptr;

                    switch (op_code)
                    {
                    case 0: // <
                        cmp_result = builder.CreateICmpSLT(lhs, rhs, "lt");
                        break;
                    case 1: // <=
                        cmp_result = builder.CreateICmpSLE(lhs, rhs, "le");
                        break;
                    case 2: // ==
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                        break;
                    case 3: // !=
                        cmp_result = builder.CreateICmpNE(lhs, rhs, "ne");
                        break;
                    case 4: // >
                        cmp_result = builder.CreateICmpSGT(lhs, rhs, "gt");
                        break;
                    case 5: // >=
                        cmp_result = builder.CreateICmpSGE(lhs, rhs, "ge");
                        break;
                    default:
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                        break;
                    }
                    // Zero-extend i1 to i64
                    llvm::Value *result = builder.CreateZExt(cmp_result, i64_type, "cmp_ext");
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                if (!stack.empty() && i + 1 < instructions.size())
                {
                    llvm::Value *cond = stack.back();
                    stack.pop_back();

                    // Compare to zero for truthiness
                    llvm::Value *bool_cond = builder.CreateICmpNE(
                        cond, llvm::ConstantInt::get(i64_type, 0), "tobool");

                    int target_offset = instr.argval;
                    int next_offset = instructions[i + 1].offset;

                    if (!jump_targets.count(target_offset))
                    {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(target_offset), func);
                    }
                    if (!jump_targets.count(next_offset))
                    {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(next_offset), func);
                    }

                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        if (instr.opcode == op::POP_JUMP_IF_FALSE)
                        {
                            builder.CreateCondBr(bool_cond, jump_targets[next_offset], jump_targets[target_offset]);
                        }
                        else
                        { // POP_JUMP_IF_TRUE
                            builder.CreateCondBr(bool_cond, jump_targets[target_offset], jump_targets[next_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::RETURN_CONST)
            {
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    if (instr.arg < int_constants.size())
                    {
                        llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                        builder.CreateRet(const_val);
                    }
                    else
                    {
                        builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
                    }
                }
            }
            else if (instr.opcode == op::RETURN_VALUE)
            {
                if (!stack.empty() && !builder.GetInsertBlock()->getTerminator())
                {
                    llvm::Value *ret_val = stack.back();
                    stack.pop_back();
                    builder.CreateRet(ret_val);
                }
            }
            else if (instr.opcode == op::POP_TOP)
            {
                if (!stack.empty())
                {
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                // Jump back to loop header
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_header_" + std::to_string(target_offset), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }
                // Create a new block for any code after the loop (unreachable but needed)
                llvm::BasicBlock *after_loop = llvm::BasicBlock::Create(
                    *local_context, "after_loop_" + std::to_string(i), func);
                builder.SetInsertPoint(after_loop);
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
                // Unconditional forward jump
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset))
                {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "forward_" + std::to_string(target_offset), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }
                // Create a new block for any code after the jump (unreachable but needed)
                llvm::BasicBlock *after_jump = llvm::BasicBlock::Create(
                    *local_context, "after_jump_" + std::to_string(i), func);
                builder.SetInsertPoint(after_jump);
            }
        }

        // Ensure function has a return
        if (!builder.GetInsertBlock()->getTerminator())
        {
            builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
        }

        // Optimize
        optimize_module(*module, func);

        // Add to JIT
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err)
        {
            llvm::errs() << "Failed to add module: " << toString(std::move(err)) << "\n";
            return false;
        }

        // Mark as compiled to prevent duplicate symbol errors on subsequent calls
        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Generator Compilation
    // =========================================================================
    // Compiles a generator function into a state machine step function.
    // The step function signature is:
    //   PyObject* step_func(int32_t* state, PyObject** locals, PyObject* sent_value)
    // 
    // State encoding:
    //   0 = initial (not started, ignore sent_value)
    //   1..N = resume after yield N
    //   -1 = completed (returned)
    //   -2 = error
    //
    // The step function:
    //   1. Switches on *state to jump to the correct resume point
    //   2. Executes bytecode until YIELD_VALUE or RETURN_VALUE
    //   3. On YIELD_VALUE: stores locals, updates *state, returns yielded value
    //   4. On RETURN_VALUE: sets *state = -1, returns the return value
    // =========================================================================

    bool JITCore::compile_generator(nb::list py_instructions, nb::list py_constants, nb::list py_names,
                                    nb::object py_globals_dict, nb::object py_builtins_dict,
                                    nb::list py_closure_cells, nb::list py_exception_table,
                                    const std::string &name, int param_count, int total_locals, int nlocals)
    {
        if (!jit)
        {
            return false;
        }

        // Check if already compiled
        std::string step_name = name + "_step";
        if (compiled_functions.count(step_name) > 0)
        {
            return true;
        }

        // Store globals and builtins for runtime lookup
        globals_dict_ptr = py_globals_dict.ptr();
        Py_INCREF(globals_dict_ptr);
        builtins_dict_ptr = py_builtins_dict.ptr();
        Py_INCREF(builtins_dict_ptr);

        // Convert Python instructions to C++ vector
        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i)
        {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse exception table for try/except handling in generators
        std::vector<ExceptionTableEntry> exception_table;
        for (size_t i = 0; i < py_exception_table.size(); ++i)
        {
            nb::dict entry_dict = nb::cast<nb::dict>(py_exception_table[i]);
            ExceptionTableEntry entry;
            entry.start = nb::cast<int32_t>(entry_dict["start"]);
            entry.end = nb::cast<int32_t>(entry_dict["end"]);
            entry.target = nb::cast<int32_t>(entry_dict["target"]);
            entry.depth = nb::cast<int32_t>(entry_dict["depth"]);
            entry.lasti = nb::cast<bool>(entry_dict["lasti"]);
            exception_table.push_back(entry);
        }

        // Find all YIELD_VALUE instructions and assign state numbers
        // Also track stack depth at each yield for restoration
        std::vector<size_t> yield_indices;
        std::unordered_map<size_t, int> yield_to_state;
        std::unordered_map<size_t, size_t> yield_stack_depth; // Stack depth at each yield
        int next_state = 1;
        
        // First pass: simulate stack depth to track depth at each yield
        size_t simulated_depth = 0;
        size_t max_stack_depth = 0;  // Track maximum stack depth for bounds checking
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            
            // Update simulated stack depth based on opcode effects
            if (instr.opcode == op::LOAD_CONST || instr.opcode == op::LOAD_FAST ||
                instr.opcode == op::LOAD_FAST_CHECK || instr.opcode == op::LOAD_ATTR ||
                instr.opcode == op::PUSH_NULL) {
                simulated_depth++;
            } else if (instr.opcode == op::LOAD_GLOBAL) {
                // LOAD_GLOBAL pushes 1 value, optionally pushes NULL too
                bool push_null = (instr.arg & 1) != 0;
                simulated_depth++;
                if (push_null) simulated_depth++;
            } else if (instr.opcode == op::CALL) {
                // CALL pops (callable + self_or_null + args), pushes 1 result
                int num_args = instr.arg;
                if (simulated_depth >= static_cast<size_t>(num_args + 2)) {
                    simulated_depth -= (num_args + 2);
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++; // result
            } else if (instr.opcode == op::STORE_FAST || instr.opcode == op::POP_TOP ||
                       instr.opcode == op::STORE_SUBSCR) {
                if (simulated_depth > 0) simulated_depth--;
            } else if (instr.opcode == op::BINARY_OP || instr.opcode == op::BINARY_SUBSCR) {
                if (simulated_depth >= 2) simulated_depth--; // 2 operands -> 1 result
            } else if (instr.opcode == op::COMPARE_OP) {
                if (simulated_depth >= 2) simulated_depth--;
            } else if (instr.opcode == op::UNARY_NEGATIVE || instr.opcode == op::UNARY_NOT ||
                       instr.opcode == op::UNARY_INVERT) {
                // Unary ops: 1 in, 1 out (no change)
            } else if (instr.opcode == op::GET_ITER || instr.opcode == op::GET_AWAITABLE) {
                // GET_ITER/GET_AWAITABLE: 1 in, 1 out (no change)
            } else if (instr.opcode == op::FOR_ITER) {
                simulated_depth++; // Pushes next value (iterator stays on stack)
            } else if (instr.opcode == op::END_FOR) {
                if (simulated_depth >= 2) simulated_depth -= 2; // Pops value and iterator
            } else if (instr.opcode == op::COPY) {
                simulated_depth++; // Duplicates stack value
            } else if (instr.opcode == op::SWAP) {
                // No depth change
            } else if (instr.opcode == op::BUILD_LIST || instr.opcode == op::BUILD_TUPLE) {
                // BUILD_* pops N items, pushes 1
                if (simulated_depth >= static_cast<size_t>(instr.arg)) {
                    simulated_depth -= instr.arg;
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::BUILD_CONST_KEY_MAP) {
                // Pops arg values + 1 keys tuple, pushes 1 dict
                if (simulated_depth >= static_cast<size_t>(instr.arg + 1)) {
                    simulated_depth -= (instr.arg + 1);
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::SEND) {
                // SEND: pops value, keeps receiver, may push result
                if (simulated_depth > 0) simulated_depth--;  // Pop value
                simulated_depth++;  // Push result
            } else if (instr.opcode == op::END_SEND) {
                // END_SEND: pops receiver and result, pushes result
                if (simulated_depth >= 2) simulated_depth--;  // Net: -1
            }
            
            // Track maximum depth
            if (simulated_depth > max_stack_depth) {
                max_stack_depth = simulated_depth;
            }
            
            if (instr.opcode == op::YIELD_VALUE)
            {
                yield_indices.push_back(i);
                yield_to_state[i] = next_state++;
                // At yield, one value (the yielded value) is popped, so remaining depth is depth-1
                // After yield returns, sent_value is pushed, so we restore depth-1 then add sent
                yield_stack_depth[i] = simulated_depth > 0 ? simulated_depth - 1 : 0;
            }
        }
        
        // Stack base: where we persist stack values in the locals array
        // Layout: [0..nlocals) = locals, [nlocals..total_locals) = stack persistence slots
        size_t stack_base = static_cast<size_t>(nlocals);
        size_t max_stack_slots = static_cast<size_t>(total_locals - nlocals);  // Available slots for stack
        
        // Safety check: ensure we have enough slots for the stack
        if (max_stack_depth > max_stack_slots) {
            // Not enough stack slots - this shouldn't happen with correct bytecode
            // but we check to prevent buffer overflow
            return false;
        }

        // Convert constants
        std::vector<int64_t> int_constants;
        std::vector<PyObject *> obj_constants;
        for (size_t i = 0; i < py_constants.size(); ++i)
        {
            nb::object const_obj = py_constants[i];
            PyObject *py_obj = const_obj.ptr();

            if (py_obj == Py_True || py_obj == Py_False)
            {
                int_constants.push_back(0);
                Py_INCREF(py_obj);
                obj_constants.push_back(py_obj);
                stored_constants.push_back(py_obj);
            }
            else if (PyLong_Check(py_obj))
            {
                try
                {
                    int64_t int_val = nb::cast<int64_t>(const_obj);
                    int_constants.push_back(int_val);
                    obj_constants.push_back(nullptr);
                }
                catch (...)
                {
                    int_constants.push_back(0);
                    Py_INCREF(py_obj);
                    obj_constants.push_back(py_obj);
                    stored_constants.push_back(py_obj);
                }
            }
            else
            {
                int_constants.push_back(0);
                Py_INCREF(py_obj);
                obj_constants.push_back(py_obj);
                stored_constants.push_back(py_obj);
            }
        }

        // Extract names
        std::vector<PyObject *> name_objects;
        for (size_t i = 0; i < py_names.size(); ++i)
        {
            nb::object name_obj = py_names[i];
            PyObject *py_name = name_obj.ptr();
            Py_INCREF(py_name);
            name_objects.push_back(py_name);
            stored_names.push_back(py_name);
        }

        // Extract closure cells
        std::vector<PyObject *> closure_cells;
        for (size_t i = 0; i < py_closure_cells.size(); ++i)
        {
            nb::object cell_obj = py_closure_cells[i];
            if (cell_obj.is_none())
            {
                closure_cells.push_back(nullptr);
            }
            else
            {
                PyObject *py_cell = cell_obj.ptr();
                Py_INCREF(py_cell);
                closure_cells.push_back(py_cell);
                stored_closure_cells.push_back(py_cell);
            }
        }

        // Create LLVM module
        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(step_name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        declare_python_api_functions(module.get(), &builder);

        llvm::Type *i32_type = llvm::Type::getInt32Ty(*local_context);
        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);
        llvm::Type *ptr_type = builder.getPtrTy();

        // Step function signature: PyObject* step(int32_t* state, PyObject** locals, PyObject* sent_value)
        std::vector<llvm::Type *> param_types = {ptr_type, ptr_type, ptr_type};
        llvm::FunctionType *func_type = llvm::FunctionType::get(ptr_type, param_types, false);

        llvm::Function *func = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage, step_name, module.get());

        auto args = func->arg_begin();
        llvm::Value *state_ptr = &*args++;
        llvm::Value *locals_array = &*args++;
        llvm::Value *sent_value = &*args++;

        // Create blocks for state machine
        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        llvm::BasicBlock *state_error = llvm::BasicBlock::Create(*local_context, "state_error", func);
        llvm::BasicBlock *gen_done = llvm::BasicBlock::Create(*local_context, "gen_done", func);

        // Create a block for initial state (state 0)
        llvm::BasicBlock *state_0 = llvm::BasicBlock::Create(*local_context, "state_0", func);

        // Create blocks for each resume point (after each yield)
        std::vector<llvm::BasicBlock *> resume_blocks;
        for (size_t i = 0; i < yield_indices.size(); ++i)
        {
            resume_blocks.push_back(llvm::BasicBlock::Create(
                *local_context, "resume_" + std::to_string(i + 1), func));
        }

        // Entry block: load state and switch
        builder.SetInsertPoint(entry);
        llvm::Value *state_val = builder.CreateLoad(i32_type, state_ptr, "state");

        // Create switch instruction
        llvm::SwitchInst *state_switch = builder.CreateSwitch(state_val, state_error, 
                                                              1 + yield_indices.size());
        state_switch->addCase(llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(i32_type, 0)), state_0);
        for (size_t i = 0; i < resume_blocks.size(); ++i)
        {
            state_switch->addCase(llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(i32_type, i + 1)), resume_blocks[i]);
        }

        // State error block: generator already exhausted or error
        builder.SetInsertPoint(state_error);
        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

        // Generator done block: set state to -1 and return
        builder.SetInsertPoint(gen_done);
        builder.CreateStore(llvm::ConstantInt::get(i32_type, -1), state_ptr);
        // Return None (the actual return value will be set by the calling code)
        llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
        llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
        builder.CreateCall(py_incref_func, {py_none});
        builder.CreateRet(py_none);

        // Now generate code for state_0 (initial execution)
        builder.SetInsertPoint(state_0);

        // Stack for operands (within a single execution slice)
        std::vector<llvm::Value *> stack;

        // Lambda to load a local from the locals array
        auto load_local = [&](int index) -> llvm::Value * {
            llvm::Value *idx = llvm::ConstantInt::get(i64_type, index);
            llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, idx);
            return builder.CreateLoad(ptr_type, slot_ptr);
        };

        // Lambda to store a local to the locals array
        auto store_local = [&](int index, llvm::Value *value) {
            llvm::Value *idx = llvm::ConstantInt::get(i64_type, index);
            llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, idx);
            builder.CreateStore(value, slot_ptr);
        };

        // Create basic blocks for bytecode offsets
        std::unordered_map<int, llvm::BasicBlock *> offset_blocks;

        // First pass: identify all jump targets
        std::unordered_set<int> jump_targets;
        for (const auto &instr : instructions)
        {
            if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE ||
                instr.opcode == op::POP_JUMP_IF_NONE || instr.opcode == op::POP_JUMP_IF_NOT_NONE ||
                instr.opcode == op::JUMP_FORWARD || instr.opcode == op::JUMP_BACKWARD ||
                instr.opcode == op::JUMP_BACKWARD_NO_INTERRUPT)
            {
                jump_targets.insert(instr.argval);
            }
        }

        // Create blocks for jump targets
        for (int target : jump_targets)
        {
            offset_blocks[target] = llvm::BasicBlock::Create(
                *local_context, "offset_" + std::to_string(target), func);
        }

        // Create blocks for exception handler targets from exception table
        std::unordered_map<int, llvm::BasicBlock *> exception_handlers;
        std::unordered_map<int, int> exception_handler_depth;
        for (const auto &exc_entry : exception_table)
        {
            if (!offset_blocks.count(exc_entry.target))
            {
                offset_blocks[exc_entry.target] = llvm::BasicBlock::Create(
                    *local_context, "exc_handler_" + std::to_string(exc_entry.target), func);
            }
            exception_handlers[exc_entry.target] = offset_blocks[exc_entry.target];
            exception_handler_depth[exc_entry.target] = exc_entry.depth;
        }

        // Build a map from instruction offset to exception handler
        std::unordered_map<int, int> offset_to_handler;
        for (const auto &exc_entry : exception_table)
        {
            for (int off = exc_entry.start; off < exc_entry.end; off += 2)
            {
                if (!offset_to_handler.count(off))
                {
                    offset_to_handler[off] = exc_entry.target;
                }
            }
        }

        // Map yield indices to their instruction index (for creating yield blocks)
        std::unordered_map<size_t, size_t> instr_idx_to_yield_idx;
        for (size_t i = 0; i < yield_indices.size(); ++i)
        {
            instr_idx_to_yield_idx[yield_indices[i]] = i;
        }

        // Track the current block for resume points
        llvm::BasicBlock *current_block = state_0;
        size_t current_yield_idx = 0;

        // Skip RETURN_GENERATOR (first instruction in generators)
        size_t start_idx = 0;
        if (!instructions.empty() && instructions[0].opcode == op::RETURN_GENERATOR)
        {
            start_idx = 1;
        }

        // Helper lambda to generate error checking code after API calls for generators
        // If an error occurred (result is NULL), branch to exception handler or return NULL
        auto check_error_and_branch_gen = [&](int current_offset, llvm::Value *result, const char *call_name)
        {
            // Check if this offset has an exception handler
            if (offset_to_handler.count(current_offset))
            {
                int handler_offset = offset_to_handler[current_offset];

                // Create blocks for error path and continue path
                llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_error_" + std::to_string(current_offset), func);
                llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_continue_" + std::to_string(current_offset), func);

                // Check if result is NULL (error occurred)
                llvm::Value *is_error = builder.CreateICmpEQ(
                    result,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                    "is_error");
                builder.CreateCondBr(is_error, error_block, continue_block);

                // Error path: branch to exception handler
                builder.SetInsertPoint(error_block);

                // Stack unwinding: decref all values on the stack that are PyObject*
                int target_depth = exception_handler_depth.count(handler_offset) 
                    ? exception_handler_depth[handler_offset] : 0;

                // Decref stack values above target depth
                for (size_t s = stack.size(); s > static_cast<size_t>(target_depth); --s)
                {
                    llvm::Value *val = stack[s - 1];
                    if (val->getType()->isPointerTy())
                    {
                        // Check not NULL before decref
                        llvm::Value *is_null = builder.CreateICmpEQ(
                            val,
                            llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                            "is_null");
                        llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(
                            *local_context, "decref_unwind", func);
                        llvm::BasicBlock *after_decref = llvm::BasicBlock::Create(
                            *local_context, "after_decref_unwind", func);
                        builder.CreateCondBr(is_null, after_decref, decref_block);
                        builder.SetInsertPoint(decref_block);
                        builder.CreateCall(py_decref_func, {val});
                        builder.CreateBr(after_decref);
                        builder.SetInsertPoint(after_decref);
                    }
                }

                // Branch to handler
                if (offset_blocks.count(handler_offset)) {
                    builder.CreateBr(offset_blocks[handler_offset]);
                } else {
                    // Handler block doesn't exist, return NULL
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                }

                // Continue on success path
                builder.SetInsertPoint(continue_block);
                current_block = continue_block;
            }
            else
            {
                // No exception handler: if error, return NULL
                llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_error_ret_" + std::to_string(current_offset), func);
                llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                    *local_context, std::string(call_name) + "_continue_ret_" + std::to_string(current_offset), func);

                llvm::Value *is_error = builder.CreateICmpEQ(
                    result,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                    "is_error");
                builder.CreateCondBr(is_error, error_block, continue_block);

                builder.SetInsertPoint(error_block);
                builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                builder.SetInsertPoint(continue_block);
                current_block = continue_block;
            }
        };

        // Second pass: generate code
        // For generators, we need special handling at basic block boundaries:
        // Jump target blocks are only entered via jumps, not fallthrough.
        // When we reach a jump target via fallthrough, we branch to it.
        // When we reach it via a jump, the stack is restored from persistent storage.
        
        // Track expected stack depth at each jump target
        std::unordered_map<int, size_t> target_stack_depth;
        
        // Track which blocks have been initialized with stack reload code
        std::unordered_set<llvm::BasicBlock*> initialized_blocks;
        initialized_blocks.insert(state_0);  // Initial block is already initialized
        
        for (size_t i = start_idx; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];

            // Check if this offset is a jump target - need new block
            if (offset_blocks.count(instr.offset))
            {
                // If we're not already in this block, we need to branch to it
                if (offset_blocks[instr.offset] != current_block)
                {
                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        // Spill stack to persistent storage before branching
                        for (size_t j = 0; j < stack.size(); ++j)
                        {
                            llvm::Value *val = stack[j];
                            builder.CreateCall(py_incref_func, {val});
                            llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                            llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                            builder.CreateStore(val, slot_ptr);
                        }
                        // Record expected depth for this target
                        if (!target_stack_depth.count(instr.offset)) {
                            target_stack_depth[instr.offset] = stack.size();
                        }
                        builder.CreateBr(offset_blocks[instr.offset]);
                    }
                    
                    // Switch to target block
                    current_block = offset_blocks[instr.offset];
                    builder.SetInsertPoint(current_block);
                    
                    // Only reload if the block hasn't been initialized yet
                    if (initialized_blocks.find(current_block) == initialized_blocks.end())
                    {
                        initialized_blocks.insert(current_block);
                        
                        // Reload stack from persistent storage
                        size_t expected_depth = target_stack_depth.count(instr.offset) 
                            ? target_stack_depth[instr.offset] : 0;
                        stack.clear();
                        for (size_t j = 0; j < expected_depth; ++j)
                        {
                            llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                            llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                            llvm::Value *val = builder.CreateLoad(ptr_type, slot_ptr);
                            stack.push_back(val);
                        }
                    }
                }
            }

            // Check if this is a resume point
            if (instr_idx_to_yield_idx.count(i))
            {
                // This yield was already processed; now we're at the resume point
                // The resume block for this yield will push sent_value onto stack
            }

            // Handle each opcode (simplified - we'll handle key ones for generators)
            if (instr.opcode == op::RESUME)
            {
                // No-op for generators
            }
            else if (instr.opcode == op::RETURN_GENERATOR)
            {
                // Skip - handled at function level
            }
            else if (instr.opcode == op::POP_TOP)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    builder.CreateCall(py_decref_func, {val});
                }
            }
            else if (instr.opcode == op::LOAD_FAST || instr.opcode == op::LOAD_FAST_CHECK)
            {
                llvm::Value *val = load_local(instr.arg);
                builder.CreateCall(py_incref_func, {val});
                stack.push_back(val);
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                // Python 3.13: Pushes co_varnames[arg>>4] then co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                
                llvm::Value *val1 = load_local(first_local);
                builder.CreateCall(py_incref_func, {val1});
                stack.push_back(val1);
                
                llvm::Value *val2 = load_local(second_local);
                builder.CreateCall(py_incref_func, {val2});
                stack.push_back(val2);
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // Decref old value if present
                    llvm::Value *old_val = load_local(instr.arg);
                    llvm::Value *is_null = builder.CreateICmpEQ(old_val, 
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(
                        *local_context, "decref", func);
                    llvm::BasicBlock *after_decref = llvm::BasicBlock::Create(
                        *local_context, "after_decref", func);
                    builder.CreateCondBr(is_null, after_decref, decref_block);

                    builder.SetInsertPoint(decref_block);
                    builder.CreateCall(py_decref_func, {old_val});
                    builder.CreateBr(after_decref);

                    builder.SetInsertPoint(after_decref);
                    store_local(instr.arg, val);
                }
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                if (instr.arg < obj_constants.size() && obj_constants[instr.arg] != nullptr)
                {
                    llvm::Value *const_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(obj_constants[instr.arg]));
                    llvm::Value *py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                    builder.CreateCall(py_incref_func, {py_obj});
                    stack.push_back(py_obj);
                }
                else
                {
                    llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                    llvm::Value *py_obj = builder.CreateCall(py_long_fromlonglong_func, {const_val});
                    stack.push_back(py_obj);
                }
            }
            else if (instr.opcode == op::BINARY_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();

                    llvm::Value *result = nullptr;
                    llvm::Value *py_none_val = builder.CreateIntToPtr(
                        llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None)), ptr_type);
                    switch (instr.arg)
                    {
                    case 0:  // ADD
                    case 13: // INPLACE_ADD
                        result = builder.CreateCall(py_number_add_func, {lhs, rhs}); break;
                    case 10: // SUB
                    case 23: // INPLACE_SUB
                        result = builder.CreateCall(py_number_subtract_func, {lhs, rhs}); break;
                    case 5:  // MUL
                    case 18: // INPLACE_MUL
                        result = builder.CreateCall(py_number_multiply_func, {lhs, rhs}); break;
                    case 11: // TRUE_DIV
                    case 24: // INPLACE_TRUE_DIV
                        result = builder.CreateCall(py_number_truedivide_func, {lhs, rhs}); break;
                    case 2:  // FLOOR_DIV
                    case 15: // INPLACE_FLOOR_DIV
                        result = builder.CreateCall(py_number_floordivide_func, {lhs, rhs}); break;
                    case 6:  // MOD
                    case 19: // INPLACE_MOD
                        result = builder.CreateCall(py_number_remainder_func, {lhs, rhs}); break;
                    case 8:  // POW
                    case 21: // INPLACE_POW
                        result = builder.CreateCall(py_number_power_func, {lhs, rhs, py_none_val}); break;
                    case 1:  // AND (bitwise)
                    case 14: // INPLACE_AND
                        result = builder.CreateCall(py_number_and_func, {lhs, rhs}); break;
                    case 7:  // OR (bitwise)
                    case 20: // INPLACE_OR
                        result = builder.CreateCall(py_number_or_func, {lhs, rhs}); break;
                    case 12: // XOR (bitwise)
                    case 25: // INPLACE_XOR
                        result = builder.CreateCall(py_number_xor_func, {lhs, rhs}); break;
                    case 3:  // LSHIFT
                    case 16: // INPLACE_LSHIFT
                        result = builder.CreateCall(py_number_lshift_func, {lhs, rhs}); break;
                    case 9:  // RSHIFT
                    case 22: // INPLACE_RSHIFT
                        result = builder.CreateCall(py_number_rshift_func, {lhs, rhs}); break;
                    case 4:  // MATMUL
                    case 17: // INPLACE_MATMUL
                        result = builder.CreateCall(py_number_matrixmultiply_func, {lhs, rhs}); break;
                    default:
                        // Unsupported binary op - set error and return NULL
                        {
                            llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                                llvm::Type::getVoidTy(*local_context),
                                {ptr_type, ptr_type}, false);
                            llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                                "PyErr_SetString", py_err_set_str_type);
                            llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(PyExc_TypeError));
                            llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                            llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported binary operation");
                            builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                            result = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        }
                        break;
                    }

                    builder.CreateCall(py_decref_func, {lhs});
                    builder.CreateCall(py_decref_func, {rhs});
                    
                    // Check for errors from BINARY_OP
                    check_error_and_branch_gen(instr.offset, result, "binop");
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::COMPARE_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();

                    // Python 3.13 encoding: (op_code << 5) | flags
                    // Compare op mapping: op_code = arg >> 5
                    int py_op = instr.arg >> 5;
                    llvm::Value *op_val = llvm::ConstantInt::get(i32_type, py_op);
                    llvm::Value *result = builder.CreateCall(py_object_richcompare_bool_func, {lhs, rhs, op_val});

                    builder.CreateCall(py_decref_func, {lhs});
                    builder.CreateCall(py_decref_func, {rhs});

                    // Convert int result to Python bool
                    llvm::Value *true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *is_true = builder.CreateICmpNE(result, llvm::ConstantInt::get(i32_type, 0));
                    llvm::Value *bool_result = builder.CreateSelect(is_true,
                        builder.CreateIntToPtr(true_ptr, ptr_type),
                        builder.CreateIntToPtr(false_ptr, ptr_type));
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::LOAD_GLOBAL)
            {
                // Python 3.13: LOAD_GLOBAL loads global variable
                // arg >> 1 = index into co_names
                // arg & 1 = if set, push NULL after global (for calling convention)
                int name_idx = instr.arg >> 1;
                bool push_null = (instr.arg & 1) != 0;

                if (name_idx < static_cast<int>(name_objects.size()))
                {
                    // Get the name object for lookup
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "name_obj");

                    // Get globals dict pointer
                    llvm::Value *globals_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr, ptr_type, "globals_dict");

                    // PyDict_GetItem(globals_dict, name) - returns borrowed reference or NULL
                    llvm::Value *global_obj = builder.CreateCall(
                        py_dict_getitem_func,
                        {globals_dict, name_obj},
                        "global_lookup");

                    // Check if found in globals, if not try builtins
                    llvm::Value *is_null = builder.CreateICmpEQ(
                        global_obj,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "not_found_in_globals");

                    llvm::BasicBlock *found_block = llvm::BasicBlock::Create(*local_context, "global_found", func);
                    llvm::BasicBlock *try_builtins_block = llvm::BasicBlock::Create(*local_context, "try_builtins", func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(*local_context, "global_continue", func);

                    builder.CreateCondBr(is_null, try_builtins_block, found_block);

                    // Try builtins
                    builder.SetInsertPoint(try_builtins_block);
                    llvm::Value *builtins_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(builtins_dict_ptr));
                    llvm::Value *builtins_dict = builder.CreateIntToPtr(builtins_ptr, ptr_type, "builtins_dict");
                    llvm::Value *builtin_obj = builder.CreateCall(
                        py_dict_getitem_func,
                        {builtins_dict, name_obj},
                        "builtin_lookup");
                    builder.CreateBr(continue_block);

                    // Found in globals
                    builder.SetInsertPoint(found_block);
                    builder.CreateBr(continue_block);

                    // Continue with PHI node to select result
                    builder.SetInsertPoint(continue_block);
                    current_block = continue_block;
                    llvm::PHINode *result_phi = builder.CreatePHI(ptr_type, 2, "global_result");
                    result_phi->addIncoming(builtin_obj, try_builtins_block);
                    result_phi->addIncoming(global_obj, found_block);

                    // Incref the result (PyDict_GetItem returns borrowed reference)
                    builder.CreateCall(py_incref_func, {result_phi});

                    stack.push_back(result_phi);

                    // Push NULL after global if needed (Python 3.13 calling convention)
                    if (push_null)
                    {
                        llvm::Value *null_ptr_val = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        stack.push_back(null_ptr_val);
                    }
                }
            }
            else if (instr.opcode == op::LOAD_ATTR)
            {
                // Implements obj.attr for generators/coroutines
                // Python 3.13: arg >> 1 = index into co_names, arg & 1 = method load flag
                int name_idx = instr.arg >> 1;
                bool is_method = (instr.arg & 1) != 0;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();

                    // Get attribute name from names
                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    // PyObject_GetAttr returns new reference
                    llvm::Value *result = builder.CreateCall(py_object_getattr_func, {obj, attr_name});

                    // Decref the object we consumed
                    builder.CreateCall(py_decref_func, {obj});

                    // Check for errors from LOAD_ATTR (AttributeError)
                    check_error_and_branch_gen(instr.offset, result, "loadattr");

                    if (is_method)
                    {
                        // Method loading: push callable (bound method), then NULL
                        llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
                            llvm::PointerType::get(*local_context, 0));
                        stack.push_back(result);   // callable = bound method
                        stack.push_back(null_ptr); // self_or_null = NULL
                    }
                    else
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::PUSH_NULL)
            {
                // Push NULL onto stack (for calling convention)
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(
                    llvm::PointerType::get(*local_context, 0));
                stack.push_back(null_ptr);
            }
            else if (instr.opcode == op::BUILD_LIST)
            {
                // Build a list from the top N items on stack
                int count = instr.arg;
                llvm::Value *list = builder.CreateCall(
                    py_list_new_func,
                    {llvm::ConstantInt::get(i64_type, count)});
                
                // Pop items and add to list in order
                for (int j = count - 1; j >= 0; j--)
                {
                    if (!stack.empty())
                    {
                        llvm::Value *item = stack.back();
                        stack.pop_back();
                        // PyList_SetItem steals reference
                        builder.CreateCall(
                            py_list_setitem_func,
                            {list, llvm::ConstantInt::get(i64_type, j), item});
                    }
                }
                stack.push_back(list);
            }
            else if (instr.opcode == op::BUILD_TUPLE)
            {
                // Build a tuple from the top N items on stack
                int count = instr.arg;
                llvm::Value *tuple = builder.CreateCall(
                    py_tuple_new_func,
                    {llvm::ConstantInt::get(i64_type, count)});
                
                for (int j = count - 1; j >= 0; j--)
                {
                    if (!stack.empty())
                    {
                        llvm::Value *item = stack.back();
                        stack.pop_back();
                        // PyTuple_SetItem steals reference
                        builder.CreateCall(
                            py_tuple_setitem_func,
                            {tuple, llvm::ConstantInt::get(i64_type, j), item});
                    }
                }
                stack.push_back(tuple);
            }
            else if (instr.opcode == op::BUILD_CONST_KEY_MAP)
            {
                // Build a dict from N values + 1 tuple of keys on stack
                // Stack: [val1, val2, ..., valN, keys_tuple]
                int count = instr.arg;
                
                if (stack.size() >= static_cast<size_t>(count + 1))
                {
                    llvm::Value *keys_tuple = stack.back();
                    stack.pop_back();
                    
                    // Create new dict
                    llvm::Value *dict = builder.CreateCall(py_dict_new_func, {});
                    
                    // Pop values in reverse order and set
                    std::vector<llvm::Value *> values;
                    for (int j = 0; j < count; j++)
                    {
                        values.push_back(stack.back());
                        stack.pop_back();
                    }
                    
                    // Add items to dict
                    for (int j = 0; j < count; j++)
                    {
                        // Get key from tuple
                        llvm::Value *key = builder.CreateCall(
                            py_tuple_getitem_func,
                            {keys_tuple, llvm::ConstantInt::get(i64_type, j)});
                        // values are in reverse order
                        llvm::Value *value = values[count - 1 - j];
                        // PyDict_SetItem does NOT steal references
                        builder.CreateCall(py_dict_setitem_func, {dict, key, value});
                        builder.CreateCall(py_decref_func, {value});
                    }
                    
                    builder.CreateCall(py_decref_func, {keys_tuple});
                    stack.push_back(dict);
                }
            }
            else if (instr.opcode == op::STORE_SUBSCR)
            {
                // Implements container[key] = value for generators/coroutines
                // Stack order: TOS=key, TOS1=container, TOS2=value
                if (stack.size() >= 3)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    // PyObject_SetItem(container, key, value)
                    builder.CreateCall(py_object_setitem_func, {container, key, value});

                    // Decref consumed references
                    builder.CreateCall(py_decref_func, {key});
                    builder.CreateCall(py_decref_func, {value});
                    // container is borrowed (it stays alive)
                    builder.CreateCall(py_decref_func, {container});
                }
            }
            else if (instr.opcode == op::BINARY_SUBSCR)
            {
                // Implements container[key] for generators/coroutines
                if (stack.size() >= 2)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();

                    // PyObject_GetItem returns new reference
                    llvm::Value *result = builder.CreateCall(py_object_getitem_func, {container, key});

                    // Decref consumed references
                    builder.CreateCall(py_decref_func, {key});
                    builder.CreateCall(py_decref_func, {container});

                    // Check for errors from BINARY_SUBSCR (KeyError, IndexError, etc.)
                    check_error_and_branch_gen(instr.offset, result, "subscr");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CALL)
            {
                // Python 3.13: CALL opcode, arg = number of arguments (excluding self/NULL)
                // Stack layout:
                //   callable = stack[-2-oparg]
                //   self_or_null = stack[-1-oparg]
                //   args = &stack[-oparg] (oparg elements)
                int num_args = instr.arg;

                if (stack.size() >= static_cast<size_t>(num_args + 2))
                {
                    size_t base = stack.size() - num_args - 2;

                    llvm::Value *callable = stack[base];
                    llvm::Value *self_or_null = stack[base + 1];

                    // Collect arguments in order
                    std::vector<llvm::Value *> args;
                    for (int ai = 0; ai < num_args; ++ai)
                    {
                        args.push_back(stack[base + 2 + ai]);
                    }

                    // Remove all CALL operands from stack
                    stack.erase(stack.begin() + base, stack.end());

                    // Create args tuple - PyTuple_SetItem steals references
                    llvm::Value *args_count = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value *args_tuple = builder.CreateCall(py_tuple_new_func, {args_count});

                    // Fill tuple with args
                    for (int ai = 0; ai < num_args; ++ai)
                    {
                        llvm::Value *index_val = llvm::ConstantInt::get(i64_type, ai);
                        llvm::Value *arg = args[ai];
                        // PyTuple_SetItem steals reference
                        builder.CreateCall(py_tuple_setitem_func, {args_tuple, index_val, arg});
                    }

                    // Call PyObject_Call(callable, args_tuple, NULL)
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *result = builder.CreateCall(py_object_call_func, {callable, args_tuple, null_kwargs});

                    // Decrement args_tuple refcount
                    builder.CreateCall(py_decref_func, {args_tuple});

                    // Decref callable
                    builder.CreateCall(py_decref_func, {callable});

                    // Handle self_or_null decref
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);
                    current_block = after_decref_self;

                    // Check for errors from CALL
                    check_error_and_branch_gen(instr.offset, result, "call");

                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::YIELD_VALUE)
            {
                // This is the core of generator support!
                // We must spill the remaining stack to persistent storage before yielding
                if (!stack.empty())
                {
                    llvm::Value *yield_val = stack.back();
                    stack.pop_back();
                    
                    // SPILL STACK: Save remaining stack values to locals[stack_base + j]
                    // This persists them across the yield/resume boundary
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        // Incref: we're creating a new reference in the locals array
                        builder.CreateCall(py_incref_func, {val});
                        // Store to locals[stack_base + j]
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }

                    // Set state to the resume point
                    int resume_state = yield_to_state[i];
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, resume_state), state_ptr);

                    // Return the yielded value
                    builder.CreateRet(yield_val);

                    // Now set up the resume block for when generator.send() is called
                    if (current_yield_idx < resume_blocks.size())
                    {
                        builder.SetInsertPoint(resume_blocks[current_yield_idx]);
                        current_block = resume_blocks[current_yield_idx];
                        
                        // RESTORE STACK: Load persisted stack values back
                        // Clear the compile-time stack first
                        stack.clear();
                        size_t saved_depth = yield_stack_depth[i];
                        
                        for (size_t j = 0; j < saved_depth; ++j)
                        {
                            // Load from locals[stack_base + j]
                            llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                            llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                            llvm::Value *restored_val = builder.CreateLoad(ptr_type, slot_ptr);
                            
                            // Incref for the stack reference
                            builder.CreateCall(py_incref_func, {restored_val});
                            stack.push_back(restored_val);
                            
                            // Decref the stored reference and clear the slot
                            builder.CreateCall(py_decref_func, {restored_val});
                            builder.CreateStore(
                                llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                                slot_ptr);
                        }
                        
                        // Push the sent value onto the stack
                        builder.CreateCall(py_incref_func, {sent_value});
                        stack.push_back(sent_value);
                        current_yield_idx++;
                    }
                }
            }
            else if (instr.opcode == op::RETURN_VALUE)
            {
                // Generator return - set state to done and return the value
                builder.CreateStore(llvm::ConstantInt::get(i32_type, -1), state_ptr);
                if (!stack.empty())
                {
                    llvm::Value *ret_val = stack.back();
                    stack.pop_back();
                    builder.CreateRet(ret_val);
                }
                else
                {
                    llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                    builder.CreateCall(py_incref_func, {py_none});
                    builder.CreateRet(py_none);
                }
            }
            else if (instr.opcode == op::RETURN_CONST)
            {
                builder.CreateStore(llvm::ConstantInt::get(i32_type, -1), state_ptr);
                if (instr.arg < obj_constants.size() && obj_constants[instr.arg] != nullptr)
                {
                    llvm::Value *const_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(obj_constants[instr.arg]));
                    llvm::Value *py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                    builder.CreateCall(py_incref_func, {py_obj});
                    builder.CreateRet(py_obj);
                }
                else
                {
                    llvm::Value *const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                    llvm::Value *py_obj = builder.CreateCall(py_long_fromlonglong_func, {const_val});
                    builder.CreateRet(py_obj);
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                int target = instr.argval;
                if (!offset_blocks.count(target))
                {
                    offset_blocks[target] = llvm::BasicBlock::Create(
                        *local_context, "loop_" + std::to_string(target), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    // CRITICAL: Spill stack before jumping back to loop head
                    // This ensures the stack state is persisted for the next iteration
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    // Record expected depth for the loop target
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    builder.CreateBr(offset_blocks[target]);
                }
                // Create unreachable continuation block (code after unconditional jump)
                llvm::BasicBlock *after = llvm::BasicBlock::Create(
                    *local_context, "after_jump_" + std::to_string(i), func);
                builder.SetInsertPoint(after);
                current_block = after;
                // Stack is now undefined - will be reloaded at next jump target
            }
            else if (instr.opcode == op::JUMP_BACKWARD_NO_INTERRUPT)
            {
                // JUMP_BACKWARD_NO_INTERRUPT: Same as JUMP_BACKWARD but doesn't check for interrupts
                // Used in exception handlers
                int target = instr.argval;
                if (!offset_blocks.count(target))
                {
                    offset_blocks[target] = llvm::BasicBlock::Create(
                        *local_context, "loop_noint_" + std::to_string(target), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    // Spill stack before jumping
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    builder.CreateBr(offset_blocks[target]);
                }
                llvm::BasicBlock *after = llvm::BasicBlock::Create(
                    *local_context, "after_jump_noint_" + std::to_string(i), func);
                builder.SetInsertPoint(after);
                current_block = after;
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
                int target = instr.argval;
                if (!offset_blocks.count(target))
                {
                    offset_blocks[target] = llvm::BasicBlock::Create(
                        *local_context, "forward_" + std::to_string(target), func);
                }
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    // Spill stack before forward jump
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    builder.CreateBr(offset_blocks[target]);
                }
                llvm::BasicBlock *after = llvm::BasicBlock::Create(
                    *local_context, "after_fwd_" + std::to_string(i), func);
                builder.SetInsertPoint(after);
                current_block = after;
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back();
                    stack.pop_back();

                    llvm::Value *is_true = builder.CreateCall(py_object_istrue_func, {cond});
                    builder.CreateCall(py_decref_func, {cond});

                    llvm::Value *cmp = builder.CreateICmpEQ(is_true, llvm::ConstantInt::get(i32_type, 0));

                    int target = instr.argval;
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "if_false_" + std::to_string(target), func);
                    }
                    
                    // Spill stack before conditional branch (needed for both paths)
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    
                    llvm::BasicBlock *fallthrough = llvm::BasicBlock::Create(
                        *local_context, "fallthrough_" + std::to_string(i), func);

                    builder.CreateCondBr(cmp, offset_blocks[target], fallthrough);
                    builder.SetInsertPoint(fallthrough);
                    current_block = fallthrough;
                    
                    // Reload stack in fallthrough (since we spilled)
                    stack.clear();
                    size_t depth = target_stack_depth[target];
                    for (size_t j = 0; j < depth; ++j)
                    {
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        llvm::Value *val = builder.CreateLoad(ptr_type, slot_ptr);
                        stack.push_back(val);
                    }
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back();
                    stack.pop_back();

                    llvm::Value *is_true = builder.CreateCall(py_object_istrue_func, {cond});
                    builder.CreateCall(py_decref_func, {cond});

                    llvm::Value *cmp = builder.CreateICmpNE(is_true, llvm::ConstantInt::get(i32_type, 0));

                    int target = instr.argval;
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "if_true_" + std::to_string(target), func);
                    }
                    
                    // Spill stack before conditional branch
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    
                    llvm::BasicBlock *fallthrough = llvm::BasicBlock::Create(
                        *local_context, "fallthrough_" + std::to_string(i), func);

                    builder.CreateCondBr(cmp, offset_blocks[target], fallthrough);
                    builder.SetInsertPoint(fallthrough);
                    current_block = fallthrough;
                    
                    // Reload stack in fallthrough
                    stack.clear();
                    size_t depth = target_stack_depth[target];
                    for (size_t j = 0; j < depth; ++j)
                    {
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        llvm::Value *val = builder.CreateLoad(ptr_type, slot_ptr);
                        stack.push_back(val);
                    }
                }
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                // Get iterator from TOS
                if (!stack.empty())
                {
                    llvm::Value *iter = stack.back();  // Don't pop - FOR_ITER keeps iterator on stack

                    llvm::Value *next_val = builder.CreateCall(py_iter_next_func, {iter});

                    // Check if NULL (iterator exhausted)
                    llvm::Value *is_null = builder.CreateICmpEQ(next_val,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                    int target = instr.argval;
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "for_end_" + std::to_string(target), func);
                    }
                    
                    // Spill stack before the branch (iterator is still on stack)
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        builder.CreateCall(py_incref_func, {val});
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        builder.CreateStore(val, slot_ptr);
                    }
                    // The exit target will have the iterator popped (END_FOR does that)
                    // But we still record current depth for the exit path
                    if (!target_stack_depth.count(target)) {
                        target_stack_depth[target] = stack.size();
                    }
                    
                    llvm::BasicBlock *continue_loop = llvm::BasicBlock::Create(
                        *local_context, "for_body_" + std::to_string(i), func);

                    builder.CreateCondBr(is_null, offset_blocks[target], continue_loop);

                    builder.SetInsertPoint(continue_loop);
                    // Clear any exception from PyIter_Next
                    builder.CreateCall(py_err_clear_func, {});
                    
                    // Reload stack and push next_val
                    stack.clear();
                    size_t depth = target_stack_depth[target];
                    for (size_t j = 0; j < depth; ++j)
                    {
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + j);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        llvm::Value *val = builder.CreateLoad(ptr_type, slot_ptr);
                        stack.push_back(val);
                    }
                    stack.push_back(next_val);
                    current_block = continue_loop;
                }
            }
            else if (instr.opcode == op::END_FOR)
            {
                // END_FOR: pop both the loop value and the iterator
                // At this point we came from FOR_ITER's exit path
                // The stack should have: [..., iterator, value] but value is NULL
                // Actually END_FOR in Python 3.12+ just pops the iterator (value was already NULL)
                if (!stack.empty())
                {
                    // Pop the NULL value that FOR_ITER would have pushed (but didn't since iter exhausted)
                    // Actually in our impl, we need to pop iterator
                    llvm::Value *iter = stack.back(); stack.pop_back();
                    builder.CreateCall(py_decref_func, {iter});
                }
            }
            else if (instr.opcode == op::GET_ITER)
            {
                if (!stack.empty())
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();
                    llvm::Value *iter = builder.CreateCall(py_object_getiter_func, {obj});
                    builder.CreateCall(py_decref_func, {obj});
                    stack.push_back(iter);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_NONE)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // Check if val is Py_None
                    llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                    llvm::Value *is_none = builder.CreateICmpEQ(val, py_none);

                    builder.CreateCall(py_decref_func, {val});

                    int target = instr.argval;
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "if_none_" + std::to_string(target), func);
                    }
                    llvm::BasicBlock *fallthrough = llvm::BasicBlock::Create(
                        *local_context, "not_none_" + std::to_string(i), func);

                    builder.CreateCondBr(is_none, offset_blocks[target], fallthrough);
                    builder.SetInsertPoint(fallthrough);
                    current_block = fallthrough;
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_NOT_NONE)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    // Check if val is NOT Py_None
                    llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                    llvm::Value *is_not_none = builder.CreateICmpNE(val, py_none);

                    builder.CreateCall(py_decref_func, {val});

                    int target = instr.argval;
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "if_not_none_" + std::to_string(target), func);
                    }
                    llvm::BasicBlock *fallthrough = llvm::BasicBlock::Create(
                        *local_context, "is_none_" + std::to_string(i), func);

                    builder.CreateCondBr(is_not_none, offset_blocks[target], fallthrough);
                    builder.SetInsertPoint(fallthrough);
                    current_block = fallthrough;
                }
            }
            else if (instr.opcode == op::COPY)
            {
                // COPY i: Push a copy of the i-th item from the stack (1-indexed from top)
                int idx = instr.arg;
                if (idx > 0 && static_cast<size_t>(idx) <= stack.size())
                {
                    llvm::Value *val = stack[stack.size() - idx];
                    builder.CreateCall(py_incref_func, {val});
                    stack.push_back(val);
                }
            }
            else if (instr.opcode == op::SWAP)
            {
                // SWAP i: Swap TOS with the item at position i (1-indexed from top)
                int idx = instr.arg;
                if (idx > 1 && static_cast<size_t>(idx) <= stack.size())
                {
                    size_t tos_idx = stack.size() - 1;
                    size_t other_idx = stack.size() - idx;
                    std::swap(stack[tos_idx], stack[other_idx]);
                }
            }
            // =========================================================================
            // Async/Await Opcodes
            // =========================================================================
            // These opcodes implement the async/await protocol for coroutines.
            // GET_AWAITABLE: Validates/gets awaitable object
            // SEND: Core await mechanism (delegate to inner awaitable)
            // END_SEND: Cleanup after SEND completes
            // =========================================================================
            else if (instr.opcode == op::GET_AWAITABLE)
            {
                // GET_AWAITABLE: Get an awaitable from TOS
                // If TOS is a coroutine, return it directly
                // Otherwise, call __await__ on it
                // arg: where (0=await, 1=async with __aenter__, 2=async with __aexit__)
                if (!stack.empty())
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();
                    
                    // Call PyObject_GetAwaitable to get the awaitable
                    // For simplicity, we'll call a helper that handles the protocol
                    // PyObject* PyObject_GetAwaitable(PyObject* obj) - not a real API,
                    // we need to implement the logic or call our helper
                    
                    // For coroutines/generators, they are their own awaitable
                    // For other objects, we need to call __await__
                    // We'll create a helper function call
                    
                    // Check if it's already a coroutine (PyCoro_CheckExact or PyGen_Check)
                    // For now, use a runtime helper
                    
                    // Declare helper: PyObject* _get_awaitable(PyObject* obj)
                    llvm::FunctionType *helper_type = llvm::FunctionType::get(
                        ptr_type, {ptr_type}, false);
                    llvm::Function *get_awaitable_helper = llvm::cast<llvm::Function>(
                        module->getOrInsertFunction("_PyJIT_GetAwaitable", helper_type).getCallee());
                    
                    llvm::Value *awaitable = builder.CreateCall(get_awaitable_helper, {obj});
                    
                    // Decref the original object if different
                    builder.CreateCall(py_decref_func, {obj});
                    
                    stack.push_back(awaitable);
                }
            }
            else if (instr.opcode == op::SEND)
            {
                // SEND: Send a value to a subgenerator/awaitable
                // STACK: [..., receiver, value] -> [..., receiver, result]
                // Essentially: result = receiver.send(value)
                // If receiver raises StopIteration, jump to target with return value
                // arg: delta (relative offset to jump on completion)
                if (stack.size() >= 2)
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    llvm::Value *receiver = stack.back();
                    // Don't pop receiver - it stays for the next iteration
                    
                    // Use PyIter_Send which is the C API for this
                    // PySendResult PyIter_Send(PyObject *iter, PyObject *arg, PyObject **result)
                    // Returns PYGEN_RETURN=0, PYGEN_NEXT=1, PYGEN_ERROR=2
                    
                    // Declare PyIter_Send
                    llvm::FunctionType *send_type = llvm::FunctionType::get(
                        i32_type, {ptr_type, ptr_type, llvm::PointerType::get(*local_context, 0)}, false);
                    llvm::Function *py_iter_send_func = llvm::cast<llvm::Function>(
                        module->getOrInsertFunction("PyIter_Send", send_type).getCallee());
                    
                    // Allocate space for result on stack (in entry block for proper LLVM semantics)
                    llvm::Value *result_ptr = builder.CreateAlloca(ptr_type, nullptr, "send_result");
                    builder.CreateStore(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)), result_ptr);
                    
                    // Call PyIter_Send
                    llvm::Value *send_result = builder.CreateCall(py_iter_send_func, {receiver, value, result_ptr});
                    
                    // Decref the value we sent
                    builder.CreateCall(py_decref_func, {value});
                    
                    // Load the result
                    llvm::Value *result = builder.CreateLoad(ptr_type, result_ptr);
                    
                    // Create blocks for handling each case
                    llvm::BasicBlock *next_block = llvm::BasicBlock::Create(
                        *local_context, "send_next_" + std::to_string(i), func);
                    llvm::BasicBlock *return_block = llvm::BasicBlock::Create(
                        *local_context, "send_return_" + std::to_string(i), func);
                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                        *local_context, "send_error_" + std::to_string(i), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "send_cont_" + std::to_string(i), func);
                    
                    // Create switch on send_result
                    llvm::SwitchInst *sw = builder.CreateSwitch(send_result, error_block, 2);
                    sw->addCase(llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(i32_type, 0)), return_block);  // PYGEN_RETURN
                    sw->addCase(llvm::cast<llvm::ConstantInt>(llvm::ConstantInt::get(i32_type, 1)), next_block);    // PYGEN_NEXT
                    
                    // Handle PYGEN_NEXT - yielded value, push result and continue
                    builder.SetInsertPoint(next_block);
                    // Receiver stays on stack, push result on top
                    // The next instruction should handle this (usually YIELD_VALUE)
                    builder.CreateBr(continue_block);
                    
                    // Handle PYGEN_RETURN - sub-iterator finished, jump to target
                    builder.SetInsertPoint(return_block);
                    // The receiver will be removed by END_SEND
                    // Jump to the target offset
                    int target = instr.argval;  // Jump target
                    if (!offset_blocks.count(target))
                    {
                        offset_blocks[target] = llvm::BasicBlock::Create(
                            *local_context, "send_done_" + std::to_string(target), func);
                    }
                    builder.CreateBr(offset_blocks[target]);
                    
                    // Handle error
                    builder.SetInsertPoint(error_block);
                    // Clean up receiver before returning error
                    builder.CreateCall(py_decref_func, {receiver});
                    stack.pop_back();  // Remove receiver from compile-time stack
                    // Return NULL to propagate error
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    
                    // Continue from continue_block - push result for next instruction
                    builder.SetInsertPoint(continue_block);
                    current_block = continue_block;
                    
                    // Push result (receiver is still on stack from before)
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::END_SEND)
            {
                // END_SEND: Clean up after SEND completes
                // STACK: [..., receiver, result] -> [..., result]
                if (stack.size() >= 2)
                {
                    llvm::Value *result = stack.back();
                    stack.pop_back();
                    llvm::Value *receiver = stack.back();
                    stack.pop_back();
                    
                    // Decref receiver
                    builder.CreateCall(py_decref_func, {receiver});
                    
                    // Push result back
                    stack.push_back(result);
                }
            }
            // ========== Exception Handling Opcodes for Generators ==========
            else if (instr.opcode == op::PUSH_EXC_INFO)
            {
                // PUSH_EXC_INFO: At start of exception handler
                // Fetch the current exception and push it onto the stack
                
                llvm::Value *type_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_type_ptr");
                llvm::Value *value_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_value_ptr");
                llvm::Value *tb_ptr = builder.CreateAlloca(ptr_type, nullptr, "exc_tb_ptr");

                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                builder.CreateStore(null_ptr, type_ptr);
                builder.CreateStore(null_ptr, value_ptr);
                builder.CreateStore(null_ptr, tb_ptr);

                // Fetch the current exception
                builder.CreateCall(py_err_fetch_func, {type_ptr, value_ptr, tb_ptr});

                llvm::Value *exc_value = builder.CreateLoad(ptr_type, value_ptr, "exc_value");
                llvm::Value *exc_type = builder.CreateLoad(ptr_type, type_ptr, "exc_type");
                llvm::Value *exc_tb = builder.CreateLoad(ptr_type, tb_ptr, "exc_tb");

                // Restore exception for CHECK_EXC_MATCH
                builder.CreateCall(py_xincref_func, {exc_type});
                builder.CreateCall(py_xincref_func, {exc_value});
                builder.CreateCall(py_xincref_func, {exc_tb});
                builder.CreateCall(py_err_restore_func, {exc_type, exc_value, exc_tb});

                // Push exc_value or exc_type if value is NULL
                llvm::Value *to_push = builder.CreateSelect(
                    builder.CreateICmpNE(exc_value, null_ptr),
                    exc_value,
                    exc_type);
                builder.CreateCall(py_xincref_func, {to_push});
                stack.push_back(to_push);
            }
            else if (instr.opcode == op::POP_EXCEPT)
            {
                // POP_EXCEPT: End of exception handler - clear the exception state
                builder.CreateCall(py_err_clear_func, {});
            }
            else if (instr.opcode == op::CHECK_EXC_MATCH)
            {
                // CHECK_EXC_MATCH: Test if exception matches type
                // Stack: [..., exc_value, exc_type] -> [..., exc_value, bool]
                if (stack.size() >= 2)
                {
                    llvm::Value *exc_type = stack.back();
                    stack.pop_back();
                    llvm::Value *exc_value = stack.back();  // Stays on stack

                    llvm::Value *actual_type = builder.CreateCall(py_object_type_func, {exc_value}, "actual_exc_type");
                    llvm::Value *match_result = builder.CreateCall(py_exception_matches_func,
                                                                   {actual_type, exc_type}, "exc_match");

                    builder.CreateCall(py_decref_func, {actual_type});
                    builder.CreateCall(py_decref_func, {exc_type});

                    llvm::Value *is_match = builder.CreateICmpNE(match_result,
                                                                 llvm::ConstantInt::get(i32_type, 0), "is_match");
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                    llvm::Value *result = builder.CreateSelect(is_match, py_true, py_false, "match_bool");
                    builder.CreateCall(py_incref_func, {result});
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RERAISE)
            {
                // RERAISE: Re-raise the current exception
                // arg=0: re-raise, arg=1: re-raise with __traceback__ update
                // Simply return NULL to propagate the exception
                builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
            }
            else if (instr.opcode == op::CALL_INTRINSIC_1)
            {
                // CALL_INTRINSIC_1: Call a 1-arg intrinsic function
                // Intrinsic function codes (same as compile_function):
                // 1: PRINT, 3: STOPITERATION_ERROR, 4: ASYNC_GEN_WRAP
                // 5: UNARY_POSITIVE, 6: LIST_TO_TUPLE
                if (!stack.empty())
                {
                    llvm::Value *arg = stack.back();
                    stack.pop_back();
                    
                    llvm::Value *result = nullptr;
                    
                    switch (instr.arg)
                    {
                    case 1: // INTRINSIC_PRINT
                        // Debug print - just consume and return None
                        builder.CreateCall(py_decref_func, {arg});
                        {
                            llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(Py_None));
                            result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                            builder.CreateCall(py_incref_func, {result});
                        }
                        break;
                    case 3: // INTRINSIC_STOPITERATION_ERROR
                        // Handle StopIteration - just decref and push None
                        builder.CreateCall(py_decref_func, {arg});
                        {
                            llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(Py_None));
                            result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                            builder.CreateCall(py_incref_func, {result});
                        }
                        break;
                    case 4: // INTRINSIC_ASYNC_GEN_WRAP
                        // For async generators - just pass through for now
                        result = arg;  // Transfer ownership
                        break;
                    case 5: // INTRINSIC_UNARY_POSITIVE
                        result = builder.CreateCall(py_number_positive_func, {arg});
                        builder.CreateCall(py_decref_func, {arg});
                        check_error_and_branch_gen(instr.offset, result, "unary_positive");
                        break;
                    case 6: // INTRINSIC_LIST_TO_TUPLE
                    {
                        llvm::FunctionType *list_as_tuple_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee list_as_tuple_func = module->getOrInsertFunction(
                            "PyList_AsTuple", list_as_tuple_type);
                        result = builder.CreateCall(list_as_tuple_func, {arg});
                        builder.CreateCall(py_decref_func, {arg});
                        check_error_and_branch_gen(instr.offset, result, "list_to_tuple");
                        break;
                    }
                    case 10: // INTRINSIC_SUBSCRIPT_GENERIC
                    {
                        // Generic[T] type subscripting - arg is tuple (origin, args)
                        llvm::FunctionType *get_item_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee get_item_func = module->getOrInsertFunction(
                            "PyObject_GetItem", get_item_type);
                        
                        llvm::FunctionType *tuple_get_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, i64_type}, false);
                        llvm::FunctionCallee tuple_get_func = module->getOrInsertFunction(
                            "PyTuple_GetItem", tuple_get_type);
                        
                        llvm::Value *origin = builder.CreateCall(tuple_get_func, 
                            {arg, llvm::ConstantInt::get(i64_type, 0)});
                        llvm::Value *args = builder.CreateCall(tuple_get_func,
                            {arg, llvm::ConstantInt::get(i64_type, 1)});
                        
                        result = builder.CreateCall(get_item_func, {origin, args});
                        builder.CreateCall(py_decref_func, {arg});
                        check_error_and_branch_gen(instr.offset, result, "subscript_generic");
                        break;
                    }
                    case 7: // INTRINSIC_TYPEVAR
                    {
                        // TypeVar creation via typing.TypeVar
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        llvm::Value *typevar_name = builder.CreateGlobalStringPtr("TypeVar");
                        llvm::Value *typevar_class = builder.CreateCall(getattr_func, {typing_mod, typevar_name});
                        
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typevar_class, arg, kwargs});
                        
                        builder.CreateCall(py_decref_func, {typevar_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {arg});
                        
                        check_error_and_branch_gen(instr.offset, result, "typevar");
                        break;
                    }
                    case 8: // INTRINSIC_PARAMSPEC
                    {
                        // ParamSpec creation via typing.ParamSpec
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        llvm::Value *paramspec_name = builder.CreateGlobalStringPtr("ParamSpec");
                        llvm::Value *paramspec_class = builder.CreateCall(getattr_func, {typing_mod, paramspec_name});
                        
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {paramspec_class, arg, kwargs});
                        
                        builder.CreateCall(py_decref_func, {paramspec_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {arg});
                        
                        check_error_and_branch_gen(instr.offset, result, "paramspec");
                        break;
                    }
                    case 9: // INTRINSIC_TYPEVARTUPLE
                    {
                        // TypeVarTuple creation via typing.TypeVarTuple
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        llvm::Value *typevartuple_name = builder.CreateGlobalStringPtr("TypeVarTuple");
                        llvm::Value *typevartuple_class = builder.CreateCall(getattr_func, {typing_mod, typevartuple_name});
                        
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typevartuple_class, arg, kwargs});
                        
                        builder.CreateCall(py_decref_func, {typevartuple_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {arg});
                        
                        check_error_and_branch_gen(instr.offset, result, "typevartuple");
                        break;
                    }
                    case 11: // INTRINSIC_TYPEALIAS
                    {
                        // TypeAlias creation via typing.TypeAliasType
                        llvm::FunctionType *import_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee import_func = module->getOrInsertFunction(
                            "PyImport_ImportModule", import_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::FunctionType *call_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type, ptr_type}, false);
                        llvm::FunctionCallee call_func = module->getOrInsertFunction(
                            "PyObject_Call", call_type);
                        
                        llvm::Value *typing_name = builder.CreateGlobalStringPtr("typing");
                        llvm::Value *typing_mod = builder.CreateCall(import_func, {typing_name});
                        
                        llvm::Value *typealias_name = builder.CreateGlobalStringPtr("TypeAliasType");
                        llvm::Value *typealias_class = builder.CreateCall(getattr_func, {typing_mod, typealias_name});
                        
                        llvm::Value *py_none_ptr_kw = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value *kwargs = builder.CreateIntToPtr(py_none_ptr_kw, ptr_type);
                        result = builder.CreateCall(call_func, {typealias_class, arg, kwargs});
                        
                        builder.CreateCall(py_decref_func, {typealias_class});
                        builder.CreateCall(py_decref_func, {typing_mod});
                        builder.CreateCall(py_decref_func, {arg});
                        
                        check_error_and_branch_gen(instr.offset, result, "typealias");
                        break;
                    }
                    case 2: // INTRINSIC_IMPORT_STAR
                    {
                        // from module import * - merge module dict into locals
                        llvm::FunctionType *get_frame_type = llvm::FunctionType::get(
                            ptr_type, {}, false);
                        llvm::FunctionCallee get_frame_func = module->getOrInsertFunction(
                            "PyEval_GetFrame", get_frame_type);
                        
                        llvm::FunctionType *frame_get_locals_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee frame_get_locals_func = module->getOrInsertFunction(
                            "PyFrame_GetLocals", frame_get_locals_type);
                        
                        llvm::FunctionType *getattr_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee getattr_func = module->getOrInsertFunction(
                            "PyObject_GetAttrString", getattr_type);
                        
                        llvm::Value *frame = builder.CreateCall(get_frame_func, {});
                        llvm::Value *locals = builder.CreateCall(frame_get_locals_func, {frame});
                        
                        llvm::FunctionType *dict_merge_type = llvm::FunctionType::get(
                            builder.getInt32Ty(), {ptr_type, ptr_type, builder.getInt32Ty()}, false);
                        llvm::FunctionCallee dict_merge_func = module->getOrInsertFunction(
                            "PyDict_Merge", dict_merge_type);
                        
                        llvm::Value *dict_name = builder.CreateGlobalStringPtr("__dict__");
                        llvm::Value *mod_dict = builder.CreateCall(getattr_func, {arg, dict_name});
                        
                        builder.CreateCall(dict_merge_func, 
                            {locals, mod_dict, llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
                        
                        builder.CreateCall(py_decref_func, {mod_dict});
                        builder.CreateCall(py_decref_func, {locals});
                        builder.CreateCall(py_decref_func, {arg});
                        
                        llvm::FunctionType *err_clear_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context), {}, false);
                        llvm::FunctionCallee err_clear_func = module->getOrInsertFunction(
                            "PyErr_Clear", err_clear_type);
                        builder.CreateCall(err_clear_func, {});
                        
                        llvm::Value *py_none_ptr_ret = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(py_none_ptr_ret, ptr_type);
                        builder.CreateCall(py_incref_func, {result});
                        break;
                    }
                    default:
                    {
                        // Unknown intrinsic - raise error and transition to error state
                        builder.CreateCall(py_decref_func, {arg});
                        llvm::FunctionType *py_err_set_str_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context),
                            {ptr_type, ptr_type}, false);
                        llvm::FunctionCallee py_err_set_str_func = module->getOrInsertFunction(
                            "PyErr_SetString", py_err_set_str_type);
                        llvm::Value *exc_type_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(PyExc_SystemError));
                        llvm::Value *exc_type = builder.CreateIntToPtr(exc_type_ptr, ptr_type);
                        llvm::Value *msg = builder.CreateGlobalStringPtr("unsupported intrinsic function in generator");
                        builder.CreateCall(py_err_set_str_func, {exc_type, msg});
                        builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                        break;
                    }
                    }
                    
                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            // Unsupported opcodes - should have been filtered by _is_simple_generator
        }

        // Ensure function has a terminator
        if (!builder.GetInsertBlock()->getTerminator())
        {
            builder.CreateBr(gen_done);
        }

        // Verify and optimize
        std::string verify_err;
        llvm::raw_string_ostream verify_stream(verify_err);
        if (llvm::verifyFunction(*func, &verify_stream))
        {
            llvm::errs() << "Generator step function verification failed: " << verify_err << "\n";
            // Don't return false - try to proceed anyway for debugging
        }

        optimize_module(*module, func);

        // Add to JIT
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err)
        {
            llvm::errs() << "Failed to add generator module: " << toString(std::move(err)) << "\n";
            return false;
        }

        compiled_functions.insert(step_name);
        return true;
    }

    // Get a callable that creates generator objects
    nb::object JITCore::get_generator_callable(const std::string &name, int param_count, int total_locals,
                                               nb::object func_name, nb::object func_qualname)
    {
        std::string step_name = name + "_step";
        uint64_t step_addr = lookup_symbol(step_name);
        if (step_addr == 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Generator step function not found");
            return nb::none();
        }

        GeneratorStepFunc step_func = reinterpret_cast<GeneratorStepFunc>(step_addr);
        PyObject *py_name = func_name.ptr();
        PyObject *py_qualname = func_qualname.ptr();
        Py_ssize_t num_locals = static_cast<Py_ssize_t>(total_locals);

        // Create a Python function that returns a new generator each time it's called
        // We'll use a lambda captured in a PyCFunction
        // For now, return a tuple that Python can use to create generators
        nb::dict result;
        result["step_func_addr"] = step_addr;
        result["num_locals"] = num_locals;
        result["name"] = func_name;
        result["qualname"] = func_qualname;
        result["param_count"] = param_count;

        return result;
    }

    // =========================================================================
    // JIT Generator Implementation
    // =========================================================================
    // This implements a state-machine based generator that can be iterated.
    // The generator step function is compiled from Python bytecode with each
    // YIELD_VALUE becoming a state transition point.
    // =========================================================================

    // Forward declarations for type methods
    static void JITGenerator_dealloc(JITGeneratorObject* self);
    static PyObject* JITGenerator_iter(JITGeneratorObject* self);
    static PyObject* JITGenerator_iternext(JITGeneratorObject* self);
    static PyObject* JITGenerator_send(JITGeneratorObject* self, PyObject* value);
    static PyObject* JITGenerator_throw(JITGeneratorObject* self, PyObject* args);
    static PyObject* JITGenerator_close(JITGeneratorObject* self, PyObject* args);
    static PyObject* JITGenerator_repr(JITGeneratorObject* self);
    static PyObject* JITGenerator_set_local(JITGeneratorObject* self, PyObject* args);

    // Method definitions for generator type
    static PyMethodDef JITGenerator_methods[] = {
        {"send", (PyCFunction)JITGenerator_send, METH_O, "Send a value into the generator."},
        {"throw", (PyCFunction)JITGenerator_throw, METH_VARARGS, "Throw an exception into the generator."},
        {"close", (PyCFunction)JITGenerator_close, METH_NOARGS, "Close the generator."},
        {"_set_local", (PyCFunction)JITGenerator_set_local, METH_VARARGS, "Set a local variable (internal use)."},
        {NULL, NULL, 0, NULL}
    };

    // Python type object for JIT generators
    // Using C++17 compatible initialization (no designated initializers)
    PyTypeObject JITGenerator_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "justjit.JITGenerator",           // tp_name
        sizeof(JITGeneratorObject),        // tp_basicsize
        0,                                 // tp_itemsize
        (destructor)JITGenerator_dealloc,  // tp_dealloc
        0,                                 // tp_vectorcall_offset
        0,                                 // tp_getattr
        0,                                 // tp_setattr
        0,                                 // tp_as_async
        (reprfunc)JITGenerator_repr,       // tp_repr
        0,                                 // tp_as_number
        0,                                 // tp_as_sequence
        0,                                 // tp_as_mapping
        0,                                 // tp_hash
        0,                                 // tp_call
        0,                                 // tp_str
        0,                                 // tp_getattro
        0,                                 // tp_setattro
        0,                                 // tp_as_buffer
        Py_TPFLAGS_DEFAULT,                // tp_flags
        "JIT-compiled generator object",  // tp_doc
        0,                                 // tp_traverse
        0,                                 // tp_clear
        0,                                 // tp_richcompare
        0,                                 // tp_weaklistoffset
        (getiterfunc)JITGenerator_iter,    // tp_iter
        (iternextfunc)JITGenerator_iternext, // tp_iternext
        JITGenerator_methods,              // tp_methods
    };

    // Deallocate generator object
    static void JITGenerator_dealloc(JITGeneratorObject* self)
    {
        // Decref all local variables
        if (self->locals != nullptr) {
            for (Py_ssize_t i = 0; i < self->num_locals; i++) {
                Py_XDECREF(self->locals[i]);
            }
            PyMem_Free(self->locals);
        }
        Py_XDECREF(self->name);
        Py_XDECREF(self->qualname);
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    // Return self for iteration
    static PyObject* JITGenerator_iter(JITGeneratorObject* self)
    {
        Py_INCREF(self);
        return (PyObject*)self;
    }

    // Get next value from generator
    static PyObject* JITGenerator_iternext(JITGeneratorObject* self)
    {
        // Send None to get next value
        return JITGenerator_Send(self, Py_None);
    }

    // Send value into generator (core implementation)
    PyObject* JITGenerator_Send(JITGeneratorObject* gen, PyObject* value)
    {
        // Check if generator is exhausted
        if (gen->state == -1) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }

        // Check if generator hit an error
        if (gen->state == -2) {
            PyErr_SetString(PyExc_RuntimeError, "generator raised StopIteration");
            return NULL;
        }

        // Cannot send non-None value to just-started generator
        if (gen->state == 0 && value != Py_None) {
            PyErr_SetString(PyExc_TypeError, 
                "can't send non-None value to a just-started generator");
            return NULL;
        }

        // Call the step function
        PyObject* result = gen->step_func(&gen->state, gen->locals, value);

        // Check if generator is done
        if (gen->state == -1) {
            // Generator returned (not yielded)
            // result is the return value, set as StopIteration value
            if (result != NULL) {
                PyObject* stop = PyObject_CallFunctionObjArgs(PyExc_StopIteration, result, NULL);
                if (stop != NULL) {
                    PyErr_SetObject(PyExc_StopIteration, stop);
                    Py_DECREF(stop);
                }
                Py_DECREF(result);
            } else {
                // No return value, just stop
                PyErr_SetNone(PyExc_StopIteration);
            }
            return NULL;
        }

        return result;  // Return yielded value
    }

    // Python-visible send method
    static PyObject* JITGenerator_send(JITGeneratorObject* self, PyObject* value)
    {
        return JITGenerator_Send(self, value);
    }

    // Throw exception into generator
    static PyObject* JITGenerator_throw(JITGeneratorObject* self, PyObject* args)
    {
        PyObject* typ;
        PyObject* val = NULL;
        PyObject* tb = NULL;

        if (!PyArg_ParseTuple(args, "O|OO:throw", &typ, &val, &tb)) {
            return NULL;
        }

        // Mark generator as errored
        self->state = -2;

        // Raise the exception
        if (PyExceptionInstance_Check(typ)) {
            PyErr_SetObject((PyObject*)Py_TYPE(typ), typ);
        } else if (PyExceptionClass_Check(typ)) {
            PyErr_SetObject(typ, val);
        } else {
            PyErr_SetString(PyExc_TypeError, "throw() argument must be an exception");
        }

        return NULL;
    }

    // Close the generator
    static PyObject* JITGenerator_close(JITGeneratorObject* self, PyObject* args)
    {
        (void)args;  // Unused
        
        if (self->state >= 0) {
            // Generator is still running, mark as done
            self->state = -1;
            
            // Clear all locals to release references (fix memory leak)
            if (self->locals != nullptr) {
                for (Py_ssize_t i = 0; i < self->num_locals; i++) {
                    Py_CLEAR(self->locals[i]);
                }
            }
        }
        Py_RETURN_NONE;
    }

    // String representation
    static PyObject* JITGenerator_repr(JITGeneratorObject* self)
    {
        if (self->qualname != NULL) {
            return PyUnicode_FromFormat("<jit_generator object %S at %p>", 
                self->qualname, (void*)self);
        } else if (self->name != NULL) {
            return PyUnicode_FromFormat("<jit_generator object %S at %p>",
                self->name, (void*)self);
        }
        return PyUnicode_FromFormat("<jit_generator object at %p>", (void*)self);
    }

    // Set a local variable in the generator (used to initialize arguments)
    static PyObject* JITGenerator_set_local(JITGeneratorObject* self, PyObject* args)
    {
        Py_ssize_t index;
        PyObject* value;

        if (!PyArg_ParseTuple(args, "nO:_set_local", &index, &value)) {
            return NULL;
        }

        if (index < 0 || index >= self->num_locals) {
            PyErr_SetString(PyExc_IndexError, "local variable index out of range");
            return NULL;
        }

        if (self->locals == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "generator has no local variables");
            return NULL;
        }

        // Decref old value if present
        Py_XDECREF(self->locals[index]);

        // Set new value with incref
        Py_INCREF(value);
        self->locals[index] = value;

        Py_RETURN_NONE;
    }

    // Create a new JIT generator object
    PyObject* JITGenerator_New(GeneratorStepFunc step_func, Py_ssize_t num_locals,
                               PyObject* name, PyObject* qualname)
    {
        // Initialize type if needed (once per process)
        static bool type_ready = false;
        if (!type_ready) {
            if (PyType_Ready(&JITGenerator_Type) < 0) {
                return NULL;
            }
            type_ready = true;
        }

        JITGeneratorObject* gen = PyObject_New(JITGeneratorObject, &JITGenerator_Type);
        if (gen == NULL) {
            return NULL;
        }

        gen->state = 0;  // Initial state (not started)
        gen->step_func = step_func;
        gen->num_locals = num_locals;

        // Allocate locals array
        if (num_locals > 0) {
            gen->locals = (PyObject**)PyMem_Calloc(num_locals, sizeof(PyObject*));
            if (gen->locals == NULL) {
                Py_DECREF(gen);
                PyErr_NoMemory();
                return NULL;
            }
        } else {
            gen->locals = NULL;
        }

        // Store name and qualname
        Py_XINCREF(name);
        gen->name = name;
        Py_XINCREF(qualname);
        gen->qualname = qualname;

        return (PyObject*)gen;
    }

    // =========================================================================
    // JIT Coroutine Implementation
    // =========================================================================
    // Coroutines are like generators but implement the awaitable protocol.
    // They are used for async/await functions.
    // Key differences from generators:
    // - Have __await__() method that returns self
    // - Can await other awaitables (SEND opcode)
    // - Use GET_AWAITABLE to validate awaited objects
    // =========================================================================

    // Forward declarations for coroutine type methods
    static void JITCoroutine_dealloc(JITCoroutineObject* self);
    static PyObject* JITCoroutine_await(JITCoroutineObject* self);
    static PyObject* JITCoroutine_iter(JITCoroutineObject* self);
    static PyObject* JITCoroutine_iternext(JITCoroutineObject* self);
    static PyObject* JITCoroutine_send(JITCoroutineObject* self, PyObject* value);
    static PyObject* JITCoroutine_throw(JITCoroutineObject* self, PyObject* args);
    static PyObject* JITCoroutine_close(JITCoroutineObject* self, PyObject* args);
    static PyObject* JITCoroutine_repr(JITCoroutineObject* self);
    static PyObject* JITCoroutine_set_local(JITCoroutineObject* self, PyObject* args);

    // Method definitions for coroutine type
    static PyMethodDef JITCoroutine_methods[] = {
        {"send", (PyCFunction)JITCoroutine_send, METH_O, "Send a value into the coroutine."},
        {"throw", (PyCFunction)JITCoroutine_throw, METH_VARARGS, "Throw an exception into the coroutine."},
        {"close", (PyCFunction)JITCoroutine_close, METH_NOARGS, "Close the coroutine."},
        {"_set_local", (PyCFunction)JITCoroutine_set_local, METH_VARARGS, "Set a local variable (internal use)."},
        {"__await__", (PyCFunction)JITCoroutine_await, METH_NOARGS, "Return an iterator for await expression."},
        {NULL, NULL, 0, NULL}
    };

    // Async methods for coroutine protocol
    static PyAsyncMethods JITCoroutine_as_async = {
        (unaryfunc)JITCoroutine_await,  // am_await
        0,                               // am_aiter
        0,                               // am_anext
        0,                               // am_send (Python 3.10+)
    };

    // Python type object for JIT coroutines
    PyTypeObject JITCoroutine_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "justjit.JITCoroutine",           // tp_name
        sizeof(JITCoroutineObject),        // tp_basicsize
        0,                                 // tp_itemsize
        (destructor)JITCoroutine_dealloc,  // tp_dealloc
        0,                                 // tp_vectorcall_offset
        0,                                 // tp_getattr
        0,                                 // tp_setattr
        &JITCoroutine_as_async,            // tp_as_async
        (reprfunc)JITCoroutine_repr,       // tp_repr
        0,                                 // tp_as_number
        0,                                 // tp_as_sequence
        0,                                 // tp_as_mapping
        0,                                 // tp_hash
        0,                                 // tp_call
        0,                                 // tp_str
        0,                                 // tp_getattro
        0,                                 // tp_setattro
        0,                                 // tp_as_buffer
        Py_TPFLAGS_DEFAULT,                // tp_flags
        "JIT-compiled coroutine object",  // tp_doc
        0,                                 // tp_traverse
        0,                                 // tp_clear
        0,                                 // tp_richcompare
        0,                                 // tp_weaklistoffset
        (getiterfunc)JITCoroutine_iter,    // tp_iter
        (iternextfunc)JITCoroutine_iternext, // tp_iternext
        JITCoroutine_methods,              // tp_methods
    };

    // Deallocate coroutine object
    static void JITCoroutine_dealloc(JITCoroutineObject* self)
    {
        // Decref all local variables
        if (self->locals != nullptr) {
            for (Py_ssize_t i = 0; i < self->num_locals; i++) {
                Py_XDECREF(self->locals[i]);
            }
            PyMem_Free(self->locals);
        }
        Py_XDECREF(self->name);
        Py_XDECREF(self->qualname);
        Py_XDECREF(self->awaiting);
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    // Return self for await expression (__await__ method)
    static PyObject* JITCoroutine_await(JITCoroutineObject* self)
    {
        Py_INCREF(self);
        return (PyObject*)self;
    }

    // Return self for iteration
    static PyObject* JITCoroutine_iter(JITCoroutineObject* self)
    {
        Py_INCREF(self);
        return (PyObject*)self;
    }

    // Get next value from coroutine
    static PyObject* JITCoroutine_iternext(JITCoroutineObject* self)
    {
        // Send None to get next value
        return JITCoroutine_Send(self, Py_None);
    }

    // Send value into coroutine (core implementation)
    PyObject* JITCoroutine_Send(JITCoroutineObject* coro, PyObject* value)
    {
        // Check if coroutine is exhausted
        if (coro->state == -1) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }

        // Check if coroutine hit an error
        if (coro->state == -2) {
            PyErr_SetString(PyExc_RuntimeError, "coroutine raised StopIteration");
            return NULL;
        }

        // Cannot send non-None value to just-started coroutine
        if (coro->state == 0 && value != Py_None) {
            PyErr_SetString(PyExc_TypeError, 
                "can't send non-None value to a just-started coroutine");
            return NULL;
        }

        // If we're awaiting something, delegate to it first
        if (coro->awaiting != NULL) {
            PyObject* result = NULL;
            
            // Try to send value to the awaited object
            if (PyGen_Check(coro->awaiting) || PyCoro_CheckExact(coro->awaiting)) {
                // Native coroutine or generator - use send
                PyObject* send_meth = PyObject_GetAttrString(coro->awaiting, "send");
                if (send_meth != NULL) {
                    result = PyObject_CallOneArg(send_meth, value);
                    Py_DECREF(send_meth);
                }
            } else {
                // Iterator - use __next__ (ignoring sent value)
                result = PyIter_Next(coro->awaiting);
            }
            
            if (result != NULL) {
                // Awaited object yielded a value - propagate it
                return result;
            }
            
            // Awaited object finished or raised exception
            if (PyErr_Occurred()) {
                if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
                    // Get the return value from StopIteration
                    PyObject *exc_type, *exc_val, *exc_tb;
                    PyErr_Fetch(&exc_type, &exc_val, &exc_tb);
                    
                    PyObject* return_value = Py_None;
                    Py_INCREF(return_value);
                    
                    if (exc_val != NULL && PyObject_HasAttrString(exc_val, "value")) {
                        PyObject* val = PyObject_GetAttrString(exc_val, "value");
                        if (val != NULL) {
                            Py_DECREF(return_value);
                            return_value = val;
                        }
                    }
                    
                    Py_XDECREF(exc_type);
                    Py_XDECREF(exc_val);
                    Py_XDECREF(exc_tb);
                    
                    // Clear awaiting
                    Py_CLEAR(coro->awaiting);
                    
                    // Continue with the return value as sent value
                    value = return_value;
                    // Fall through to call step function with return value
                    Py_DECREF(return_value);  // Will be re-incref'd by step function
                } else {
                    // Some other exception - propagate it
                    Py_CLEAR(coro->awaiting);
                    return NULL;
                }
            } else {
                // No result and no exception - iterator exhausted
                Py_CLEAR(coro->awaiting);
                value = Py_None;
            }
        }

        // Call the step function
        PyObject* result = coro->step_func(&coro->state, coro->locals, value);

        // Check if coroutine is done
        if (coro->state == -1) {
            // Coroutine returned (not yielded)
            if (result != NULL) {
                PyObject* stop = PyObject_CallFunctionObjArgs(PyExc_StopIteration, result, NULL);
                if (stop != NULL) {
                    PyErr_SetObject(PyExc_StopIteration, stop);
                    Py_DECREF(stop);
                }
                Py_DECREF(result);
            } else {
                PyErr_SetNone(PyExc_StopIteration);
            }
            return NULL;
        }

        return result;  // Return yielded value (for event loop)
    }

    // Python-visible send method
    static PyObject* JITCoroutine_send(JITCoroutineObject* self, PyObject* value)
    {
        return JITCoroutine_Send(self, value);
    }

    // Throw exception into coroutine
    static PyObject* JITCoroutine_throw(JITCoroutineObject* self, PyObject* args)
    {
        PyObject* typ;
        PyObject* val = NULL;
        PyObject* tb = NULL;

        if (!PyArg_ParseTuple(args, "O|OO:throw", &typ, &val, &tb)) {
            return NULL;
        }

        // If awaiting something, throw into it first
        if (self->awaiting != NULL) {
            PyObject* throw_meth = PyObject_GetAttrString(self->awaiting, "throw");
            if (throw_meth != NULL) {
                PyObject* result = PyObject_CallFunctionObjArgs(throw_meth, typ, val, tb, NULL);
                Py_DECREF(throw_meth);
                if (result != NULL) {
                    return result;  // Awaited object handled the exception
                }
                // Exception propagated or new exception raised
            }
            Py_CLEAR(self->awaiting);
        }

        // Mark coroutine as errored
        self->state = -2;

        // Raise the exception
        if (PyExceptionInstance_Check(typ)) {
            PyErr_SetObject((PyObject*)Py_TYPE(typ), typ);
        } else if (PyExceptionClass_Check(typ)) {
            PyErr_SetObject(typ, val);
        } else {
            PyErr_SetString(PyExc_TypeError, "throw() argument must be an exception");
        }

        return NULL;
    }

    // Close the coroutine
    static PyObject* JITCoroutine_close(JITCoroutineObject* self, PyObject* args)
    {
        (void)args;  // Unused
        
        // If awaiting something, close it first
        if (self->awaiting != NULL) {
            PyObject* close_meth = PyObject_GetAttrString(self->awaiting, "close");
            if (close_meth != NULL) {
                PyObject* result = PyObject_CallNoArgs(close_meth);
                Py_XDECREF(result);
                Py_DECREF(close_meth);
            }
            Py_CLEAR(self->awaiting);
        }
        
        if (self->state >= 0) {
            self->state = -1;
            
            // Clear all locals to release references (fix memory leak)
            if (self->locals != nullptr) {
                for (Py_ssize_t i = 0; i < self->num_locals; i++) {
                    Py_CLEAR(self->locals[i]);
                }
            }
        }
        Py_RETURN_NONE;
    }

    // String representation
    static PyObject* JITCoroutine_repr(JITCoroutineObject* self)
    {
        if (self->qualname != NULL) {
            return PyUnicode_FromFormat("<jit_coroutine object %S at %p>", 
                self->qualname, (void*)self);
        } else if (self->name != NULL) {
            return PyUnicode_FromFormat("<jit_coroutine object %S at %p>",
                self->name, (void*)self);
        }
        return PyUnicode_FromFormat("<jit_coroutine object at %p>", (void*)self);
    }

    // Set a local variable in the coroutine
    static PyObject* JITCoroutine_set_local(JITCoroutineObject* self, PyObject* args)
    {
        Py_ssize_t index;
        PyObject* value;

        if (!PyArg_ParseTuple(args, "nO:_set_local", &index, &value)) {
            return NULL;
        }

        if (index < 0 || index >= self->num_locals) {
            PyErr_SetString(PyExc_IndexError, "local variable index out of range");
            return NULL;
        }

        if (self->locals == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "coroutine has no local variables");
            return NULL;
        }

        Py_XDECREF(self->locals[index]);
        Py_INCREF(value);
        self->locals[index] = value;

        Py_RETURN_NONE;
    }

    // Create a new JIT coroutine object
    PyObject* JITCoroutine_New(GeneratorStepFunc step_func, Py_ssize_t num_locals,
                               PyObject* name, PyObject* qualname)
    {
        // Initialize type if needed (once per process)
        static bool type_ready = false;
        if (!type_ready) {
            if (PyType_Ready(&JITCoroutine_Type) < 0) {
                return NULL;
            }
            type_ready = true;
        }

        JITCoroutineObject* coro = PyObject_New(JITCoroutineObject, &JITCoroutine_Type);
        if (coro == NULL) {
            return NULL;
        }

        coro->state = 0;  // Initial state (not started)
        coro->step_func = step_func;
        coro->num_locals = num_locals;
        coro->awaiting = NULL;  // Not currently awaiting anything

        // Allocate locals array
        if (num_locals > 0) {
            coro->locals = (PyObject**)PyMem_Calloc(num_locals, sizeof(PyObject*));
            if (coro->locals == NULL) {
                Py_DECREF(coro);
                PyErr_NoMemory();
                return NULL;
            }
        } else {
            coro->locals = NULL;
        }

        // Store name and qualname
        Py_XINCREF(name);
        coro->name = name;
        Py_XINCREF(qualname);
        coro->qualname = qualname;

        return (PyObject*)coro;
    }

} // namespace justjit
