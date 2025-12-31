#include "jit_core.h"
#include "raii_wrapper.h"
#include "opcodes.h"
#include "type_system.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
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
#include <set>
#include <map>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <complex>

// Clang includes for inline C compilation
#ifdef JUSTJIT_HAS_CLANG
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <fstream>
#include <cstdlib>
#include <sstream>
#endif // JUSTJIT_HAS_CLANG

// Python code object flags - define if not available
// CO_ITERABLE_COROUTINE marks generators that can be used in await expressions
// (i.e., decorated with @types.coroutine)
#ifndef CO_ITERABLE_COROUTINE
#define CO_ITERABLE_COROUTINE 0x0100
#endif

// LLVM API compatibility: getDeclaration was renamed to getOrInsertDeclaration in LLVM 20+
// Check LLVM version and provide compatibility macro
#include <llvm/Config/llvm-config.h>
#if LLVM_VERSION_MAJOR >= 20
#define LLVM_GET_INTRINSIC_DECLARATION llvm::Intrinsic::getOrInsertDeclaration
#else
#define LLVM_GET_INTRINSIC_DECLARATION llvm::Intrinsic::getDeclaration
#endif

// C helper function for NULL-safe Py_XINCREF (since Py_XINCREF is a macro)
extern "C" JIT_EXPORT void jit_xincref(PyObject *obj)
{
    Py_XINCREF(obj);
}

// C helper function for NULL-safe Py_XDECREF (since Py_XDECREF is a macro)
extern "C" JIT_EXPORT void jit_xdecref(PyObject *obj)
{
    Py_XDECREF(obj);
}

// =========================================================================
// Box/Unbox Helper Functions (Phase 1 Type System)
// =========================================================================
// These functions convert between Python objects and native C types.
// They are called at function entry (unbox) and exit (box) for typed mode.
// =========================================================================

// Unbox Python int to native int64
extern "C" JIT_EXPORT int64_t jit_unbox_int(PyObject *obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot unbox None to int");
        return -1;
    }
    return PyLong_AsLongLong(obj);
}

// Unbox Python float to native double
extern "C" JIT_EXPORT double jit_unbox_float(PyObject *obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot unbox None to float");
        return -1.0;
    }
    return PyFloat_AsDouble(obj);
}

// Unbox Python bool to native int (0 or 1)
extern "C" JIT_EXPORT int64_t jit_unbox_bool(PyObject *obj)
{
    if (obj == NULL) {
        return 0;
    }
    return PyObject_IsTrue(obj);
}

// Box native int64 to Python int
extern "C" JIT_EXPORT PyObject *jit_box_int(int64_t val)
{
    return PyLong_FromLongLong(val);
}

// Box native double to Python float
extern "C" JIT_EXPORT PyObject *jit_box_float(double val)
{
    return PyFloat_FromDouble(val);
}

// Box native bool (0/1) to Python bool
extern "C" JIT_EXPORT PyObject *jit_box_bool(int64_t val)
{
    return PyBool_FromLong(val);
}

// C helper function for CALL_KW opcode
// Splits args array into positional tuple and kwargs dict based on kwnames tuple
extern "C" JIT_EXPORT PyObject *jit_call_with_kwargs(
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
extern "C" JIT_EXPORT PyObject *JITGetAwaitable(PyObject *obj)
{
    // Check if it's a native coroutine by checking type name
    // (avoids using PyCoro_CheckExact which has symbol issues on some platforms)
    const char* type_name = Py_TYPE(obj)->tp_name;
    if (strcmp(type_name, "coroutine") == 0) {
        Py_INCREF(obj);
        return obj;
    }
    
    // Check if it's a generator with CO_ITERABLE_COROUTINE flag
    // (decorated with @types.coroutine)
    if (strcmp(type_name, "generator") == 0) {
        // Access gi_code attribute to get code flags
        PyObject *gi_code = PyObject_GetAttrString(obj, "gi_code");
        if (gi_code != NULL) {
            PyObject *co_flags_obj = PyObject_GetAttrString(gi_code, "co_flags");
            Py_DECREF(gi_code);
            if (co_flags_obj != NULL) {
                long flags = PyLong_AsLong(co_flags_obj);
                Py_DECREF(co_flags_obj);
                if (flags & CO_ITERABLE_COROUTINE) {
                    Py_INCREF(obj);
                    return obj;
                }
            }
        }
        PyErr_Clear();  // Clear any errors from attribute access
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

// C helper function for GET_AITER opcode
// Gets an async iterator from an object by calling __aiter__
// Equivalent to Python's aiter() builtin
extern "C" PyObject *JITGetAIter(PyObject *obj)
{
    // Use the C API function PyObject_GetAIter (available since Python 3.10)
    // This calls obj.__aiter__() and validates the result
    PyObject *aiter = PyObject_GetAIter(obj);
    if (aiter == NULL) {
        // PyObject_GetAIter already sets appropriate TypeError
        return NULL;
    }
    return aiter;
}

// C helper function for GET_ANEXT opcode
// Gets the next awaitable from an async iterator by calling __anext__
// Returns an awaitable object that yields the next value
extern "C" PyObject *JITGetANext(PyObject *aiter)
{
    // Call __anext__ method on the async iterator
    PyObject *anext_method = PyObject_GetAttrString(aiter, "__anext__");
    if (anext_method == NULL) {
        PyErr_Format(PyExc_TypeError,
            "async iterator has no __anext__ method");
        return NULL;
    }
    
    // Call __anext__() - this returns an awaitable
    PyObject *awaitable = PyObject_CallNoArgs(anext_method);
    Py_DECREF(anext_method);
    
    if (awaitable == NULL) {
        return NULL;
    }
    
    // The result should be an awaitable - wrap it similar to GET_AWAITABLE
    // Most __anext__ implementations return a coroutine or awaitable directly
    return awaitable;
}

// C helper function for END_ASYNC_FOR opcode
// Handles exception at end of async for loop
// If exception is StopAsyncIteration, clears it and returns success (1)
// Otherwise re-raises the exception and returns failure (0)
extern "C" int JITEndAsyncFor(PyObject *exc)
{
    if (exc == NULL) {
        return 1;  // No exception, success
    }
    
    // Check if the exception is StopAsyncIteration
    if (PyErr_GivenExceptionMatches(exc, PyExc_StopAsyncIteration)) {
        PyErr_Clear();
        return 1;  // Expected end of iteration
    }
    
    // Re-raise other exceptions
    PyErr_SetObject((PyObject*)Py_TYPE(exc), exc);
    return 0;  // Failure - exception should propagate
}

// C helper function for ASYNC_GEN_WRAP intrinsic
// Wraps a yielded value from an async generator
// This is needed to distinguish yielded values from awaited values
extern "C" PyObject *JITAsyncGenWrap(PyObject *value)
{
    // In CPython, this creates a _PyAsyncGenWrappedValue
    // Since that's an internal type, we'll use a simpler approach:
    // Create a tuple with a marker and the value
    // Format: ("__jit_async_gen_wrap__", value)
    // 
    // Note: For full compatibility, we should use the actual CPython
    // internal function _PyAsyncGenValueWrapperNew, but that's not
    // part of the stable C API. This approach works for JIT generators.
    
    PyObject *marker = PyUnicode_FromString("__jit_async_gen_wrap__");
    if (marker == NULL) {
        return NULL;
    }
    
    PyObject *wrapped = PyTuple_Pack(2, marker, value);
    Py_DECREF(marker);
    
    if (wrapped == NULL) {
        return NULL;
    }
    
    // Incref the value since we're keeping a reference in the tuple
    // The caller is responsible for the original value's refcount
    return wrapped;
}

// C helper function to unwrap an async generator wrapped value
// Returns the unwrapped value if it's wrapped, NULL otherwise (not an error)
extern "C" PyObject *JITAsyncGenUnwrap(PyObject *obj)
{
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) != 2) {
        return NULL;
    }
    
    PyObject *marker = PyTuple_GET_ITEM(obj, 0);
    if (!PyUnicode_Check(marker)) {
        return NULL;
    }
    
    const char *marker_str = PyUnicode_AsUTF8(marker);
    if (marker_str == NULL || strcmp(marker_str, "__jit_async_gen_wrap__") != 0) {
        PyErr_Clear();  // Clear any error from AsUTF8
        return NULL;
    }
    
    PyObject *value = PyTuple_GET_ITEM(obj, 1);
    Py_INCREF(value);
    return value;
}

// C helper function for MATCH_KEYS opcode
// Extracts values from a mapping for the given keys tuple
// Returns a tuple of values if all keys found, Py_None (incref'd) otherwise
extern "C" PyObject *JITMatchKeys(PyObject *subject, PyObject *keys)
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
extern "C" PyObject *JITMatchClass(PyObject *subject, PyObject *cls, int nargs, PyObject *names)
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

// Debug helper function for tracing generator execution
// Prints offset, opcode name, stack depth, and optionally a value
extern "C" JIT_EXPORT void jit_debug_trace(int offset, const char* opname, int stack_depth, PyObject* value)
{
    fprintf(stderr, "[JIT TRACE] @%d %s | stack_depth=%d", offset, opname, stack_depth);
    if (value != NULL) {
        PyObject* repr = PyObject_Repr(value);
        if (repr != NULL) {
            const char* repr_str = PyUnicode_AsUTF8(repr);
            if (repr_str != NULL) {
                fprintf(stderr, " | value=%s", repr_str);
            }
            Py_DECREF(repr);
        }
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

// Debug helper to print stack contents
extern "C" JIT_EXPORT void jit_debug_stack(const char* label, PyObject** stack_base, int stack_size)
{
    fprintf(stderr, "[JIT STACK] %s | size=%d | [", label, stack_size);
    for (int i = 0; i < stack_size; i++) {
        if (i > 0) fprintf(stderr, ", ");
        PyObject* obj = stack_base[i];
        if (obj == NULL) {
            fprintf(stderr, "NULL");
        } else {
            PyObject* repr = PyObject_Repr(obj);
            if (repr != NULL) {
                const char* repr_str = PyUnicode_AsUTF8(repr);
                if (repr_str != NULL) {
                    // Truncate long reprs
                    if (strlen(repr_str) > 50) {
                        fprintf(stderr, "%.47s...", repr_str);
                    } else {
                        fprintf(stderr, "%s", repr_str);
                    }
                }
                Py_DECREF(repr);
            } else {
                fprintf(stderr, "<%s>", Py_TYPE(obj)->tp_name);
            }
        }
    }
    fprintf(stderr, "]\n");
    fflush(stderr);
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

        // Register jit_xdecref helper (NULL-safe Py_XDECREF)
        helper_symbols[es.intern("jit_xdecref")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_xdecref),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register JITGetAwaitable helper for GET_AWAITABLE opcode
        helper_symbols[es.intern("JITGetAwaitable")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITGetAwaitable),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register JITMatchKeys helper for MATCH_KEYS opcode
        helper_symbols[es.intern("JITMatchKeys")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITMatchKeys),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register JITMatchClass helper for MATCH_CLASS opcode
        helper_symbols[es.intern("JITMatchClass")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITMatchClass),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register async iteration helpers for async generators
        helper_symbols[es.intern("JITGetAIter")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITGetAIter),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("JITGetANext")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITGetANext),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("JITEndAsyncFor")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITEndAsyncFor),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("JITAsyncGenWrap")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITAsyncGenWrap),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("JITAsyncGenUnwrap")] = {
            llvm::orc::ExecutorAddr::fromPtr(JITAsyncGenUnwrap),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register debug trace helpers
        helper_symbols[es.intern("jit_debug_trace")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_debug_trace),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_debug_stack")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_debug_stack),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Register box/unbox helpers (Phase 1 Type System)
        helper_symbols[es.intern("jit_unbox_int")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_unbox_int),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_unbox_float")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_unbox_float),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_unbox_bool")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_unbox_bool),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_box_int")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_box_int),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_box_float")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_box_float),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_box_bool")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_box_bool),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // RAII helper symbols for inline C code
        helper_symbols[es.intern("jit_gil_acquire")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_gil_acquire),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_gil_release")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_gil_release),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_gil_release_begin")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_gil_release_begin),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_gil_release_end")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_gil_release_end),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_pyobj_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_pyobj_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_pyobj_free")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_pyobj_free),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_pyobj_get")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_pyobj_get),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_buffer_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_buffer_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_buffer_free")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_buffer_free),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_buffer_data")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_buffer_data),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_buffer_size")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_buffer_size),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_py_to_long")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_py_to_long),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_py_to_double")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_py_to_double),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_long_to_py")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_long_to_py),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_double_to_py")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_double_to_py),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        helper_symbols[es.intern("jit_call_python")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_python),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // List operations
        helper_symbols[es.intern("jit_list_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_list_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_list_size")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_list_size),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_list_get")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_list_get),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_list_set")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_list_set),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_list_append")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_list_append),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Dict operations
        helper_symbols[es.intern("jit_dict_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_get")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_get),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_get_obj")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_get_obj),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_set")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_set),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_set_obj")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_set_obj),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_del")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_del),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_dict_keys")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_dict_keys),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Tuple operations
        helper_symbols[es.intern("jit_tuple_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_tuple_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_tuple_get")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_tuple_get),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_tuple_set")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_tuple_set),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Object attribute/method access
        helper_symbols[es.intern("jit_getattr")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_getattr),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_setattr")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_setattr),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_hasattr")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_hasattr),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call_method")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_method),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call_method0")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_method0),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Reference counting
        helper_symbols[es.intern("jit_incref")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_incref),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_decref")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_decref),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Module import
        helper_symbols[es.intern("jit_import")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_import),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Sequence/iterator operations
        helper_symbols[es.intern("jit_len")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_len),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_getitem")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_getitem),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_setitem")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_setitem),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_getitem_obj")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_getitem_obj),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_setitem_obj")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_setitem_obj),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Type checking
        helper_symbols[es.intern("jit_is_list")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_list),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_dict")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_dict),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_tuple")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_tuple),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_int")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_int),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_float")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_float),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_str")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_str),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_none")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_none),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_is_callable")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_is_callable),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Constants
        helper_symbols[es.intern("jit_none")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_none),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_true")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_true),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_false")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_false),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Error handling
        helper_symbols[es.intern("jit_error_occurred")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_error_occurred),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_error_clear")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_error_clear),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_error_print")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_error_print),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // String conversion
        helper_symbols[es.intern("jit_py_to_string")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_py_to_string),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_string_to_py")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_string_to_py),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Enhanced callbacks
        helper_symbols[es.intern("jit_call1")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call1),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call2")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call2),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call3")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call3),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call_method1")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_method1),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_call_method2")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_call_method2),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Argument builders
        helper_symbols[es.intern("jit_build_args1")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_args1),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_args2")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_args2),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_args3")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_args3),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_int_args1")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_int_args1),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_int_args2")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_int_args2),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_float_args1")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_float_args1),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_build_float_args2")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_build_float_args2),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Iterator support
        helper_symbols[es.intern("jit_get_iter")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_get_iter),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_iter_next")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_iter_next),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_iter_check")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_iter_check),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Bytes support
        helper_symbols[es.intern("jit_bytes_new")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_bytes_new),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_bytes_data")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_bytes_data),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_bytes_len")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_bytes_len),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};

        // Simplified Python expression evaluation
        helper_symbols[es.intern("jit_py_eval")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_py_eval),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
        helper_symbols[es.intern("jit_py_exec")] = {
            llvm::orc::ExecutorAddr::fromPtr(jit_py_exec),
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

        // void jit_xdecref(PyObject* o) - our NULL-safe wrapper for Py_XDECREF
        llvm::FunctionType *xdecref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
        py_xdecref_func = llvm::Function::Create(xdecref_type, llvm::Function::ExternalLinkage, "jit_xdecref", module);

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

        // ========== Box/Unbox API (Phase 1 Type System) ==========

        // long long PyLong_AsLongLong(PyObject* obj)
        // Convert Python int to C long long (64-bit signed integer)
        // Returns -1 on error with exception set
        llvm::FunctionType *long_aslonglong_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
        py_long_aslonglong_func = llvm::Function::Create(long_aslonglong_type, llvm::Function::ExternalLinkage, "PyLong_AsLongLong", module);

        // double PyFloat_AsDouble(PyObject* obj)
        // Convert Python float to C double
        // Returns -1.0 on error with exception set
        llvm::FunctionType *float_asdouble_type = llvm::FunctionType::get(builder->getDoubleTy(), {ptr_type}, false);
        py_float_asdouble_func = llvm::Function::Create(float_asdouble_type, llvm::Function::ExternalLinkage, "PyFloat_AsDouble", module);

        // PyObject* PyFloat_FromDouble(double v)
        // Create Python float from C double
        // Returns new reference on success, NULL on failure
        llvm::FunctionType *float_fromdouble_type = llvm::FunctionType::get(ptr_type, {builder->getDoubleTy()}, false);
        py_float_fromdouble_func = llvm::Function::Create(float_fromdouble_type, llvm::Function::ExternalLinkage, "PyFloat_FromDouble", module);

        // PyObject* PyBool_FromLong(long v)
        // Return Py_True or Py_False depending on v
        // Always succeeds (no NULL return)
        llvm::FunctionType *bool_fromlong_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
        py_bool_fromlong_func = llvm::Function::Create(bool_fromlong_type, llvm::Function::ExternalLinkage, "PyBool_FromLong", module);

        // PyObject* jit_call_with_kwargs(PyObject* callable, PyObject** args, Py_ssize_t nargs, PyObject* kwnames)
        // Our C helper for CALL_KW opcode - splits args and builds kwargs dict at runtime
        llvm::FunctionType *call_with_kwargs_type = llvm::FunctionType::get(
            ptr_type, {ptr_type, ptr_type, i64_type, ptr_type}, false);
        jit_call_with_kwargs_func = llvm::Function::Create(call_with_kwargs_type, llvm::Function::ExternalLinkage, "jit_call_with_kwargs", module);

        // void jit_debug_trace(int offset, const char* opname, int stack_depth, PyObject* value)
        // Debug helper for tracing generator execution
        llvm::Type *i32_type = builder->getInt32Ty();
        llvm::FunctionType *debug_trace_type = llvm::FunctionType::get(
            void_type, {i32_type, ptr_type, i32_type, ptr_type}, false);
        jit_debug_trace_func = llvm::Function::Create(debug_trace_type, llvm::Function::ExternalLinkage, "jit_debug_trace", module);

        // void jit_debug_stack(const char* label, PyObject** stack_base, int stack_size)
        // Debug helper for printing stack contents
        llvm::FunctionType *debug_stack_type = llvm::FunctionType::get(
            void_type, {ptr_type, ptr_type, i32_type}, false);
        jit_debug_stack_func = llvm::Function::Create(debug_stack_type, llvm::Function::ExternalLinkage, "jit_debug_stack", module);

        // ========== Async Generator Support ==========

        // PyObject* JITGetAIter(PyObject* obj)
        // Get async iterator from object by calling __aiter__
        // Equivalent to Python's aiter() builtin
        llvm::FunctionType *jit_get_aiter_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        jit_get_aiter_func = llvm::Function::Create(jit_get_aiter_type, llvm::Function::ExternalLinkage, "JITGetAIter", module);

        // PyObject* JITGetANext(PyObject* aiter)
        // Get next awaitable from async iterator by calling __anext__
        llvm::FunctionType *jit_get_anext_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        jit_get_anext_func = llvm::Function::Create(jit_get_anext_type, llvm::Function::ExternalLinkage, "JITGetANext", module);

        // int JITEndAsyncFor(PyObject* exc)
        // Handle exception at end of async for loop
        // Returns 1 if StopAsyncIteration (success), 0 otherwise (propagate)
        llvm::FunctionType *jit_end_async_for_type = llvm::FunctionType::get(i32_type, {ptr_type}, false);
        jit_end_async_for_func = llvm::Function::Create(jit_end_async_for_type, llvm::Function::ExternalLinkage, "JITEndAsyncFor", module);

        // PyObject* JITAsyncGenWrap(PyObject* value)
        // Wrap yielded value from async generator
        llvm::FunctionType *jit_async_gen_wrap_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        jit_async_gen_wrap_func = llvm::Function::Create(jit_async_gen_wrap_type, llvm::Function::ExternalLinkage, "JITAsyncGenWrap", module);

        // PyObject* JITAsyncGenUnwrap(PyObject* obj)
        // Unwrap async generator wrapped value
        llvm::FunctionType *jit_async_gen_unwrap_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
        jit_async_gen_unwrap_func = llvm::Function::Create(jit_async_gen_unwrap_type, llvm::Function::ExternalLinkage, "JITAsyncGenUnwrap", module);
    }

    // =========================================================================
    // CFG Analysis Helper Functions
    // =========================================================================
    // These functions perform control flow analysis on Python bytecode to
    // determine basic block boundaries, predecessors/successors, and stack
    // depths at each block entry for proper PHI node generation.
    // =========================================================================

    // Identify all basic block start offsets from bytecode
    // Block boundaries occur at:
    // 1. Start of function (offset 0)
    // 2. Jump targets (POP_JUMP_IF_*, JUMP_*, FOR_ITER targets)
    // 3. Fall-through after conditional jumps
    // 4. Exception handler entry points
    static std::set<int> find_block_starts(
        const std::vector<Instruction>& instructions,
        const std::vector<ExceptionTableEntry>& exception_table)
    {
        std::set<int> block_starts;
        block_starts.insert(0);  // Entry block always starts at 0

        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto& instr = instructions[i];

            // Jump targets create new blocks
            if (instr.opcode == op::POP_JUMP_IF_FALSE || 
                instr.opcode == op::POP_JUMP_IF_TRUE ||
                instr.opcode == op::POP_JUMP_IF_NONE || 
                instr.opcode == op::POP_JUMP_IF_NOT_NONE)
            {
                // Target of the jump
                block_starts.insert(instr.argval);
                // Fall-through to next instruction
                if (i + 1 < instructions.size())
                {
                    block_starts.insert(instructions[i + 1].offset);
                }
            }
            else if (instr.opcode == op::JUMP_FORWARD || instr.opcode == op::JUMP_BACKWARD)
            {
                block_starts.insert(instr.argval);
                // Fall-through is not reachable for unconditional jumps,
                // but the next instruction might be a target of another jump
                if (i + 1 < instructions.size())
                {
                    // Only add if it's the start of a new logical block
                    // (could be dead code otherwise)
                    block_starts.insert(instructions[i + 1].offset);
                }
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                // FOR_ITER jumps forward on exhaustion
                block_starts.insert(instr.argval);
                // Fall-through when iterator has more
                if (i + 1 < instructions.size())
                {
                    block_starts.insert(instructions[i + 1].offset);
                }
            }
        }

        // Exception handlers are block starts
        for (const auto& exc_entry : exception_table)
        {
            block_starts.insert(exc_entry.target);
        }

        return block_starts;
    }

    // Build CFG: map each block start to its BasicBlockInfo
    static std::map<int, BasicBlockInfo> build_cfg(
        const std::vector<Instruction>& instructions,
        const std::vector<ExceptionTableEntry>& exception_table,
        const std::set<int>& block_starts)
    {
        std::map<int, BasicBlockInfo> cfg;

        // Initialize all blocks
        std::vector<int> sorted_starts(block_starts.begin(), block_starts.end());
        std::sort(sorted_starts.begin(), sorted_starts.end());

        for (size_t b = 0; b < sorted_starts.size(); ++b)
        {
            int start = sorted_starts[b];
            BasicBlockInfo info;
            info.start_offset = start;
            info.end_offset = (b + 1 < sorted_starts.size()) ? sorted_starts[b + 1] : 
                (instructions.empty() ? start : instructions.back().offset + 2);
            info.stack_depth_at_entry = -1;  // Unknown initially
            info.is_exception_handler = false;
            info.needs_phi_nodes = false;
            info.llvm_block = nullptr;
            cfg[start] = info;
        }

        // Mark exception handlers
        for (const auto& exc_entry : exception_table)
        {
            if (cfg.count(exc_entry.target))
            {
                cfg[exc_entry.target].is_exception_handler = true;
                // Exception handlers have a specific stack depth
                cfg[exc_entry.target].stack_depth_at_entry = exc_entry.depth;
            }
        }

        // Build predecessor/successor edges by analyzing instructions
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto& instr = instructions[i];
            int current_offset = instr.offset;

            // Find which block this instruction belongs to
            int current_block_start = -1;
            for (auto it = block_starts.rbegin(); it != block_starts.rend(); ++it)
            {
                if (*it <= current_offset)
                {
                    current_block_start = *it;
                    break;
                }
            }
            if (current_block_start < 0) continue;

            // Analyze control flow instructions
            if (instr.opcode == op::POP_JUMP_IF_FALSE || 
                instr.opcode == op::POP_JUMP_IF_TRUE ||
                instr.opcode == op::POP_JUMP_IF_NONE || 
                instr.opcode == op::POP_JUMP_IF_NOT_NONE)
            {
                int target = instr.argval;
                int fall_through = (i + 1 < instructions.size()) ? instructions[i + 1].offset : -1;

                // Add edges
                if (cfg.count(target))
                {
                    cfg[current_block_start].successors.push_back(target);
                    cfg[target].predecessors.push_back(current_block_start);
                }
                if (fall_through >= 0 && cfg.count(fall_through))
                {
                    cfg[current_block_start].successors.push_back(fall_through);
                    cfg[fall_through].predecessors.push_back(current_block_start);
                }
            }
            else if (instr.opcode == op::JUMP_FORWARD || instr.opcode == op::JUMP_BACKWARD)
            {
                int target = instr.argval;
                if (cfg.count(target))
                {
                    cfg[current_block_start].successors.push_back(target);
                    cfg[target].predecessors.push_back(current_block_start);
                }
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                int target = instr.argval;  // Jump when exhausted
                int fall_through = (i + 1 < instructions.size()) ? instructions[i + 1].offset : -1;

                if (cfg.count(target))
                {
                    cfg[current_block_start].successors.push_back(target);
                    cfg[target].predecessors.push_back(current_block_start);
                }
                if (fall_through >= 0 && cfg.count(fall_through))
                {
                    cfg[current_block_start].successors.push_back(fall_through);
                    cfg[fall_through].predecessors.push_back(current_block_start);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE || instr.opcode == op::RETURN_CONST)
            {
                // No successors - this ends the function
            }
            else if (i + 1 < instructions.size())
            {
                // Check if next instruction starts a new block (fall-through edge)
                int next_offset = instructions[i + 1].offset;
                if (cfg.count(next_offset) && current_block_start != next_offset)
                {
                    // Only add if this is the last instruction of current block
                    if (cfg[current_block_start].end_offset == next_offset)
                    {
                        cfg[current_block_start].successors.push_back(next_offset);
                        cfg[next_offset].predecessors.push_back(current_block_start);
                    }
                }
            }
        }

        // Mark blocks that need PHI nodes (multiple predecessors)
        for (auto& [offset, info] : cfg)
        {
            if (info.predecessors.size() > 1)
            {
                info.needs_phi_nodes = true;
            }
        }

        return cfg;
    }

    // Compute stack depth at entry for each block using dataflow analysis
    // Returns true if analysis succeeded, false if inconsistent
    static bool compute_stack_depths(
        std::map<int, BasicBlockInfo>& cfg,
        const std::vector<Instruction>& instructions,
        int initial_stack_depth = 0)
    {
        if (cfg.empty()) return true;

        // Entry block starts with initial_stack_depth (usually 0)
        auto entry_it = cfg.find(0);
        if (entry_it != cfg.end())
        {
            entry_it->second.stack_depth_at_entry = initial_stack_depth;
        }

        // Worklist algorithm for dataflow
        std::set<int> worklist;
        for (const auto& [offset, info] : cfg)
        {
            worklist.insert(offset);
        }

        // Map opcode to stack effect (delta)
        auto get_stack_effect = [](const Instruction& instr) -> int
        {
            // Stack effects for common opcodes
            // Positive = push, Negative = pop
            switch (instr.opcode)
            {
                // Push 1
                case op::LOAD_FAST:
                case op::LOAD_CONST:
                case op::LOAD_GLOBAL:
                case op::LOAD_NAME:
                case op::LOAD_ATTR:
                case op::LOAD_DEREF:
                case op::BUILD_TUPLE:  // pops N, pushes 1 - handled specially
                case op::BUILD_LIST:
                case op::BUILD_SET:
                case op::BUILD_MAP:
                    return 1;

                // Pop 1
                case op::POP_TOP:
                case op::STORE_FAST:
                case op::STORE_NAME:
                case op::STORE_GLOBAL:
                case op::STORE_DEREF:
                case op::POP_JUMP_IF_FALSE:
                case op::POP_JUMP_IF_TRUE:
                case op::POP_JUMP_IF_NONE:
                case op::POP_JUMP_IF_NOT_NONE:
                case op::RETURN_VALUE:
                    return -1;

                // Pop 2, Push 1
                case op::BINARY_OP:
                case op::COMPARE_OP:
                case op::CONTAINS_OP:
                case op::IS_OP:
                    return -1;  // Net effect: -2 + 1 = -1

                // Pop 2
                case op::STORE_ATTR:
                case op::STORE_SUBSCR:
                    return -2;

                // Pop 3
                case op::STORE_SLICE:
                    return -3;

                // MATCH opcodes: push result without popping subject
                case op::MATCH_SEQUENCE:
                case op::MATCH_MAPPING:
                    return 1;  // Push bool result, subject stays

                // MATCH_KEYS: pops nothing, pushes 1 (values tuple or None)
                case op::MATCH_KEYS:
                    return 1;

                // MATCH_CLASS: pops 2 (cls, names), pushes 1 (attrs or None)
                case op::MATCH_CLASS:
                    return -1;

                // Neutral
                case op::RESUME:
                case op::CACHE:
                case op::NOP:
                case op::EXTENDED_ARG:
                case op::JUMP_FORWARD:
                case op::JUMP_BACKWARD:
                    return 0;

                // FOR_ITER: pushes 1 on continue, pops 1 on exhaustion
                case op::FOR_ITER:
                    return 1;  // When continuing

                // GET_ITER: pops 1, pushes 1 (iterator)
                case op::GET_ITER:
                    return 0;

                // CALL: pops callable + N args, pushes result
                case op::CALL:
                    // Stack effect depends on arg, but net is -(arg+1)+1 = -arg
                    return -(int)instr.arg;

                // COPY: duplicates TOS[i] to TOS
                case op::COPY:
                    return 1;

                // SWAP: swaps, no net change
                case op::SWAP:
                    return 0;

                // UNPACK_SEQUENCE: pops 1, pushes N
                case op::UNPACK_SEQUENCE:
                    return (int)instr.arg - 1;

                default:
                    return 0;  // Unknown - assume neutral
            }
        };

        int iterations = 0;
        const int max_iterations = 1000;  // Safety limit

        while (!worklist.empty() && iterations++ < max_iterations)
        {
            int block_offset = *worklist.begin();
            worklist.erase(worklist.begin());

            auto& block = cfg[block_offset];
            
            // Skip if we don't know entry depth yet
            if (block.stack_depth_at_entry < 0)
            {
                // Check if any predecessor has known exit depth
                for (int pred_offset : block.predecessors)
                {
                    if (cfg.count(pred_offset) && cfg[pred_offset].stack_depth_at_entry >= 0)
                    {
                        // Compute predecessor's exit depth
                        int pred_depth = cfg[pred_offset].stack_depth_at_entry;
                        
                        // Find instructions in predecessor block
                        for (const auto& instr : instructions)
                        {
                            if (instr.offset >= cfg[pred_offset].start_offset &&
                                instr.offset < cfg[pred_offset].end_offset)
                            {
                                pred_depth += get_stack_effect(instr);
                                if (pred_depth < 0) pred_depth = 0;  // Safety
                            }
                        }

                        block.stack_depth_at_entry = pred_depth;
                        break;
                    }
                }

                // Still unknown - re-add to worklist if predecessors might update
                if (block.stack_depth_at_entry < 0 && !block.predecessors.empty())
                {
                    worklist.insert(block_offset);
                }
                continue;
            }

            // Compute exit depth for this block
            int exit_depth = block.stack_depth_at_entry;
            for (const auto& instr : instructions)
            {
                if (instr.offset >= block.start_offset && instr.offset < block.end_offset)
                {
                    exit_depth += get_stack_effect(instr);
                    if (exit_depth < 0) exit_depth = 0;  // Safety
                }
            }

            // Propagate to successors
            for (int succ_offset : block.successors)
            {
                if (!cfg.count(succ_offset)) continue;
                auto& succ = cfg[succ_offset];

                if (succ.stack_depth_at_entry < 0)
                {
                    succ.stack_depth_at_entry = exit_depth;
                    worklist.insert(succ_offset);
                }
                else if (succ.stack_depth_at_entry != exit_depth)
                {
                    // Inconsistency - this can happen with complex control flow
                    // Mark as needing PHI nodes
                    succ.needs_phi_nodes = true;
                }
            }
        }

        return iterations < max_iterations;
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

        // =====================================================================
        // CFG Analysis: Build control flow graph for proper PHI node placement
        // This enables support for complex control flow like pattern matching
        // =====================================================================
        std::set<int> block_starts = find_block_starts(instructions, exception_table);
        std::map<int, BasicBlockInfo> cfg = build_cfg(instructions, exception_table, block_starts);
        compute_stack_depths(cfg, instructions, 0);

        // Mark blocks that need PHI nodes based on CFG analysis
        for (const auto& [offset, info] : cfg)
        {
            if (info.needs_phi_nodes)
            {
                block_needs_phi[offset] = true;
            }
        }

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

        // First pass: Create basic blocks for all CFG block starts
        // This ensures we have blocks at all merge points for PHI nodes
        jump_targets[0] = entry;
        
        // Create blocks for all CFG block starts (except entry which already exists)
        for (int block_offset : block_starts)
        {
            if (block_offset == 0) continue;  // Entry block already created
            
            if (!jump_targets.count(block_offset))
            {
                std::string block_name;
                if (cfg.count(block_offset) && cfg[block_offset].is_exception_handler)
                {
                    block_name = "exc_handler_" + std::to_string(block_offset);
                }
                else if (cfg.count(block_offset) && cfg[block_offset].needs_phi_nodes)
                {
                    block_name = "merge_" + std::to_string(block_offset);
                }
                else
                {
                    block_name = "block_" + std::to_string(block_offset);
                }
                jump_targets[block_offset] = llvm::BasicBlock::Create(
                    *local_context, block_name, func);
                
                // Store LLVM block reference in CFG
                if (cfg.count(block_offset))
                {
                    cfg[block_offset].llvm_block = jump_targets[block_offset];
                }
            }
        }
        
        // Store entry block in CFG
        if (cfg.count(0))
        {
            cfg[0].llvm_block = entry;
        }

        // Also create blocks for jump targets not in block_starts (legacy compatibility)
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
        
        // Helper to check if current block is a dead block (unreachable code placeholder)
        // This includes "dead_" blocks from switch_to_dead_block() and "unreachable_" blocks
        // from JUMP_BACKWARD which creates unreachable_after_jump blocks with terminators
        auto is_dead_block = [&]() -> bool
        {
            llvm::BasicBlock *current_block = builder.GetInsertBlock();
            if (!current_block) return false;
            llvm::StringRef name = current_block->getName();
            return name.starts_with("dead_") || name.starts_with("unreachable_");
        };
        
        // Track whether we're currently in an unreachable code region
        // This propagates across blocks when we enter from a dead block with no legitimate predecessors
        bool in_unreachable_region = false;

        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            int current_offset = instructions[i].offset;

            // If this offset is a jump target, switch to that block and handle PHI nodes
            if (jump_targets.count(current_offset) && jump_targets[current_offset] != builder.GetInsertBlock())
            {
                // Record current stack state before branching (for fall-through)
                // But skip if we're in a dead block or unreachable region - its stack state is invalid
                llvm::BasicBlock *current_block = builder.GetInsertBlock();
                bool from_dead_block = is_dead_block() || in_unreachable_region;
                
                if (!current_block->getTerminator() && !from_dead_block)
                {
                    // Record stack state for this predecessor
                    BlockStackState state;
                    state.stack = stack;
                    state.predecessor = current_block;
                    block_incoming_stacks[current_offset].push_back(state);

                    builder.CreateBr(jump_targets[current_offset]);
                }
                builder.SetInsertPoint(jump_targets[current_offset]);
                
                // Check if the target block has legitimate predecessors (recorded stacks)
                // If not, we're entering dead code from dead code
                if (!block_incoming_stacks.count(current_offset) || 
                    block_incoming_stacks[current_offset].empty())
                {
                    // No legitimate predecessors - this block is unreachable
                    // Mark that we're in unreachable region
                    in_unreachable_region = true;
                }
                else
                {
                    // This block has legitimate predecessors - we're back to live code
                    in_unreachable_region = false;
                }

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
            else if (instr.opcode == op::GET_LEN)
            {
                // GET_LEN: Push len(TOS) onto stack without popping TOS
                // Used in match statements for sequence length comparison
                // Stack: [..., obj] -> [..., obj, len(obj)]
                if (!stack.empty())
                {
                    llvm::Value *obj = stack.back();
                    // Don't pop - GET_LEN leaves object on stack
                    
                    // Box if needed
                    if (obj->getType()->isIntegerTy(64))
                    {
                        obj = builder.CreateCall(py_long_fromlonglong_func, {obj});
                        stack.back() = obj;
                        builder.CreateCall(py_incref_func, {obj});
                    }
                    
                    // Call PyObject_Size to get length
                    llvm::FunctionType *py_object_size_type = llvm::FunctionType::get(
                        i64_type, {ptr_type}, false);
                    llvm::FunctionCallee py_object_size_func = module->getOrInsertFunction(
                        "PyObject_Size", py_object_size_type);
                    llvm::Value *length = builder.CreateCall(py_object_size_func, {obj}, "len");
                    
                    // Convert to PyLong for stack (consistent with other stack values)
                    llvm::Value *len_obj = builder.CreateCall(py_long_fromlonglong_func, {length}, "len_obj");
                    stack.push_back(len_obj);
                }
            }
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
                    // Note: PyBytes_Check, PyUnicode_Check, PyByteArray_Check are macros, not functions.
                    // We use PyObject_IsInstance with the type objects instead.

                    // Check if it's a sequence
                    llvm::FunctionType *py_sequence_check_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type}, false);
                    llvm::FunctionCallee py_sequence_check_func = module->getOrInsertFunction(
                        "PySequence_Check", py_sequence_check_type);
                    llvm::Value *is_sequence = builder.CreateCall(py_sequence_check_func, {subject}, "is_sequence");

                    // PyObject_IsInstance returns 1 if true, 0 if false, -1 on error
                    llvm::FunctionType *py_isinstance_type = llvm::FunctionType::get(
                        llvm::Type::getInt32Ty(*local_context),
                        {ptr_type, ptr_type}, false);
                    llvm::FunctionCallee py_isinstance_func = module->getOrInsertFunction(
                        "PyObject_IsInstance", py_isinstance_type);

                    // Get type objects for str, bytes, bytearray
                    llvm::Value *unicode_type_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(&PyUnicode_Type));
                    llvm::Value *unicode_type = builder.CreateIntToPtr(unicode_type_ptr, ptr_type);
                    
                    llvm::Value *bytes_type_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(&PyBytes_Type));
                    llvm::Value *bytes_type = builder.CreateIntToPtr(bytes_type_ptr, ptr_type);
                    
                    llvm::Value *bytearray_type_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(&PyByteArray_Type));
                    llvm::Value *bytearray_type = builder.CreateIntToPtr(bytearray_type_ptr, ptr_type);

                    // Check isinstance for each type
                    llvm::Value *is_unicode = builder.CreateCall(py_isinstance_func, {subject, unicode_type}, "is_unicode");
                    llvm::Value *is_bytes = builder.CreateCall(py_isinstance_func, {subject, bytes_type}, "is_bytes");
                    llvm::Value *is_bytearray = builder.CreateCall(py_isinstance_func, {subject, bytearray_type}, "is_bytearray");

                    // Result = is_sequence && !is_unicode && !is_bytes && !is_bytearray
                    // isinstance returns > 0 for true
                    llvm::Value *zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*local_context), 0);
                    llvm::Value *seq_ok = builder.CreateICmpSGT(is_sequence, zero, "seq_ok");
                    llvm::Value *not_unicode = builder.CreateICmpSLE(is_unicode, zero, "not_unicode");
                    llvm::Value *not_bytes = builder.CreateICmpSLE(is_bytes, zero, "not_bytes");
                    llvm::Value *not_bytearray = builder.CreateICmpSLE(is_bytearray, zero, "not_bytearray");
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

                    // Call helper: PyObject* JITMatchKeys(PyObject* subject, PyObject* keys)
                    // Returns tuple of values if all keys found, or None if any key missing
                    // The helper handles incref on the result
                    llvm::FunctionType *match_keys_helper_type = llvm::FunctionType::get(
                        ptr_type, {ptr_type, ptr_type}, false);
                    llvm::FunctionCallee match_keys_helper = module->getOrInsertFunction(
                        "JITMatchKeys", match_keys_helper_type);
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

                    // Call helper: PyObject* JITMatchClass(subject, cls, nargs, names)
                    // Returns tuple of matched attributes if successful, Py_None (incref'd) otherwise
                    llvm::FunctionType *match_class_helper_type = llvm::FunctionType::get(
                        ptr_type, {ptr_type, ptr_type, llvm::Type::getInt32Ty(*local_context), ptr_type}, false);
                    llvm::FunctionCallee match_class_helper = module->getOrInsertFunction(
                        "JITMatchClass", match_class_helper_type);
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
                    // After return, switch to a dead block to prevent stale stack values
                    // from being used when compiling subsequent unreachable code
                    switch_to_dead_block();
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
                    // After return, switch to a dead block to prevent stale stack values
                    // from being used when compiling subsequent unreachable code
                    switch_to_dead_block();
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

                    // FIX: In Python 3.12+, FOR_ITER jumps directly to END_FOR (argval).
                    // The iterator STAYS on the stack; END_FOR will pop it.
                    // Do NOT add +4 to skip instructions - let them execute normally.
                    int end_for_offset = instr.argval;  // This is where we jump on exhaustion
                    int next_offset = instructions[i + 1].offset;

                    if (!jump_targets.count(end_for_offset))
                    {
                        jump_targets[end_for_offset] = llvm::BasicBlock::Create(
                            *local_context, "end_for_" + std::to_string(end_for_offset), func);
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

                    // Exhausted path: Jump to END_FOR with iterator still on stack
                    // END_FOR will pop and decref the iterator
                    builder.SetInsertPoint(exhausted_block);
                    // Clear any StopIteration exception from PyIter_Next
                    builder.CreateCall(py_err_clear_func, {});

                    // Record the exhaust-path stack state for end_for_offset
                    // Iterator is STILL on the stack (END_FOR will pop it)
                    {
                        BlockStackState exhaust_state;
                        exhaust_state.predecessor = exhausted_block;
                        exhaust_state.stack = stack;  // Iterator still on stack
                        block_incoming_stacks[end_for_offset].push_back(exhaust_state);
                    }

                    builder.CreateBr(jump_targets[end_for_offset]);

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
            // ========== Async With Support ==========
            else if (instr.opcode == op::BEFORE_ASYNC_WITH)
            {
                // BEFORE_ASYNC_WITH: Set up an async with block
                // Python 3.13: Resolves __aenter__ and __aexit__ from STACK[-1]
                // Pushes __aexit__ and result of __aenter__() to the stack
                // Stack: TOS=context_manager -> __aexit__, __aenter__() result
                if (!stack.empty())
                {
                    llvm::Value *context_mgr = stack.back();
                    stack.pop_back();

                    // Get __aexit__ method
                    llvm::Value *aexit_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(PyUnicode_FromString("__aexit__")));
                    llvm::Value *aexit_name = builder.CreateIntToPtr(aexit_name_ptr, ptr_type);
                    llvm::Value *aexit_method = builder.CreateCall(py_object_getattr_func, {context_mgr, aexit_name}, "aexit_method");
                    check_error_and_branch(current_offset, aexit_method, "before_async_with_aexit");

                    // Get __aenter__ method
                    llvm::Value *aenter_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(PyUnicode_FromString("__aenter__")));
                    llvm::Value *aenter_name = builder.CreateIntToPtr(aenter_name_ptr, ptr_type);
                    llvm::Value *aenter_method = builder.CreateCall(py_object_getattr_func, {context_mgr, aenter_name}, "aenter_method");
                    check_error_and_branch(current_offset, aenter_method, "before_async_with_aenter");

                    // Call __aenter__() - returns an awaitable
                    llvm::Value *empty_args = builder.CreateCall(py_tuple_new_func, {llvm::ConstantInt::get(i64_type, 0)}, "empty_args");
                    llvm::Value *null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *aenter_result = builder.CreateCall(py_object_call_func, {aenter_method, empty_args, null_kwargs}, "aenter_result");
                    builder.CreateCall(py_decref_func, {empty_args});
                    builder.CreateCall(py_decref_func, {aenter_method});
                    builder.CreateCall(py_decref_func, {context_mgr});
                    check_error_and_branch(current_offset, aenter_result, "before_async_with_call");

                    // Push __aexit__ and aenter result (the awaitable)
                    stack.push_back(aexit_method);
                    stack.push_back(aenter_result);
                }
            }
            // ========== Exception Group Matching ==========
            else if (instr.opcode == op::CHECK_EG_MATCH)
            {
                // CHECK_EG_MATCH: Match exception group for except*
                // Python 3.13: Applies split(STACK[-1]) on exception group at STACK[-2]
                // On match: pop 2, push non-matching subgroup (or None) + matching subgroup
                // On no match: pop 1 (match type), push None
                // For now, push None to indicate no match (conservative fallback)
                if (stack.size() >= 2)
                {
                    llvm::Value *match_type = stack.back();
                    stack.pop_back();
                    // Leave exception group on stack, push None for "no match"
                    builder.CreateCall(py_decref_func, {match_type});

                    llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                    builder.CreateCall(py_incref_func, {py_none});
                    stack.push_back(py_none);
                }
            }
            // ========== Init Check ==========
            else if (instr.opcode == op::EXIT_INIT_CHECK)
            {
                // EXIT_INIT_CHECK: Verify __init__ returned None
                // Python 3.13: If TOS is not None, raise TypeError
                // Stack: TOS=init_return_value (popped)
                if (!stack.empty())
                {
                    llvm::Value *init_result = stack.back();
                    stack.pop_back();

                    // Check if result is None
                    llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                    llvm::Value *is_none = builder.CreateICmpEQ(init_result, py_none, "is_none");

                    llvm::BasicBlock *ok_block = llvm::BasicBlock::Create(*local_context, "init_ok", func);
                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(*local_context, "init_error", func);

                    builder.CreateCondBr(is_none, ok_block, error_block);

                    // Error block: raise TypeError
                    builder.SetInsertPoint(error_block);
                    llvm::Value *type_error_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(PyExc_TypeError));
                    llvm::Value *type_error = builder.CreateIntToPtr(type_error_ptr, ptr_type);
                    llvm::Value *err_msg_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(PyUnicode_FromString("__init__ returned non-None")));
                    llvm::Value *err_msg = builder.CreateIntToPtr(err_msg_ptr, ptr_type);
                    builder.CreateCall(py_err_set_object_func, {type_error, err_msg});
                    builder.CreateCall(py_decref_func, {init_result});
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                    // OK block: continue
                    builder.SetInsertPoint(ok_block);
                    builder.CreateCall(py_decref_func, {init_result});
                }
            }
            // ========== Annotation Scope Operations ==========
            else if (instr.opcode == op::LOAD_LOCALS)
            {
                // LOAD_LOCALS: Push reference to locals dictionary
                // Python 3.13: Used to prepare namespace for LOAD_FROM_DICT_OR_DEREF/GLOBALS
                // For JIT compiled functions, we use the globals dict at module level
                llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                    i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                llvm::Value *locals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "locals_dict");
                builder.CreateCall(py_incref_func, {locals_dict});
                stack.push_back(locals_dict);
            }
            else if (instr.opcode == op::LOAD_FROM_DICT_OR_DEREF)
            {
                // LOAD_FROM_DICT_OR_DEREF: Pop mapping, lookup name, else load from cell
                // Python 3.13: Used for closure variables in class bodies
                // arg = slot i in fast locals storage
                int slot_idx = instr.arg;

                if (!stack.empty())
                {
                    llvm::Value *mapping = stack.back();
                    stack.pop_back();

                    // For now, just try to load from the mapping using the variable name
                    // If not found, fall back to cell
                    // This is a simplified implementation

                    llvm::Value *result = nullptr;

                    // Try loading from cell if available
                    if (slot_idx < static_cast<int>(closure_cells.size()) && closure_cells[slot_idx] != nullptr)
                    {
                        llvm::Value *cell_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(closure_cells[slot_idx]));
                        llvm::Value *cell = builder.CreateIntToPtr(cell_ptr, ptr_type);
                        result = builder.CreateCall(py_cell_get_func, {cell}, "cell_value");
                        builder.CreateCall(py_incref_func, {result});
                    }
                    else
                    {
                        // No cell, return None as fallback
                        llvm::Value *none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {result});
                    }

                    builder.CreateCall(py_decref_func, {mapping});
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::LOAD_FROM_DICT_OR_GLOBALS)
            {
                // LOAD_FROM_DICT_OR_GLOBALS: Pop mapping, lookup name, else load global
                // Python 3.13: Used for loading global variables in annotation scopes
                // arg = index into co_names
                int name_idx = instr.arg;

                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *mapping = stack.back();
                    stack.pop_back();

                    // Get the name object
                    llvm::Value *name_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type);

                    // Try mapping first
                    llvm::Value *dict_result = builder.CreateCall(py_dict_getitem_func, {mapping, name_obj}, "dict_lookup");

                    llvm::Value *is_null = builder.CreateIsNull(dict_result);
                    llvm::BasicBlock *found_block = llvm::BasicBlock::Create(*local_context, "dict_found", func);
                    llvm::BasicBlock *try_globals_block = llvm::BasicBlock::Create(*local_context, "try_globals", func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(*local_context, "load_continue", func);

                    builder.CreateCondBr(is_null, try_globals_block, found_block);

                    // Found in mapping
                    builder.SetInsertPoint(found_block);
                    builder.CreateCall(py_incref_func, {dict_result}); // Borrowed ref
                    builder.CreateBr(continue_block);

                    // Try globals
                    builder.SetInsertPoint(try_globals_block);
                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type);
                    llvm::Value *global_result = builder.CreateCall(py_dict_getitem_func, {globals_dict, name_obj}, "global_lookup");

                    // Check if found in globals
                    llvm::Value *global_is_null = builder.CreateIsNull(global_result);
                    llvm::BasicBlock *global_found_block = llvm::BasicBlock::Create(*local_context, "global_found", func);
                    llvm::BasicBlock *try_builtins_block = llvm::BasicBlock::Create(*local_context, "try_builtins", func);

                    builder.CreateCondBr(global_is_null, try_builtins_block, global_found_block);

                    // Found in globals
                    builder.SetInsertPoint(global_found_block);
                    builder.CreateCall(py_incref_func, {global_result});
                    builder.CreateBr(continue_block);

                    // Try builtins
                    builder.SetInsertPoint(try_builtins_block);
                    llvm::Value *builtins_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(builtins_dict_ptr));
                    llvm::Value *builtins_dict = builder.CreateIntToPtr(builtins_ptr, ptr_type);
                    llvm::Value *builtin_result = builder.CreateCall(py_dict_getitem_func, {builtins_dict, name_obj}, "builtin_lookup");
                    builder.CreateCall(py_incref_func, {builtin_result});
                    builder.CreateBr(continue_block);

                    // Continue with PHI
                    builder.SetInsertPoint(continue_block);
                    llvm::PHINode *result_phi = builder.CreatePHI(ptr_type, 3, "load_result");
                    result_phi->addIncoming(dict_result, found_block);
                    result_phi->addIncoming(global_result, global_found_block);
                    result_phi->addIncoming(builtin_result, try_builtins_block);

                    builder.CreateCall(py_decref_func, {mapping});
                    stack.push_back(result_phi);
                }
            }
            else if (instr.opcode == op::SETUP_ANNOTATIONS)
            {
                // SETUP_ANNOTATIONS: Create __annotations__ dict if not exists
                // Python 3.13: Checks if __annotations__ is in locals(), if not creates empty dict
                // For JIT, we set it in globals (module level)
                llvm::Value *globals_ptr_val = llvm::ConstantInt::get(
                    i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type);

                // Get "__annotations__" string
                llvm::Value *annot_name_ptr = llvm::ConstantInt::get(
                    i64_type, reinterpret_cast<uint64_t>(PyUnicode_FromString("__annotations__")));
                llvm::Value *annot_name = builder.CreateIntToPtr(annot_name_ptr, ptr_type);

                // Check if __annotations__ exists
                llvm::Value *existing = builder.CreateCall(py_dict_getitem_func, {globals_dict, annot_name}, "existing_annot");
                llvm::Value *is_null = builder.CreateIsNull(existing);

                llvm::BasicBlock *create_block = llvm::BasicBlock::Create(*local_context, "create_annot", func);
                llvm::BasicBlock *done_block = llvm::BasicBlock::Create(*local_context, "annot_done", func);

                builder.CreateCondBr(is_null, create_block, done_block);

                // Create new empty dict
                builder.SetInsertPoint(create_block);
                llvm::Value *new_dict = builder.CreateCall(py_dict_new_func, {}, "new_annot");
                builder.CreateCall(py_dict_setitem_func, {globals_dict, annot_name, new_dict});
                builder.CreateCall(py_decref_func, {new_dict}); // SetItem increfs
                builder.CreateBr(done_block);

                builder.SetInsertPoint(done_block);
            }
            // ========== Call Intrinsic 2 ==========
            else if (instr.opcode == op::CALL_INTRINSIC_2)
            {
                // CALL_INTRINSIC_2: Two-argument intrinsic functions
                // Python 3.13 operands:
                //   1 = INTRINSIC_PREP_RERAISE_STAR - Exception group handling
                //   2 = INTRINSIC_TYPEVAR_WITH_BOUND - typing.TypeVar with bound
                //   3 = INTRINSIC_TYPEVAR_WITH_CONSTRAINTS - typing.TypeVar with constraints
                //   4 = INTRINSIC_SET_FUNCTION_TYPE_PARAMS - Set __type_params__ on function
                if (stack.size() >= 2)
                {
                    llvm::Value *arg2 = stack.back();
                    stack.pop_back();
                    llvm::Value *arg1 = stack.back();
                    stack.pop_back();

                    int intrinsic = instr.arg;
                    llvm::Value *result = nullptr;

                    if (intrinsic == 4)
                    {
                        // INTRINSIC_SET_FUNCTION_TYPE_PARAMS
                        // arg1 = function, arg2 = type_params tuple
                        // Set function.__type_params__ = type_params
                        llvm::Value *attr_name_ptr = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(PyUnicode_FromString("__type_params__")));
                        llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);
                        builder.CreateCall(py_object_setattr_func, {arg1, attr_name, arg2});
                        builder.CreateCall(py_decref_func, {arg2});
                        // Return the function
                        result = arg1;
                    }
                    else
                    {
                        // For other intrinsics (typing-related), return arg1 as fallback
                        // These are rarely used and mainly for type annotations
                        builder.CreateCall(py_decref_func, {arg2});
                        result = arg1;
                    }

                    stack.push_back(result);
                }
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

    // Float-mode callable generators (native f64 -> f64 functions)
    // These bypass PyObject* entirely for maximum performance with floating-point
    nb::object JITCore::create_float_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<double (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> double
                                { return fn_ptr(); });
    }

    nb::object JITCore::create_float_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<double (*)(double)>(func_ptr);
        return nb::cpp_function([fn_ptr](double a) -> double
                                { return fn_ptr(a); });
    }

    nb::object JITCore::create_float_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<double (*)(double, double)>(func_ptr);
        return nb::cpp_function([fn_ptr](double a, double b) -> double
                                { return fn_ptr(a, b); });
    }

    nb::object JITCore::create_float_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<double (*)(double, double, double)>(func_ptr);
        return nb::cpp_function([fn_ptr](double a, double b, double c) -> double
                                { return fn_ptr(a, b, c); });
    }

    nb::object JITCore::create_float_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<double (*)(double, double, double, double)>(func_ptr);
        return nb::cpp_function([fn_ptr](double a, double b, double c, double d) -> double
                                { return fn_ptr(a, b, c, d); });
    }

    nb::object JITCore::get_float_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_float_callable_0(func_ptr);
        case 1:
            return create_float_callable_1(func_ptr);
        case 2:
            return create_float_callable_2(func_ptr);
        case 3:
            return create_float_callable_3(func_ptr);
        case 4:
            return create_float_callable_4(func_ptr);
        default:
            throw std::runtime_error("Float mode supports up to 4 parameters");
        }
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

    // Bool-mode callable generators (native i64 -> Python bool functions)
    // These return True/False based on native 0/1 values
    nb::object JITCore::create_bool_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> bool
                                { return fn_ptr() != 0; });
    }

    nb::object JITCore::create_bool_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](bool a) -> bool
                                { return fn_ptr(a ? 1 : 0) != 0; });
    }

    nb::object JITCore::create_bool_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](bool a, bool b) -> bool
                                { return fn_ptr(a ? 1 : 0, b ? 1 : 0) != 0; });
    }

    nb::object JITCore::create_bool_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](bool a, bool b, bool c) -> bool
                                { return fn_ptr(a ? 1 : 0, b ? 1 : 0, c ? 1 : 0) != 0; });
    }

    nb::object JITCore::create_bool_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](bool a, bool b, bool c, bool d) -> bool
                                { return fn_ptr(a ? 1 : 0, b ? 1 : 0, c ? 1 : 0, d ? 1 : 0) != 0; });
    }

    nb::object JITCore::get_bool_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_bool_callable_0(func_ptr);
        case 1:
            return create_bool_callable_1(func_ptr);
        case 2:
            return create_bool_callable_2(func_ptr);
        case 3:
            return create_bool_callable_3(func_ptr);
        case 4:
            return create_bool_callable_4(func_ptr);
        default:
            throw std::runtime_error("Bool mode supports up to 4 parameters");
        }
    }

    // Int32-mode callable generators (native i32 functions)
    nb::object JITCore::create_int32_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int32_t (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> int32_t { return fn_ptr(); });
    }

    nb::object JITCore::create_int32_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int32_t (*)(int32_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int32_t a) -> int32_t { return fn_ptr(a); });
    }

    nb::object JITCore::create_int32_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int32_t (*)(int32_t, int32_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int32_t a, int32_t b) -> int32_t { return fn_ptr(a, b); });
    }

    nb::object JITCore::create_int32_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int32_t (*)(int32_t, int32_t, int32_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int32_t a, int32_t b, int32_t c) -> int32_t { return fn_ptr(a, b, c); });
    }

    nb::object JITCore::create_int32_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<int32_t (*)(int32_t, int32_t, int32_t, int32_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](int32_t a, int32_t b, int32_t c, int32_t d) -> int32_t { return fn_ptr(a, b, c, d); });
    }

    nb::object JITCore::get_int32_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_int32_callable_0(func_ptr);
        case 1:
            return create_int32_callable_1(func_ptr);
        case 2:
            return create_int32_callable_2(func_ptr);
        case 3:
            return create_int32_callable_3(func_ptr);
        case 4:
            return create_int32_callable_4(func_ptr);
        default:
            throw std::runtime_error("Int32 mode supports up to 4 parameters");
        }
    }

    // Float32-mode callable generators (native f32 functions)
    nb::object JITCore::create_float32_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<float (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> float { return fn_ptr(); });
    }

    nb::object JITCore::create_float32_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<float (*)(float)>(func_ptr);
        return nb::cpp_function([fn_ptr](float a) -> float { return fn_ptr(a); });
    }

    nb::object JITCore::create_float32_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<float (*)(float, float)>(func_ptr);
        return nb::cpp_function([fn_ptr](float a, float b) -> float { return fn_ptr(a, b); });
    }

    nb::object JITCore::create_float32_callable_3(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<float (*)(float, float, float)>(func_ptr);
        return nb::cpp_function([fn_ptr](float a, float b, float c) -> float { return fn_ptr(a, b, c); });
    }

    nb::object JITCore::create_float32_callable_4(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<float (*)(float, float, float, float)>(func_ptr);
        return nb::cpp_function([fn_ptr](float a, float b, float c, float d) -> float { return fn_ptr(a, b, c, d); });
    }

    nb::object JITCore::get_float32_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_float32_callable_0(func_ptr);
        case 1:
            return create_float32_callable_1(func_ptr);
        case 2:
            return create_float32_callable_2(func_ptr);
        case 3:
            return create_float32_callable_3(func_ptr);
        case 4:
            return create_float32_callable_4(func_ptr);
        default:
            throw std::runtime_error("Float32 mode supports up to 4 parameters");
        }
    }

    // Complex128 struct for passing complex numbers by value
    struct Complex128 {
        double real;
        double imag;
    };

    // Complex128-mode callable generators (native {double,double} functions)
    nb::object JITCore::create_complex128_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex128 (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> nb::object {
            Complex128 result = fn_ptr();
            return nb::cast(std::complex<double>(result.real, result.imag));
        });
    }

    nb::object JITCore::create_complex128_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex128 (*)(Complex128)>(func_ptr);
        return nb::cpp_function([fn_ptr](std::complex<double> a) -> nb::object {
            Complex128 arg = {a.real(), a.imag()};
            Complex128 result = fn_ptr(arg);
            return nb::cast(std::complex<double>(result.real, result.imag));
        });
    }

    nb::object JITCore::create_complex128_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex128 (*)(Complex128, Complex128)>(func_ptr);
        return nb::cpp_function([fn_ptr](std::complex<double> a, std::complex<double> b) -> nb::object {
            Complex128 arg1 = {a.real(), a.imag()};
            Complex128 arg2 = {b.real(), b.imag()};
            Complex128 result = fn_ptr(arg1, arg2);
            return nb::cast(std::complex<double>(result.real, result.imag));
        });
    }

    nb::object JITCore::get_complex128_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_complex128_callable_0(func_ptr);
        case 1:
            return create_complex128_callable_1(func_ptr);
        case 2:
            return create_complex128_callable_2(func_ptr);
        default:
            throw std::runtime_error("Complex128 mode supports up to 2 parameters");
        }
    }

    // Complex64 struct for passing single-precision complex numbers by value
    struct Complex64 {
        float real;
        float imag;
    };

    // Complex64-mode callable generators (native {float,float} functions)
    nb::object JITCore::create_complex64_callable_0(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex64 (*)()>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> nb::object {
            Complex64 result = fn_ptr();
            return nb::cast(std::complex<float>(result.real, result.imag));
        });
    }

    nb::object JITCore::create_complex64_callable_1(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex64 (*)(Complex64)>(func_ptr);
        return nb::cpp_function([fn_ptr](std::complex<float> a) -> nb::object {
            Complex64 arg = {a.real(), a.imag()};
            Complex64 result = fn_ptr(arg);
            return nb::cast(std::complex<float>(result.real, result.imag));
        });
    }

    nb::object JITCore::create_complex64_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<Complex64 (*)(Complex64, Complex64)>(func_ptr);
        return nb::cpp_function([fn_ptr](std::complex<float> a, std::complex<float> b) -> nb::object {
            Complex64 arg1 = {a.real(), a.imag()};
            Complex64 arg2 = {b.real(), b.imag()};
            Complex64 result = fn_ptr(arg1, arg2);
            return nb::cast(std::complex<float>(result.real, result.imag));
        });
    }

    nb::object JITCore::get_complex64_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_complex64_callable_0(func_ptr);
        case 1:
            return create_complex64_callable_1(func_ptr);
        case 2:
            return create_complex64_callable_2(func_ptr);
        default:
            throw std::runtime_error("Complex64 mode supports up to 2 parameters");
        }
    }

    // OptionalF64 struct for nullable float64 values
    // Uses int64_t for has_value to ensure ABI compatibility with LLVM {i64, f64}
    struct OptionalF64 {
        int64_t has_value;  // 0 = None, 1 = Some
        double value;
    };

    // OptionalF64-mode callable generators using pointer-based ABI
    // Signature: void fn(OptionalF64* out, OptionalF64* a, OptionalF64* b)
    // Returns None if has_value is 0, otherwise returns the double
    nb::object JITCore::create_optional_f64_callable_0(uint64_t func_ptr)
    {
        // No-arg function: void fn(OptionalF64* out)
        auto fn_ptr = reinterpret_cast<void (*)(OptionalF64*)>(func_ptr);
        return nb::cpp_function([fn_ptr]() -> nb::object {
            OptionalF64 result;
            fn_ptr(&result);
            if (result.has_value) {
                return nb::cast(result.value);
            }
            return nb::none();
        });
    }

    nb::object JITCore::create_optional_f64_callable_1(uint64_t func_ptr)
    {
        // 1-arg function: void fn(OptionalF64* out, OptionalF64* a)
        auto fn_ptr = reinterpret_cast<void (*)(OptionalF64*, OptionalF64*)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a) -> nb::object {
            OptionalF64 arg, result;
            if (a.is_none()) {
                arg = {0, 0.0};
            } else {
                arg = {1, nb::cast<double>(a)};
            }
            fn_ptr(&result, &arg);
            if (result.has_value) {
                return nb::cast(result.value);
            }
            return nb::none();
        });
    }

    nb::object JITCore::create_optional_f64_callable_2(uint64_t func_ptr)
    {
        // 2-arg function: void fn(OptionalF64* out, OptionalF64* a, OptionalF64* b)
        auto fn_ptr = reinterpret_cast<void (*)(OptionalF64*, OptionalF64*, OptionalF64*)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a, nb::object b) -> nb::object {
            OptionalF64 arg1, arg2, result;
            if (a.is_none()) {
                arg1 = {0, 0.0};
            } else {
                arg1 = {1, nb::cast<double>(a)};
            }
            if (b.is_none()) {
                arg2 = {0, 0.0};
            } else {
                arg2 = {1, nb::cast<double>(b)};
            }
            fn_ptr(&result, &arg1, &arg2);
            if (result.has_value) {
                return nb::cast(result.value);
            }
            return nb::none();
        });
    }

    nb::object JITCore::get_optional_f64_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 0:
            return create_optional_f64_callable_0(func_ptr);
        case 1:
            return create_optional_f64_callable_1(func_ptr);
        case 2:
            return create_optional_f64_callable_2(func_ptr);
        default:
            throw std::runtime_error("OptionalF64 mode supports up to 2 parameters");
        }
    }

    // Ptr-mode callable generators (for array operations)
    // Ptr mode takes a pointer (as i64) and index, returns double
    nb::object JITCore::create_ptr_callable_2(uint64_t func_ptr)
    {
        // Function signature: double fn(ptr, i64)
        auto fn_ptr = reinterpret_cast<double (*)(double*, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object arr_obj, int64_t idx) -> double {
            // Handle numpy arrays by extracting data pointer
            double* ptr = nullptr;
            if (nb::hasattr(arr_obj, "ctypes")) {
                // NumPy array - get data pointer via ctypes.data
                nb::object ctypes = arr_obj.attr("ctypes");
                nb::object data = ctypes.attr("data");
                ptr = reinterpret_cast<double*>(nb::cast<uintptr_t>(data));
            } else if (nb::isinstance<nb::int_>(arr_obj)) {
                // Raw pointer passed as integer
                ptr = reinterpret_cast<double*>(nb::cast<uintptr_t>(arr_obj));
            } else {
                throw std::runtime_error("ptr mode requires numpy array or raw pointer");
            }
            return fn_ptr(ptr, idx);
        });
    }

    nb::object JITCore::create_ptr_callable_3(uint64_t func_ptr)
    {
        // Function signature: double fn(ptr, i64, i64) - e.g., array sum with ptr, start, end
        auto fn_ptr = reinterpret_cast<double (*)(double*, int64_t, int64_t)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object arr_obj, int64_t arg1, int64_t arg2) -> double {
            double* ptr = nullptr;
            if (nb::hasattr(arr_obj, "ctypes")) {
                nb::object ctypes = arr_obj.attr("ctypes");
                nb::object data = ctypes.attr("data");
                ptr = reinterpret_cast<double*>(nb::cast<uintptr_t>(data));
            } else if (nb::isinstance<nb::int_>(arr_obj)) {
                ptr = reinterpret_cast<double*>(nb::cast<uintptr_t>(arr_obj));
            } else {
                throw std::runtime_error("ptr mode requires numpy array or raw pointer");
            }
            return fn_ptr(ptr, arg1, arg2);
        });
    }

    nb::object JITCore::get_ptr_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
        {
            throw std::runtime_error("Failed to find JIT function: " + name);
        }

        switch (param_count)
        {
        case 2:
            return create_ptr_callable_2(func_ptr);
        case 3:
            return create_ptr_callable_3(func_ptr);
        default:
            throw std::runtime_error("Ptr mode supports 2-3 parameters (ptr + indices)");
        }
    }

    // Vec4f struct for 4-way float SIMD
    struct alignas(16) Vec4f {
        float data[4];
    };

    // Vec8i struct for 8-way int32 SIMD
    struct alignas(32) Vec8i {
        int32_t data[8];
    };

    // Vec4f-mode callable (takes 2 numpy float32 arrays of length 4, returns float32 array)
    // New signature: void fn(float* out, float* a, float* b)
    nb::object JITCore::create_vec4f_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<void (*)(float*, float*, float*)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a_obj, nb::object b_obj) -> nb::object {
            alignas(16) float a_buf[4] = {};
            alignas(16) float b_buf[4] = {};
            alignas(16) float out_buf[4] = {};
            
            // Extract data from NumPy arrays
            if (nb::hasattr(a_obj, "ctypes")) {
                nb::object ctypes_a = a_obj.attr("ctypes");
                nb::object data_a = ctypes_a.attr("data");
                float* ptr_a = reinterpret_cast<float*>(nb::cast<uintptr_t>(data_a));
                for (int i = 0; i < 4; ++i) a_buf[i] = ptr_a[i];
            }
            if (nb::hasattr(b_obj, "ctypes")) {
                nb::object ctypes_b = b_obj.attr("ctypes");
                nb::object data_b = ctypes_b.attr("data");
                float* ptr_b = reinterpret_cast<float*>(nb::cast<uintptr_t>(data_b));
                for (int i = 0; i < 4; ++i) b_buf[i] = ptr_b[i];
            }
            
            fn_ptr(out_buf, a_buf, b_buf);
            
            nb::list ret;
            for (int i = 0; i < 4; ++i) ret.append(out_buf[i]);
            return ret;
        });
    }

    nb::object JITCore::get_vec4f_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
            throw std::runtime_error("Failed to find JIT function: " + name);

        switch (param_count)
        {
        case 2:
            return create_vec4f_callable_2(func_ptr);
        default:
            throw std::runtime_error("Vec4f mode supports 2 parameters");
        }
    }

    // Vec8i-mode callable (takes 2 numpy int32 arrays of length 8, returns int32 array)
    // New signature: void fn(int32_t* out, int32_t* a, int32_t* b)
    nb::object JITCore::create_vec8i_callable_2(uint64_t func_ptr)
    {
        auto fn_ptr = reinterpret_cast<void (*)(int32_t*, int32_t*, int32_t*)>(func_ptr);
        return nb::cpp_function([fn_ptr](nb::object a_obj, nb::object b_obj) -> nb::object {
            alignas(32) int32_t a_buf[8] = {};
            alignas(32) int32_t b_buf[8] = {};
            alignas(32) int32_t out_buf[8] = {};
            
            if (nb::hasattr(a_obj, "ctypes")) {
                nb::object ctypes_a = a_obj.attr("ctypes");
                nb::object data_a = ctypes_a.attr("data");
                int32_t* ptr_a = reinterpret_cast<int32_t*>(nb::cast<uintptr_t>(data_a));
                for (int i = 0; i < 8; ++i) a_buf[i] = ptr_a[i];
            }
            if (nb::hasattr(b_obj, "ctypes")) {
                nb::object ctypes_b = b_obj.attr("ctypes");
                nb::object data_b = ctypes_b.attr("data");
                int32_t* ptr_b = reinterpret_cast<int32_t*>(nb::cast<uintptr_t>(data_b));
                for (int i = 0; i < 8; ++i) b_buf[i] = ptr_b[i];
            }
            
            fn_ptr(out_buf, a_buf, b_buf);
            
            nb::list ret;
            for (int i = 0; i < 8; ++i) ret.append(out_buf[i]);
            return ret;
        });
    }

    nb::object JITCore::get_vec8i_callable(const std::string &name, int param_count)
    {
        uint64_t func_ptr = lookup_symbol(name);
        if (!func_ptr)
            throw std::runtime_error("Failed to find JIT function: " + name);

        switch (param_count)
        {
        case 2:
            return create_vec8i_callable_2(func_ptr);
        default:
            throw std::runtime_error("Vec8i mode supports 2 parameters");
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

        // First pass: Detect range() loop patterns and check for unsupported opcodes
        // A range loop pattern looks like:
        //   PUSH_NULL (optional in some cases)
        //   LOAD_GLOBAL (range)
        //   LOAD_CONST or LOAD_FAST (stop value)
        //   [LOAD_CONST or LOAD_FAST (start value) - optional for range(start, stop)]
        //   [LOAD_CONST or LOAD_FAST (step value) - optional for range(start, stop, step)]
        //   CALL
        //   GET_ITER
        //   FOR_ITER (loop body follows)
        //   ...
        //   END_FOR
        
        // Track offsets that are part of range loop patterns (allowed in int mode)
        std::unordered_set<int> range_loop_offsets;
        
        // Structure to track detected range loops
        struct RangeLoop {
            int for_iter_idx;      // Index of FOR_ITER instruction
            int end_for_idx;       // Index of END_FOR instruction
            int loop_var_idx;      // Local variable index for loop counter
            int start_const_idx;   // Constant index for start value (-1 if from LOAD_FAST)
            int stop_const_idx;    // Constant index for stop value (-1 if from LOAD_FAST)
            int start_local_idx;   // Local variable index for start (-1 if from constant)
            int stop_local_idx;    // Local variable index for stop (-1 if from constant)
            bool valid;
        };
        std::vector<RangeLoop> detected_range_loops;
        
        // Scan for range() patterns
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            // Look for GET_ITER followed by FOR_ITER pattern
            if (instructions[i].opcode == op::GET_ITER && 
                i + 1 < instructions.size() && 
                instructions[i + 1].opcode == op::FOR_ITER)
            {
                // Found potential range loop - trace back to find CALL and LOAD_GLOBAL
                // Pattern: PUSH_NULL, LOAD_GLOBAL(range), LOAD_CONST/FAST, CALL, GET_ITER, FOR_ITER
                size_t call_idx = i - 1; // CALL should be right before GET_ITER
                
                if (call_idx < instructions.size() && instructions[call_idx].opcode == op::CALL)
                {
                    int arg_count = instructions[call_idx].arg; // Number of args to range()
                    
                    // Python 3.13 can have either:
                    // 1. PUSH_NULL, LOAD_GLOBAL, [args...], CALL (separate opcodes)
                    // 2. LOAD_GLOBAL (with NULL flag), [args...], CALL (combined)
                    // The LOAD_GLOBAL arg encodes: (name_idx << 1) | push_null_flag
                    // So arg & 1 == 1 means NULL is pushed along with LOAD_GLOBAL
                    
                    size_t first_arg_idx = call_idx - arg_count;
                    size_t load_global_idx = first_arg_idx - 1;
                    
                    bool has_separate_push_null = false;
                    size_t push_null_idx = load_global_idx; // May not exist
                    
                    if (load_global_idx < instructions.size() &&
                        instructions[load_global_idx].opcode == op::LOAD_GLOBAL)
                    {
                        // Check if LOAD_GLOBAL has combined NULL flag (arg & 1 == 1)
                        bool combined_null = (instructions[load_global_idx].arg & 1) == 1;
                        
                        if (!combined_null && load_global_idx > 0)
                        {
                            // Check for separate PUSH_NULL before LOAD_GLOBAL
                            push_null_idx = load_global_idx - 1;
                            if (push_null_idx < instructions.size() &&
                                instructions[push_null_idx].opcode == op::PUSH_NULL)
                            {
                                has_separate_push_null = true;
                            }
                        }
                        
                        // Valid range pattern if we have combined NULL or separate PUSH_NULL
                        if (combined_null || has_separate_push_null)
                        {
                            // Mark these offsets as allowed for int mode
                            if (has_separate_push_null)
                            {
                                range_loop_offsets.insert(instructions[push_null_idx].offset);
                            }
                            range_loop_offsets.insert(instructions[load_global_idx].offset);
                            for (size_t j = first_arg_idx; j <= call_idx; ++j)
                            {
                                range_loop_offsets.insert(instructions[j].offset);
                            }
                            range_loop_offsets.insert(instructions[i].offset);     // GET_ITER
                            range_loop_offsets.insert(instructions[i + 1].offset); // FOR_ITER
                            
                            // Find END_FOR
                            int for_iter_target = instructions[i + 1].argval;
                            size_t end_for_idx = i + 2;
                            for (size_t j = i + 2; j < instructions.size(); ++j)
                            {
                                if (instructions[j].opcode == op::END_FOR &&
                                    instructions[j].offset >= for_iter_target - 4) // END_FOR is near target
                                {
                                    end_for_idx = j;
                                    range_loop_offsets.insert(instructions[j].offset);
                                    break;
                                }
                            }
                            
                            // Also mark POP_TOP after END_FOR if present
                            if (end_for_idx + 1 < instructions.size() &&
                                instructions[end_for_idx + 1].opcode == op::POP_TOP)
                            {
                                range_loop_offsets.insert(instructions[end_for_idx + 1].offset);
                            }
                            
                            // Record the range loop info
                            RangeLoop rl;
                            rl.for_iter_idx = i + 1;
                            rl.end_for_idx = end_for_idx;
                            rl.loop_var_idx = -1;
                            rl.valid = true;
                            
                            // Determine start/stop from arguments
                            if (arg_count == 1)
                            {
                                // range(stop) - start is 0
                                rl.start_const_idx = -2; // Special: literal 0
                                if (instructions[first_arg_idx].opcode == op::LOAD_CONST)
                                {
                                    rl.stop_const_idx = instructions[first_arg_idx].arg;
                                    rl.stop_local_idx = -1;
                                }
                                else if (instructions[first_arg_idx].opcode == op::LOAD_FAST)
                                {
                                    rl.stop_const_idx = -1;
                                    rl.stop_local_idx = instructions[first_arg_idx].arg;
                                }
                                rl.start_local_idx = -1;
                            }
                            else if (arg_count >= 2)
                            {
                                // range(start, stop)
                                if (instructions[first_arg_idx].opcode == op::LOAD_CONST)
                                {
                                    rl.start_const_idx = instructions[first_arg_idx].arg;
                                    rl.start_local_idx = -1;
                                }
                                else
                                {
                                    rl.start_const_idx = -1;
                                    rl.start_local_idx = instructions[first_arg_idx].arg;
                                }
                                if (instructions[first_arg_idx + 1].opcode == op::LOAD_CONST)
                                {
                                    rl.stop_const_idx = instructions[first_arg_idx + 1].arg;
                                    rl.stop_local_idx = -1;
                                }
                                else
                                {
                                    rl.stop_const_idx = -1;
                                    rl.stop_local_idx = instructions[first_arg_idx + 1].arg;
                                }
                            }
                            
                            detected_range_loops.push_back(rl);
                        }
                    }
                }
            }
        }
        
        // Extended supported opcodes for int mode (including range loop opcodes)
        static const std::unordered_set<uint8_t> supported_int_opcodes = {
            op::RESUME, op::LOAD_FAST, op::LOAD_FAST_LOAD_FAST, op::LOAD_CONST,
            op::STORE_FAST, op::BINARY_OP, op::UNARY_NEGATIVE, op::COMPARE_OP,
            op::POP_JUMP_IF_FALSE, op::POP_JUMP_IF_TRUE, op::RETURN_VALUE, op::RETURN_CONST,
            op::POP_TOP, op::JUMP_BACKWARD, op::JUMP_FORWARD, op::COPY,
            op::NOP, op::CACHE,
            // Range loop opcodes (only valid within detected range patterns)
            op::PUSH_NULL, op::LOAD_GLOBAL, op::CALL, op::GET_ITER, op::FOR_ITER, op::END_FOR
        };
        
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            bool is_supported = supported_int_opcodes.find(instr.opcode) != supported_int_opcodes.end();
            
            // For range-related opcodes, check if they're part of a detected range pattern
            if (is_supported && (instr.opcode == op::PUSH_NULL || instr.opcode == op::LOAD_GLOBAL ||
                instr.opcode == op::CALL || instr.opcode == op::GET_ITER || 
                instr.opcode == op::FOR_ITER || instr.opcode == op::END_FOR))
            {
                if (range_loop_offsets.find(instr.offset) == range_loop_offsets.end())
                {
                    // These opcodes are not part of a range pattern - unsupported
                    llvm::errs() << "Integer mode: opcode " << static_cast<int>(instr.opcode) 
                                 << " at offset " << instr.offset << " is not part of a range() pattern. Use mode='generic' or mode='auto'.\n";
                    return false;
                }
            }
            else if (!is_supported)
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
                
                // Check if this JUMP_BACKWARD targets a range loop FOR_ITER
                // If so, skip creating a block (we create range_header_X instead)
                bool is_range_target = false;
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.for_iter_idx < static_cast<int>(instructions.size()) &&
                        instructions[rl.for_iter_idx].offset == target_offset)
                    {
                        is_range_target = true;
                        break;
                    }
                }
                
                // Only create block for non-range while loops
                if (!is_range_target && !jump_targets.count(target_offset))
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
                    case 0:  // ADD
                    case 13: // INPLACE_ADD (+=)
                        result = builder.CreateAdd(first, second, "add");
                        break;
                    case 10: // SUB
                    case 23: // INPLACE_SUB (-=)
                        result = builder.CreateSub(first, second, "sub");
                        break;
                    case 5:  // MUL
                    case 18: // INPLACE_MUL (*=)
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
                    
                    // Special case: if next instruction is JUMP_BACKWARD, branch directly to its target
                    // This avoids creating a spurious intermediate block
                    bool next_is_jump_backward = (instructions[i + 1].opcode == op::JUMP_BACKWARD);
                    if (next_is_jump_backward)
                    {
                        next_offset = instructions[i + 1].argval; // Use JUMP_BACKWARD's target directly
                    }

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
                    
                    // If we consumed the JUMP_BACKWARD, skip it
                    if (next_is_jump_backward)
                    {
                        i++; // Skip the JUMP_BACKWARD since we already handled it
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
                
                // Check if this JUMP_BACKWARD is for a range loop
                // If so, increment the range counter before jumping
                bool is_range_loop = false;
                int range_for_iter_idx = -1;
                
                for (const auto& rl : detected_range_loops)
                {
                    // If the for_iter_idx's offset matches the target, this is a range loop
                    if (rl.for_iter_idx < static_cast<int>(instructions.size()) &&
                        instructions[rl.for_iter_idx].offset == target_offset)
                    {
                        is_range_loop = true;
                        range_for_iter_idx = rl.for_iter_idx;
                        break;
                    }
                }
                
                if (is_range_loop && range_for_iter_idx >= 0)
                {
                    // This is a range loop - increment counter and jump to our native header
                    int loop_counter_idx = 10000 + range_for_iter_idx;
                    
                    if (local_allocas.count(loop_counter_idx))
                    {
                        // Increment the loop counter
                        llvm::Value* counter_val = builder.CreateLoad(i64_type, local_allocas[loop_counter_idx], "counter_inc");
                        llvm::Value* next_val = builder.CreateAdd(counter_val, llvm::ConstantInt::get(i64_type, 1), "counter_next");
                        builder.CreateStore(next_val, local_allocas[loop_counter_idx]);
                        
                        // Jump to our native range_header block
                        if (!builder.GetInsertBlock()->getTerminator())
                        {
                            llvm::BasicBlock* loop_header = nullptr;
                            std::string expected_name = "range_header_" + std::to_string(range_for_iter_idx);
                            
                            for (auto& BB : *func)
                            {
                                if (BB.getName().starts_with(expected_name))
                                {
                                    loop_header = &BB;
                                    break;
                                }
                            }
                            if (loop_header)
                            {
                                builder.CreateBr(loop_header);
                            }
                        }
                    }
                    // Note: No after_loop block needed for range loops
                    // Control flow goes: loop_body -> increment -> range_header -> (body or exit)
                }
                else
                {
                    // Normal while loop - just jump
                    if (!jump_targets.count(target_offset))
                    {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "loop_header_" + std::to_string(target_offset), func);
                    }
                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        builder.CreateBr(jump_targets[target_offset]);
                    }
                    
                    // Create a new block for any code after the while loop (unreachable but needed for CFG)
                    llvm::BasicBlock *after_loop = llvm::BasicBlock::Create(
                        *local_context, "after_loop_" + std::to_string(i), func);
                    builder.SetInsertPoint(after_loop);
                }
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
            // ========== Native Range Loop Opcodes ==========
            // These opcodes are part of detected range() patterns and generate native LLVM loops
            else if (instr.opcode == op::PUSH_NULL || instr.opcode == op::LOAD_GLOBAL)
            {
                // Skip - these are part of range() call setup
                // The range() call is handled specially in FOR_ITER
                continue;
            }
            else if (instr.opcode == op::CALL)
            {
                // Skip - range() call is handled in FOR_ITER
                // Pop the arguments and callable from the conceptual stack
                // (they were never actually pushed in native mode)
                continue;
            }
            else if (instr.opcode == op::GET_ITER)
            {
                // Skip - iterator creation is handled in FOR_ITER
                continue;
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                // Native range loop implementation
                // Find the corresponding RangeLoop info we detected earlier
                const RangeLoop* rl_ptr = nullptr;
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.for_iter_idx == static_cast<int>(i))
                    {
                        rl_ptr = &rl;
                        break;
                    }
                }
                
                if (!rl_ptr)
                {
                    llvm::errs() << "Integer mode: FOR_ITER at " << i << " not in detected range loops\n";
                    return false;
                }
                
                // Get loop bounds
                llvm::Value* start_val = nullptr;
                llvm::Value* stop_val = nullptr;
                
                if (rl_ptr->start_const_idx == -2)
                {
                    // Literal 0 for range(stop)
                    start_val = llvm::ConstantInt::get(i64_type, 0);
                }
                else if (rl_ptr->start_const_idx >= 0 && static_cast<size_t>(rl_ptr->start_const_idx) < int_constants.size())
                {
                    start_val = llvm::ConstantInt::get(i64_type, int_constants[rl_ptr->start_const_idx]);
                }
                else if (rl_ptr->start_local_idx >= 0 && local_allocas.count(rl_ptr->start_local_idx))
                {
                    start_val = builder.CreateLoad(i64_type, local_allocas[rl_ptr->start_local_idx], "range_start");
                }
                else
                {
                    start_val = llvm::ConstantInt::get(i64_type, 0);
                }
                
                if (rl_ptr->stop_const_idx >= 0 && static_cast<size_t>(rl_ptr->stop_const_idx) < int_constants.size())
                {
                    stop_val = llvm::ConstantInt::get(i64_type, int_constants[rl_ptr->stop_const_idx]);
                }
                else if (rl_ptr->stop_local_idx >= 0 && local_allocas.count(rl_ptr->stop_local_idx))
                {
                    stop_val = builder.CreateLoad(i64_type, local_allocas[rl_ptr->stop_local_idx], "range_stop");
                }
                else
                {
                    llvm::errs() << "Integer mode: Cannot determine range stop value\n";
                    return false;
                }
                
                // Create loop structure:
                // entry -> loop_header -> (loop_body or loop_exit)
                // loop_body -> ... -> loop_latch -> loop_header
                
                int for_iter_target = instr.argval; // END_FOR offset
                
                // Save current insert point
                llvm::BasicBlock* current_block = builder.GetInsertBlock();
                llvm::BasicBlock::iterator current_point = builder.GetInsertPoint();
                
                // Create allocas at the entry block (after existing allocas)
                builder.SetInsertPoint(&func->getEntryBlock(), func->getEntryBlock().getFirstInsertionPt());
                
                llvm::AllocaInst* loop_counter = builder.CreateAlloca(
                    i64_type, nullptr, "range_counter_" + std::to_string(i));
                llvm::AllocaInst* stop_alloca = builder.CreateAlloca(
                    i64_type, nullptr, "range_stop_" + std::to_string(i));
                
                // Restore insert point
                builder.SetInsertPoint(current_block, current_point);
                
                llvm::BasicBlock* loop_header = llvm::BasicBlock::Create(
                    *local_context, "range_header_" + std::to_string(i), func);
                llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(
                    *local_context, "range_body_" + std::to_string(i), func);
                llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(
                    *local_context, "range_exit_" + std::to_string(i), func);
                
                // Register the exit block for END_FOR target
                jump_targets[for_iter_target] = loop_exit;
                
                // Initialize counter to start value and store stop value
                builder.CreateStore(start_val, loop_counter);
                builder.CreateStore(stop_val, stop_alloca);
                builder.CreateBr(loop_header);
                
                // Loop header: check if counter < stop
                builder.SetInsertPoint(loop_header);
                llvm::Value* header_counter = builder.CreateLoad(i64_type, loop_counter, "counter");
                llvm::Value* header_stop = builder.CreateLoad(i64_type, stop_alloca, "stop_val");
                llvm::Value* cmp = builder.CreateICmpSLT(header_counter, header_stop, "range_cond");
                builder.CreateCondBr(cmp, loop_body, loop_exit);
                
                // Loop body: load current counter value and push to stack
                builder.SetInsertPoint(loop_body);
                llvm::Value* body_counter = builder.CreateLoad(i64_type, loop_counter, "loop_var");
                stack.push_back(body_counter);
                
                // Store the loop counter alloca for END_FOR to increment
                // We use a map from FOR_ITER index to the alloca
                // For simplicity, store in local_allocas with a special index
                int loop_counter_idx = 10000 + static_cast<int>(i); // Use high index to avoid conflicts
                local_allocas[loop_counter_idx] = loop_counter;
                
                // Mark which instruction index has the loop exit target
                // so JUMP_BACKWARD can find the right increment
                continue; // Let the loop body execute naturally
            }
            else if (instr.opcode == op::END_FOR)
            {
                // END_FOR is the exit point of the loop - reached when iterator is exhausted
                // The loop back happens via JUMP_BACKWARD, not END_FOR
                // We just need to switch to the loop exit block which was already created by FOR_ITER
                
                // Find the corresponding FOR_ITER to get the exit block
                int for_iter_idx = -1;
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.end_for_idx == static_cast<int>(i))
                    {
                        for_iter_idx = rl.for_iter_idx;
                        break;
                    }
                }
                
                if (for_iter_idx >= 0)
                {
                    // Find and switch to the loop exit block
                    llvm::BasicBlock* loop_exit = nullptr;
                    std::string exit_name = "range_exit_" + std::to_string(for_iter_idx);
                    for (auto& BB : *func)
                    {
                        if (BB.getName().starts_with(exit_name))
                        {
                            loop_exit = &BB;
                            break;
                        }
                    }
                    if (loop_exit)
                    {
                        // Just set the insert point to the exit block
                        // The FOR_ITER condBr already branches here when counter >= stop
                        builder.SetInsertPoint(loop_exit);
                    }
                }
                // Continue to next instruction - code generation will resume in the exit block
            }
        }

        // Ensure function has a return
        if (!builder.GetInsertBlock()->getTerminator())
        {
            builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
        }
        // Capture IR if dump_ir is enabled
        if (dump_ir)
        {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
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
    // Float Mode Compilation
    // =========================================================================
    // Compiles a function that uses only native f64 (double) types.
    // Parameters and return value are all double. No Python object overhead.
    // =========================================================================
    bool JITCore::compile_float_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
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

        // Extract float constants
        std::vector<double> float_constants;
        for (size_t i = 0; i < py_constants.size(); ++i)
        {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::float_>(const_obj))
            {
                float_constants.push_back(nb::cast<double>(const_obj));
            }
            else if (nb::isinstance<nb::int_>(const_obj))
            {
                // Allow int constants in float mode (promote to double)
                float_constants.push_back(static_cast<double>(nb::cast<int64_t>(const_obj)));
            }
            else
            {
                float_constants.push_back(0.0); // Non-numeric constants default to 0.0
            }
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        llvm::Type *f64_type = llvm::Type::getDoubleTy(*local_context);

        // Create function type - all double for float mode
        std::vector<llvm::Type *> param_types(param_count, f64_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(
            f64_type, // Return double
            param_types,
            false);

        llvm::Function *func = llvm::Function::Create(
            func_type,
            llvm::Function::ExternalLinkage,
            name,
            module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        // Create stack for values
        std::vector<llvm::Value *> stack;

        // Create allocas for local variables (all double)
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
        {
            local_allocas[i] = builder.CreateAlloca(f64_type, nullptr, "local_" + std::to_string(i));
        }

        // Store parameters in local variables
        int arg_idx = 0;
        for (auto &arg : func->args())
        {
            builder.CreateStore(&arg, local_allocas[arg_idx]);
            ++arg_idx;
        }

        // =========================================================================
        // Range Loop Detection for Float Mode
        // Detect pattern: PUSH_NULL, LOAD_GLOBAL(range), args, CALL, GET_ITER, FOR_ITER
        // =========================================================================
        struct RangeLoop {
            int for_iter_idx;
            int end_for_idx;
            int arg_count;
        };
        std::vector<RangeLoop> detected_range_loops;
        std::unordered_set<int> range_loop_offsets;
        
        for (size_t i = 0; i + 1 < instructions.size(); ++i)
        {
            if (instructions[i].opcode == op::GET_ITER &&
                instructions[i + 1].opcode == op::FOR_ITER)
            {
                size_t call_idx = i - 1;
                
                if (call_idx < instructions.size() && instructions[call_idx].opcode == op::CALL)
                {
                    int arg_count = instructions[call_idx].arg;
                    
                    size_t first_arg_idx = call_idx - arg_count;
                    size_t load_global_idx = first_arg_idx - 1;
                    
                    bool has_separate_push_null = false;
                    size_t push_null_idx = load_global_idx;
                    
                    if (load_global_idx < instructions.size() &&
                        instructions[load_global_idx].opcode == op::LOAD_GLOBAL)
                    {
                        bool combined_null = (instructions[load_global_idx].arg & 1) == 1;
                        
                        if (!combined_null && load_global_idx > 0)
                        {
                            push_null_idx = load_global_idx - 1;
                            if (push_null_idx < instructions.size() &&
                                instructions[push_null_idx].opcode == op::PUSH_NULL)
                            {
                                has_separate_push_null = true;
                            }
                        }
                        
                        if (combined_null || has_separate_push_null)
                        {
                            // Mark offsets as valid
                            if (has_separate_push_null)
                            {
                                range_loop_offsets.insert(instructions[push_null_idx].offset);
                            }
                            range_loop_offsets.insert(instructions[load_global_idx].offset);
                            for (size_t j = first_arg_idx; j <= call_idx; ++j)
                            {
                                range_loop_offsets.insert(instructions[j].offset);
                            }
                            range_loop_offsets.insert(instructions[i].offset);     // GET_ITER
                            range_loop_offsets.insert(instructions[i + 1].offset); // FOR_ITER
                            
                            // Find END_FOR
                            int for_iter_target = instructions[i + 1].argval;
                            for (size_t j = i + 2; j < instructions.size(); ++j)
                            {
                                if (instructions[j].opcode == op::END_FOR &&
                                    instructions[j].offset >= for_iter_target - 4)
                                {
                                    range_loop_offsets.insert(instructions[j].offset);
                                    
                                    RangeLoop rl;
                                    rl.for_iter_idx = static_cast<int>(i + 1);
                                    rl.end_for_idx = static_cast<int>(j);
                                    rl.arg_count = arg_count;
                                    detected_range_loops.push_back(rl);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Jump targets for control flow
        std::unordered_map<int, llvm::BasicBlock *> jump_targets;
        jump_targets[0] = entry;

        // First pass: Create basic blocks for jump targets
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
                
                // Skip creating block for range loop targets (we create range_header_X instead)
                bool is_range_target = false;
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.for_iter_idx < static_cast<int>(instructions.size()) &&
                        instructions[rl.for_iter_idx].offset == target_offset)
                    {
                        is_range_target = true;
                        break;
                    }
                }
                
                if (!is_range_target && !jump_targets.count(target_offset))
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

        // Supported opcodes for float mode
        std::unordered_set<uint16_t> supported_float_opcodes = {
            op::RESUME, op::LOAD_FAST, op::LOAD_FAST_LOAD_FAST, op::LOAD_CONST,
            op::STORE_FAST, op::BINARY_OP, op::UNARY_NEGATIVE, op::COMPARE_OP,
            op::POP_JUMP_IF_FALSE, op::POP_JUMP_IF_TRUE, op::RETURN_VALUE, op::RETURN_CONST,
            op::POP_TOP, op::JUMP_BACKWARD, op::JUMP_FORWARD, op::COPY,
            op::NOP, op::CACHE,
            // Range loop opcodes (only valid within detected range patterns)
            op::PUSH_NULL, op::LOAD_GLOBAL, op::CALL, op::GET_ITER, op::FOR_ITER, op::END_FOR
        };

        // Validate all opcodes are supported
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            bool is_supported = supported_float_opcodes.find(instr.opcode) != supported_float_opcodes.end();
            
            // For range-related opcodes, check if they're part of a detected range pattern
            if (is_supported && (instr.opcode == op::PUSH_NULL || instr.opcode == op::LOAD_GLOBAL ||
                instr.opcode == op::CALL || instr.opcode == op::GET_ITER || 
                instr.opcode == op::FOR_ITER || instr.opcode == op::END_FOR))
            {
                if (range_loop_offsets.find(instr.offset) == range_loop_offsets.end())
                {
                    llvm::errs() << "Float mode: opcode " << static_cast<int>(instr.opcode) 
                                 << " at offset " << instr.offset << " is not part of a range() pattern. Use mode='auto' or mode='object'.\n";
                    return false;
                }
            }
            else if (!is_supported)
            {
                llvm::errs() << "Float mode: unsupported opcode " << static_cast<int>(instr.opcode)
                             << " at offset " << instr.offset << ". Use mode='auto' or mode='object'.\n";
                return false;
            }
        }

        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];

            // Check if we need to insert at a jump target block
            if (jump_targets.count(instr.offset) && jump_targets[instr.offset] != entry)
            {
                llvm::BasicBlock *target_block = jump_targets[instr.offset];
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(target_block);
                }
                builder.SetInsertPoint(target_block);
            }

            if (instr.opcode == op::RESUME)
            {
                continue;
            }
            else if (instr.opcode == op::LOAD_FAST)
            {
                if (local_allocas.count(instr.arg))
                {
                    llvm::Value *loaded = builder.CreateLoad(f64_type, local_allocas[instr.arg], "load_" + std::to_string(instr.arg));
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                if (local_allocas.count(first_local))
                {
                    stack.push_back(builder.CreateLoad(f64_type, local_allocas[first_local], "load_" + std::to_string(first_local)));
                }
                if (local_allocas.count(second_local))
                {
                    stack.push_back(builder.CreateLoad(f64_type, local_allocas[second_local], "load_" + std::to_string(second_local)));
                }
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                if (instr.arg < float_constants.size())
                {
                    llvm::Value *const_val = llvm::ConstantFP::get(f64_type, float_constants[instr.arg]);
                    stack.push_back(const_val);
                }
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                if (!stack.empty() && local_allocas.count(instr.arg))
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    builder.CreateStore(val, local_allocas[instr.arg]);
                }
            }
            else if (instr.opcode == op::BINARY_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *result = nullptr;

                    switch (instr.arg)
                    {
                    case 0: // ADD
                        result = builder.CreateFAdd(lhs, rhs, "fadd");
                        break;
                    case 10: // SUBTRACT
                        result = builder.CreateFSub(lhs, rhs, "fsub");
                        break;
                    case 5: // MULTIPLY
                        result = builder.CreateFMul(lhs, rhs, "fmul");
                        break;
                    case 11: // TRUE_DIVIDE
                        result = builder.CreateFDiv(lhs, rhs, "fdiv");
                        break;
                    case 2: // FLOOR_DIVIDE
                    {
                        llvm::Value *div_result = builder.CreateFDiv(lhs, rhs, "fdiv_floor");
                        // Call floor intrinsic
                        llvm::Function *floor_fn = LLVM_GET_INTRINSIC_DECLARATION(module.get(), llvm::Intrinsic::floor, {f64_type});
                        result = builder.CreateCall(floor_fn, {div_result}, "floor");
                        break;
                    }
                    case 6: // REMAINDER
                        result = builder.CreateFRem(lhs, rhs, "fmod");
                        break;
                    case 8: // POWER
                    {
                        llvm::Function *pow_fn = LLVM_GET_INTRINSIC_DECLARATION(module.get(), llvm::Intrinsic::pow, {f64_type});
                        result = builder.CreateCall(pow_fn, {lhs, rhs}, "pow");
                        break;
                    }
                    default:
                        result = lhs; // Fallback
                    }

                    if (result)
                    {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_NEGATIVE)
            {
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back(); stack.pop_back();
                    stack.push_back(builder.CreateFNeg(val, "neg"));
                }
            }
            else if (instr.opcode == op::COMPARE_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *cmp_result = nullptr;

                    // Compare op mapping for float (Python 3.13 encoding: arg >> 5)
                    switch (instr.arg >> 5)
                    {
                    case 0: // LT
                        cmp_result = builder.CreateFCmpOLT(lhs, rhs, "flt");
                        break;
                    case 1: // LE
                        cmp_result = builder.CreateFCmpOLE(lhs, rhs, "fle");
                        break;
                    case 2: // EQ
                        cmp_result = builder.CreateFCmpOEQ(lhs, rhs, "feq");
                        break;
                    case 3: // NE
                        cmp_result = builder.CreateFCmpONE(lhs, rhs, "fne");
                        break;
                    case 4: // GT
                        cmp_result = builder.CreateFCmpOGT(lhs, rhs, "fgt");
                        break;
                    case 5: // GE
                        cmp_result = builder.CreateFCmpOGE(lhs, rhs, "fge");
                        break;
                    default:
                        cmp_result = builder.CreateFCmpOLT(lhs, rhs, "fcmp");
                    }

                    // Convert i1 to f64 (0.0 or 1.0) for stack consistency
                    llvm::Value *f64_result = builder.CreateUIToFP(cmp_result, f64_type, "cmp_f64");
                    stack.push_back(f64_result);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back(); stack.pop_back();
                    // Convert f64 to bool (non-zero is true)
                    llvm::Value *is_true = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(f64_type, 0.0), "is_true");

                    int target_offset = instr.argval;
                    llvm::BasicBlock *true_block = llvm::BasicBlock::Create(*local_context, "true_" + std::to_string(i), func);
                    llvm::BasicBlock *false_block = jump_targets.count(target_offset) ? jump_targets[target_offset] : entry;

                    builder.CreateCondBr(is_true, true_block, false_block);
                    builder.SetInsertPoint(true_block);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back(); stack.pop_back();
                    llvm::Value *is_true = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(f64_type, 0.0), "is_true");

                    int target_offset = instr.argval;
                    llvm::BasicBlock *false_block = llvm::BasicBlock::Create(*local_context, "false_" + std::to_string(i), func);
                    llvm::BasicBlock *true_block = jump_targets.count(target_offset) ? jump_targets[target_offset] : entry;

                    builder.CreateCondBr(is_true, true_block, false_block);
                    builder.SetInsertPoint(false_block);
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD)
            {
                int target_offset = instr.argval;
                
                // Check if this JUMP_BACKWARD is for a range loop
                bool is_range_loop = false;
                int range_for_iter_idx = -1;
                
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.for_iter_idx < static_cast<int>(instructions.size()) &&
                        instructions[rl.for_iter_idx].offset == target_offset)
                    {
                        is_range_loop = true;
                        range_for_iter_idx = rl.for_iter_idx;
                        break;
                    }
                }
                
                if (is_range_loop && range_for_iter_idx >= 0)
                {
                    // Range loop - increment counter and jump to our native header
                    int loop_counter_idx = 10000 + range_for_iter_idx;
                    
                    if (local_allocas.count(loop_counter_idx))
                    {
                        // Increment the loop counter (as f64)
                        llvm::Value* counter_val = builder.CreateLoad(f64_type, local_allocas[loop_counter_idx], "counter_inc");
                        llvm::Value* next_val = builder.CreateFAdd(counter_val, llvm::ConstantFP::get(f64_type, 1.0), "counter_next");
                        builder.CreateStore(next_val, local_allocas[loop_counter_idx]);
                        
                        // Jump to native range_header block
                        if (!builder.GetInsertBlock()->getTerminator())
                        {
                            llvm::BasicBlock* loop_header = nullptr;
                            std::string expected_name = "range_header_" + std::to_string(range_for_iter_idx);
                            
                            for (auto& BB : *func)
                            {
                                if (BB.getName().starts_with(expected_name))
                                {
                                    loop_header = &BB;
                                    break;
                                }
                            }
                            if (loop_header)
                            {
                                builder.CreateBr(loop_header);
                            }
                        }
                    }
                }
                else
                {
                    // Normal while loop
                    if (!jump_targets.count(target_offset))
                    {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "loop_header_" + std::to_string(target_offset), func);
                    }
                    if (!builder.GetInsertBlock()->getTerminator())
                    {
                        builder.CreateBr(jump_targets[target_offset]);
                    }
                    
                    llvm::BasicBlock *after_loop = llvm::BasicBlock::Create(
                        *local_context, "after_loop_" + std::to_string(i), func);
                    builder.SetInsertPoint(after_loop);
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
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE)
            {
                if (!stack.empty())
                {
                    llvm::Value *ret = stack.back();
                    builder.CreateRet(ret);
                }
                else
                {
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));
                }
            }
            else if (instr.opcode == op::RETURN_CONST)
            {
                if (instr.arg < float_constants.size())
                {
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, float_constants[instr.arg]));
                }
                else
                {
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));
                }
            }
            else if (instr.opcode == op::POP_TOP)
            {
                if (!stack.empty())
                {
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::COPY)
            {
                if (instr.arg <= stack.size())
                {
                    stack.push_back(stack[stack.size() - instr.arg]);
                }
            }
            // Range loop opcodes - handled natively for performance
            else if (instr.opcode == op::PUSH_NULL || instr.opcode == op::LOAD_GLOBAL ||
                     instr.opcode == op::CALL || instr.opcode == op::GET_ITER)
            {
                // Skip these - they're part of range() setup, handled by FOR_ITER
                continue;
            }
            else if (instr.opcode == op::FOR_ITER)
            {
                // Find the matching range loop info
                int current_idx = static_cast<int>(i);
                RangeLoop* current_rl = nullptr;
                for (auto& rl : detected_range_loops)
                {
                    if (rl.for_iter_idx == current_idx)
                    {
                        current_rl = &rl;
                        break;
                    }
                }
                
                if (!current_rl)
                {
                    llvm::errs() << "Float mode: FOR_ITER without detected range pattern\n";
                    return false;
                }
                
                // Get range arguments (currently only supports range(stop) or range(start, stop))
                // For float mode, we store as f64
                llvm::Value* start_val = llvm::ConstantFP::get(f64_type, 0.0);
                llvm::Value* stop_val = nullptr;
                
                // Get stop value from stack (pushed by LOAD_CONST/LOAD_FAST before CALL)
                if (!stack.empty())
                {
                    stop_val = stack.back();
                    stack.pop_back();
                    
                    if (current_rl->arg_count == 2 && !stack.empty())
                    {
                        start_val = stop_val;
                        stop_val = stack.back();
                        stack.pop_back();
                    }
                }
                else
                {
                    stop_val = llvm::ConstantFP::get(f64_type, 0.0);
                }
                
                int for_iter_target = instr.argval;
                
                // Save insert point and create allocas at entry
                llvm::BasicBlock* current_block = builder.GetInsertBlock();
                llvm::BasicBlock::iterator current_point = builder.GetInsertPoint();
                builder.SetInsertPoint(&func->getEntryBlock(), func->getEntryBlock().getFirstInsertionPt());
                
                llvm::AllocaInst* loop_counter = builder.CreateAlloca(
                    f64_type, nullptr, "range_counter_" + std::to_string(i));
                llvm::AllocaInst* stop_alloca = builder.CreateAlloca(
                    f64_type, nullptr, "range_stop_" + std::to_string(i));
                
                builder.SetInsertPoint(current_block, current_point);
                
                // Create loop blocks
                llvm::BasicBlock* loop_header = llvm::BasicBlock::Create(
                    *local_context, "range_header_" + std::to_string(i), func);
                llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(
                    *local_context, "range_body_" + std::to_string(i), func);
                llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(
                    *local_context, "range_exit_" + std::to_string(i), func);
                
                jump_targets[for_iter_target] = loop_exit;
                
                // Initialize counter and store stop value
                builder.CreateStore(start_val, loop_counter);
                builder.CreateStore(stop_val, stop_alloca);
                builder.CreateBr(loop_header);
                
                // Loop header: check counter < stop
                builder.SetInsertPoint(loop_header);
                llvm::Value* header_counter = builder.CreateLoad(f64_type, loop_counter, "counter");
                llvm::Value* header_stop = builder.CreateLoad(f64_type, stop_alloca, "stop_val");
                llvm::Value* cmp = builder.CreateFCmpOLT(header_counter, header_stop, "range_cond");
                builder.CreateCondBr(cmp, loop_body, loop_exit);
                
                // Loop body: load counter and push to stack
                builder.SetInsertPoint(loop_body);
                llvm::Value* body_counter = builder.CreateLoad(f64_type, loop_counter, "loop_var");
                stack.push_back(body_counter);
                
                // Store loop counter alloca for JUMP_BACKWARD increment
                int loop_counter_idx = 10000 + current_idx;
                local_allocas[loop_counter_idx] = loop_counter;
            }
            else if (instr.opcode == op::END_FOR)
            {
                // Find corresponding FOR_ITER to get exit block
                int for_iter_idx = -1;
                for (const auto& rl : detected_range_loops)
                {
                    if (rl.end_for_idx == static_cast<int>(i))
                    {
                        for_iter_idx = rl.for_iter_idx;
                        break;
                    }
                }
                
                if (for_iter_idx >= 0)
                {
                    // Switch to loop exit block
                    llvm::BasicBlock* loop_exit = nullptr;
                    std::string exit_name = "range_exit_" + std::to_string(for_iter_idx);
                    for (auto& BB : *func)
                    {
                        if (BB.getName().starts_with(exit_name))
                        {
                            loop_exit = &BB;
                            break;
                        }
                    }
                    if (loop_exit)
                    {
                        builder.SetInsertPoint(loop_exit);
                    }
                }
            }
        }

        // Ensure function has a return
        if (!builder.GetInsertBlock()->getTerminator())
        {
            builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));
        }

        // Capture IR if dump_ir is enabled
        if (dump_ir)
        {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
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
    // Bool Mode Compilation
    // =========================================================================
    // Compiles a function that uses only native boolean types.
    // Parameters and return value are all i64 (0 = false, 1 = true).
    // =========================================================================
    bool JITCore::compile_bool_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit)
        {
            return false;
        }

        // Check if already compiled to prevent duplicate symbol errors
        if (compiled_functions.count(name) > 0)
        {
            return true;
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

        // Extract bool constants (convert to 0/1)
        std::vector<int64_t> bool_constants;
        for (size_t i = 0; i < py_constants.size(); ++i)
        {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::bool_>(const_obj))
            {
                bool_constants.push_back(nb::cast<bool>(const_obj) ? 1 : 0);
            }
            else if (nb::isinstance<nb::int_>(const_obj))
            {
                bool_constants.push_back(nb::cast<int64_t>(const_obj) != 0 ? 1 : 0);
            }
            else
            {
                bool_constants.push_back(0);
            }
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);

        // Create function type - all i64 for bool mode (0 = false, 1 = true)
        std::vector<llvm::Type *> param_types(param_count, i64_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(
            i64_type, // Return i64 (0 or 1)
            param_types,
            false);

        llvm::Function *func = llvm::Function::Create(
            func_type,
            llvm::Function::ExternalLinkage,
            name,
            module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        // Create stack for values
        std::vector<llvm::Value *> stack;

        // Create allocas for local variables (all i64)
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
        {
            local_allocas[i] = builder.CreateAlloca(i64_type, nullptr, "local_" + std::to_string(i));
        }

        // Store function parameters
        int arg_idx = 0;
        for (auto &arg : func->args())
        {
            arg.setName("param_" + std::to_string(arg_idx));
            if (arg_idx < total_locals)
            {
                builder.CreateStore(&arg, local_allocas[arg_idx]);
            }
            ++arg_idx;
        }

        // Jump targets for control flow
        std::unordered_map<int, llvm::BasicBlock *> jump_targets;
        jump_targets[0] = entry;
        
        // Track stack values for PHI nodes at merge points
        // Maps target block offset -> vector of (incoming value, incoming block)
        std::unordered_map<int, std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>>> block_incoming_values;

        // First pass: Create basic blocks for jump targets
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

        // Supported opcodes for bool mode
        std::unordered_set<uint16_t> supported_bool_opcodes = {
            op::RESUME, op::LOAD_FAST, op::LOAD_FAST_LOAD_FAST, op::LOAD_CONST,
            op::STORE_FAST, op::COMPARE_OP, op::UNARY_NOT, op::TO_BOOL,
            op::POP_JUMP_IF_FALSE, op::POP_JUMP_IF_TRUE, op::RETURN_VALUE, op::RETURN_CONST,
            op::POP_TOP, op::JUMP_BACKWARD, op::JUMP_FORWARD, op::COPY,
            op::NOP, op::CACHE
        };

        // Validate all opcodes are supported
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            if (supported_bool_opcodes.find(instr.opcode) == supported_bool_opcodes.end())
            {
                llvm::errs() << "Bool mode: unsupported opcode " << static_cast<int>(instr.opcode)
                             << " at offset " << instr.offset << ". Use mode='auto' or mode='object'.\n";
                return false;
            }
        }

        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];

            // Check if we need to insert at a jump target block
            if (jump_targets.count(instr.offset) && jump_targets[instr.offset] != entry)
            {
                llvm::BasicBlock *target_block = jump_targets[instr.offset];
                
                // Record incoming value from fallthrough path if stack not empty
                if (!stack.empty() && !builder.GetInsertBlock()->getTerminator())
                {
                    block_incoming_values[instr.offset].push_back({stack.back(), builder.GetInsertBlock()});
                }
                
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(target_block);
                }
                builder.SetInsertPoint(target_block);
                
                // Create PHI node only if we have 2+ incoming values (actual merge point)
                if (block_incoming_values.count(instr.offset) && 
                    block_incoming_values[instr.offset].size() >= 2)
                {
                    auto& incoming = block_incoming_values[instr.offset];
                    
                    // Create PHI for merge
                    llvm::PHINode *phi = builder.CreatePHI(i64_type, incoming.size(), "merge_phi");
                    for (auto& [val, block] : incoming)
                    {
                        phi->addIncoming(val, block);
                    }
                    
                    // Replace stack top with PHI (or push if stack was empty)
                    if (!stack.empty())
                    {
                        stack.back() = phi;
                    }
                    else
                    {
                        stack.push_back(phi);
                    }
                    
                    // Clear incoming values (only use once)
                    block_incoming_values[instr.offset].clear();
                }
            }

            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE || instr.opcode == op::TO_BOOL)
            {
                // No-op - in bool mode, TO_BOOL is a no-op since values are already 0/1
            }
            else if (instr.opcode == op::LOAD_FAST)
            {
                if (local_allocas.count(instr.arg))
                {
                    llvm::Value *val = builder.CreateLoad(i64_type, local_allocas[instr.arg], "load_" + std::to_string(instr.arg));
                    stack.push_back(val);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                {
                    llvm::Value *val1 = builder.CreateLoad(i64_type, local_allocas[idx1], "load_" + std::to_string(idx1));
                    stack.push_back(val1);
                }
                if (local_allocas.count(idx2))
                {
                    llvm::Value *val2 = builder.CreateLoad(i64_type, local_allocas[idx2], "load_" + std::to_string(idx2));
                    stack.push_back(val2);
                }
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                if (instr.arg < bool_constants.size())
                {
                    llvm::Value *val = llvm::ConstantInt::get(i64_type, bool_constants[instr.arg]);
                    stack.push_back(val);
                }
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                if (!stack.empty() && local_allocas.count(instr.arg))
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    builder.CreateStore(val, local_allocas[instr.arg]);
                }
            }
            else if (instr.opcode == op::UNARY_NOT)
            {
                // Logical NOT: 0 -> 1, non-zero -> 0
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back(); stack.pop_back();
                    llvm::Value *is_zero = builder.CreateICmpEQ(val, llvm::ConstantInt::get(i64_type, 0), "is_zero");
                    llvm::Value *result = builder.CreateZExt(is_zero, i64_type, "not_result");
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::COMPARE_OP)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *cmp_result = nullptr;

                    // Compare op mapping (Python 3.13 encoding: arg >> 5)
                    switch (instr.arg >> 5)
                    {
                    case 0: // LT
                        cmp_result = builder.CreateICmpSLT(lhs, rhs, "lt");
                        break;
                    case 1: // LE
                        cmp_result = builder.CreateICmpSLE(lhs, rhs, "le");
                        break;
                    case 2: // EQ
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                        break;
                    case 3: // NE
                        cmp_result = builder.CreateICmpNE(lhs, rhs, "ne");
                        break;
                    case 4: // GT
                        cmp_result = builder.CreateICmpSGT(lhs, rhs, "gt");
                        break;
                    case 5: // GE
                        cmp_result = builder.CreateICmpSGE(lhs, rhs, "ge");
                        break;
                    default:
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "cmp");
                    }

                    // Convert i1 to i64 (0 or 1) for stack
                    llvm::Value *i64_result = builder.CreateZExt(cmp_result, i64_type, "cmp_i64");
                    stack.push_back(i64_result);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back(); stack.pop_back();
                    llvm::Value *is_true = builder.CreateICmpNE(cond, llvm::ConstantInt::get(i64_type, 0), "is_true");

                    int target_offset = instr.argval;
                    llvm::BasicBlock *true_block = llvm::BasicBlock::Create(*local_context, "true_" + std::to_string(i), func);
                    llvm::BasicBlock *false_block = jump_targets.count(target_offset) ? jump_targets[target_offset] : entry;

                    builder.CreateCondBr(is_true, true_block, false_block);
                    builder.SetInsertPoint(true_block);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_TRUE)
            {
                if (!stack.empty())
                {
                    llvm::Value *cond = stack.back(); stack.pop_back();
                    llvm::Value *is_true = builder.CreateICmpNE(cond, llvm::ConstantInt::get(i64_type, 0), "is_true");

                    int target_offset = instr.argval;
                    llvm::BasicBlock *false_block = llvm::BasicBlock::Create(*local_context, "false_" + std::to_string(i), func);
                    llvm::BasicBlock *true_block = jump_targets.count(target_offset) ? jump_targets[target_offset] : entry;

                    // For short-circuit or: if stack not empty, record value for PHI at target
                    if (!stack.empty())
                    {
                        llvm::Value *stack_top = stack.back();
                        block_incoming_values[target_offset].push_back({stack_top, builder.GetInsertBlock()});
                    }

                    builder.CreateCondBr(is_true, true_block, false_block);
                    builder.SetInsertPoint(false_block);
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
                if (!builder.GetInsertBlock()->getTerminator())
                {
                    builder.CreateBr(jump_targets[target_offset]);
                }

                llvm::BasicBlock *after_loop = llvm::BasicBlock::Create(
                    *local_context, "after_loop_" + std::to_string(i), func);
                builder.SetInsertPoint(after_loop);
            }
            else if (instr.opcode == op::JUMP_FORWARD)
            {
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
            }
            else if (instr.opcode == op::RETURN_VALUE)
            {
                if (!stack.empty())
                {
                    llvm::Value *ret = stack.back();
                    builder.CreateRet(ret);
                }
                else
                {
                    builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
                }
            }
            else if (instr.opcode == op::RETURN_CONST)
            {
                if (instr.arg < bool_constants.size())
                {
                    builder.CreateRet(llvm::ConstantInt::get(i64_type, bool_constants[instr.arg]));
                }
                else
                {
                    builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
                }
            }
            else if (instr.opcode == op::POP_TOP)
            {
                if (!stack.empty())
                {
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::COPY)
            {
                if (instr.arg <= stack.size())
                {
                    stack.push_back(stack[stack.size() - instr.arg]);
                }
            }
        }

        // Ensure function has a return
        if (!builder.GetInsertBlock()->getTerminator())
        {
            builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
        }

        // Capture IR if dump_ir is enabled
        if (dump_ir)
        {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
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

        // Mark as compiled
        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Int32 Mode Compilation (C Interop)
    // =========================================================================
    bool JITCore::compile_int32_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        std::vector<int32_t> int_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::int_>(const_obj))
                int_constants.push_back(static_cast<int32_t>(nb::cast<int64_t>(const_obj)));
            else
                int_constants.push_back(0);
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        llvm::Type *i32_type = llvm::Type::getInt32Ty(*local_context);

        std::vector<llvm::Type *> param_types(param_count, i32_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(i32_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(i32_type, nullptr, "local_" + std::to_string(i));

        int arg_idx = 0;
        for (auto &arg : func->args()) {
            if (arg_idx < total_locals) builder.CreateStore(&arg, local_allocas[arg_idx]);
            ++arg_idx;
        }

        // Simple code generation for basic arithmetic
        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(i32_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                // Python 3.13: Load two locals at once (arg encodes both indices)
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(i32_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(i32_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < int_constants.size())
                    stack.push_back(llvm::ConstantInt::get(i32_type, int_constants[instr.arg]));
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *result = nullptr;
                    switch (instr.arg) {
                        case 0: result = builder.CreateAdd(lhs, rhs); break;
                        case 10: result = builder.CreateSub(lhs, rhs); break;
                        case 5: result = builder.CreateMul(lhs, rhs); break;
                        case 2: result = builder.CreateSDiv(lhs, rhs); break;
                        default: result = lhs;
                    }
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) builder.CreateRet(stack.back());
                else builder.CreateRet(llvm::ConstantInt::get(i32_type, 0));
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < int_constants.size())
                    builder.CreateRet(llvm::ConstantInt::get(i32_type, int_constants[instr.arg]));
                else
                    builder.CreateRet(llvm::ConstantInt::get(i32_type, 0));
            }
        }

        if (!builder.GetInsertBlock()->getTerminator())
            builder.CreateRet(llvm::ConstantInt::get(i32_type, 0));

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Float32 Mode Compilation (SIMD/ML)
    // =========================================================================
    bool JITCore::compile_float32_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        std::vector<float> float_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::float_>(const_obj))
                float_constants.push_back(static_cast<float>(nb::cast<double>(const_obj)));
            else if (nb::isinstance<nb::int_>(const_obj))
                float_constants.push_back(static_cast<float>(nb::cast<int64_t>(const_obj)));
            else
                float_constants.push_back(0.0f);
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        llvm::Type *f32_type = llvm::Type::getFloatTy(*local_context);

        std::vector<llvm::Type *> param_types(param_count, f32_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(f32_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(f32_type, nullptr, "local_" + std::to_string(i));

        int arg_idx = 0;
        for (auto &arg : func->args()) {
            if (arg_idx < total_locals) builder.CreateStore(&arg, local_allocas[arg_idx]);
            ++arg_idx;
        }

        // Simple code generation for basic arithmetic
        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(f32_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                // Python 3.13: Load two locals at once (arg encodes both indices)
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(f32_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(f32_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < float_constants.size())
                    stack.push_back(llvm::ConstantFP::get(f32_type, float_constants[instr.arg]));
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *result = nullptr;
                    switch (instr.arg) {
                        case 0: result = builder.CreateFAdd(lhs, rhs); break;
                        case 10: result = builder.CreateFSub(lhs, rhs); break;
                        case 5: result = builder.CreateFMul(lhs, rhs); break;
                        case 11: result = builder.CreateFDiv(lhs, rhs); break;
                        default: result = lhs;
                    }
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) builder.CreateRet(stack.back());
                else builder.CreateRet(llvm::ConstantFP::get(f32_type, 0.0f));
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < float_constants.size())
                    builder.CreateRet(llvm::ConstantFP::get(f32_type, float_constants[instr.arg]));
                else
                    builder.CreateRet(llvm::ConstantFP::get(f32_type, 0.0f));
            }
        }

        if (!builder.GetInsertBlock()->getTerminator())
            builder.CreateRet(llvm::ConstantFP::get(f32_type, 0.0f));

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Complex128 Mode Compilation (Scientific Computing)
    // =========================================================================
    bool JITCore::compile_complex128_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse constants - complex numbers stored as (real, imag) pairs
        std::vector<std::pair<double, double>> complex_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            double real_part = 0.0, imag_part = 0.0;
            // Check if it's a complex number
            if (PyComplex_Check(const_obj.ptr())) {
                real_part = PyComplex_RealAsDouble(const_obj.ptr());
                imag_part = PyComplex_ImagAsDouble(const_obj.ptr());
            } else if (nb::isinstance<nb::float_>(const_obj)) {
                real_part = nb::cast<double>(const_obj);
            } else if (nb::isinstance<nb::int_>(const_obj)) {
                real_part = static_cast<double>(nb::cast<int64_t>(const_obj));
            }
            complex_constants.push_back({real_part, imag_part});
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Complex128 type: {double, double} for (real, imag)
        llvm::Type *f64_type = llvm::Type::getDoubleTy(*local_context);
        llvm::StructType *complex_type = llvm::StructType::get(*local_context, {f64_type, f64_type}, false);

        std::vector<llvm::Type *> param_types(param_count, complex_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(complex_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(complex_type, nullptr, "local_" + std::to_string(i));

        int arg_idx = 0;
        for (auto &arg : func->args()) {
            if (arg_idx < total_locals) builder.CreateStore(&arg, local_allocas[arg_idx]);
            ++arg_idx;
        }

        // Helper lambdas for complex operations
        auto extract_real = [&](llvm::Value *c) -> llvm::Value* {
            return builder.CreateExtractValue(c, {0}, "real");
        };
        auto extract_imag = [&](llvm::Value *c) -> llvm::Value* {
            return builder.CreateExtractValue(c, {1}, "imag");
        };
        auto make_complex = [&](llvm::Value *real, llvm::Value *imag) -> llvm::Value* {
            llvm::Value *c = llvm::UndefValue::get(complex_type);
            c = builder.CreateInsertValue(c, real, {0}, "c_real");
            c = builder.CreateInsertValue(c, imag, {1}, "c_imag");
            return c;
        };

        // Code generation
        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < complex_constants.size()) {
                    auto &cc = complex_constants[instr.arg];
                    llvm::Value *real = llvm::ConstantFP::get(f64_type, cc.first);
                    llvm::Value *imag = llvm::ConstantFP::get(f64_type, cc.second);
                    stack.push_back(make_complex(real, imag));
                }
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    
                    llvm::Value *ar = extract_real(lhs);
                    llvm::Value *ai = extract_imag(lhs);
                    llvm::Value *br = extract_real(rhs);
                    llvm::Value *bi = extract_imag(rhs);
                    
                    llvm::Value *result_real = nullptr;
                    llvm::Value *result_imag = nullptr;
                    
                    switch (instr.arg) {
                        case 0: // Add: (ar+br, ai+bi)
                            result_real = builder.CreateFAdd(ar, br, "add_real");
                            result_imag = builder.CreateFAdd(ai, bi, "add_imag");
                            break;
                        case 10: // Sub: (ar-br, ai-bi)
                            result_real = builder.CreateFSub(ar, br, "sub_real");
                            result_imag = builder.CreateFSub(ai, bi, "sub_imag");
                            break;
                        case 5: { // Mul: (ar*br - ai*bi, ar*bi + ai*br)
                            llvm::Value *ar_br = builder.CreateFMul(ar, br);
                            llvm::Value *ai_bi = builder.CreateFMul(ai, bi);
                            llvm::Value *ar_bi = builder.CreateFMul(ar, bi);
                            llvm::Value *ai_br = builder.CreateFMul(ai, br);
                            result_real = builder.CreateFSub(ar_br, ai_bi, "mul_real");
                            result_imag = builder.CreateFAdd(ar_bi, ai_br, "mul_imag");
                            break;
                        }
                        case 11: { // Div: (ar*br + ai*bi, ai*br - ar*bi) / (br*br + bi*bi)
                            llvm::Value *ar_br = builder.CreateFMul(ar, br);
                            llvm::Value *ai_bi = builder.CreateFMul(ai, bi);
                            llvm::Value *ai_br = builder.CreateFMul(ai, br);
                            llvm::Value *ar_bi = builder.CreateFMul(ar, bi);
                            llvm::Value *br_br = builder.CreateFMul(br, br);
                            llvm::Value *bi_bi = builder.CreateFMul(bi, bi);
                            llvm::Value *denom = builder.CreateFAdd(br_br, bi_bi, "denom");
                            llvm::Value *num_real = builder.CreateFAdd(ar_br, ai_bi);
                            llvm::Value *num_imag = builder.CreateFSub(ai_br, ar_bi);
                            result_real = builder.CreateFDiv(num_real, denom, "div_real");
                            result_imag = builder.CreateFDiv(num_imag, denom, "div_imag");
                            break;
                        }
                        default:
                            result_real = ar;
                            result_imag = ai;
                    }
                    stack.push_back(make_complex(result_real, result_imag));
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) {
                    builder.CreateRet(stack.back());
                } else {
                    llvm::Value *zero = llvm::ConstantFP::get(f64_type, 0.0);
                    builder.CreateRet(make_complex(zero, zero));
                }
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < complex_constants.size()) {
                    auto &cc = complex_constants[instr.arg];
                    llvm::Value *real = llvm::ConstantFP::get(f64_type, cc.first);
                    llvm::Value *imag = llvm::ConstantFP::get(f64_type, cc.second);
                    builder.CreateRet(make_complex(real, imag));
                } else {
                    llvm::Value *zero = llvm::ConstantFP::get(f64_type, 0.0);
                    builder.CreateRet(make_complex(zero, zero));
                }
            }
        }

        if (!builder.GetInsertBlock()->getTerminator()) {
            llvm::Value *zero = llvm::ConstantFP::get(f64_type, 0.0);
            builder.CreateRet(make_complex(zero, zero));
        }

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Complex64 Mode Compilation (Single-Precision Complex)
    // =========================================================================
    bool JITCore::compile_complex64_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse constants - complex numbers stored as (real, imag) pairs
        std::vector<std::pair<float, float>> complex_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            float real_part = 0.0f, imag_part = 0.0f;
            if (PyComplex_Check(const_obj.ptr())) {
                real_part = static_cast<float>(PyComplex_RealAsDouble(const_obj.ptr()));
                imag_part = static_cast<float>(PyComplex_ImagAsDouble(const_obj.ptr()));
            } else if (nb::isinstance<nb::float_>(const_obj)) {
                real_part = static_cast<float>(nb::cast<double>(const_obj));
            } else if (nb::isinstance<nb::int_>(const_obj)) {
                real_part = static_cast<float>(nb::cast<int64_t>(const_obj));
            }
            complex_constants.push_back({real_part, imag_part});
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Complex64 type: {float, float} for (real, imag)
        llvm::Type *f32_type = llvm::Type::getFloatTy(*local_context);
        llvm::StructType *complex_type = llvm::StructType::get(*local_context, {f32_type, f32_type}, false);

        std::vector<llvm::Type *> param_types(param_count, complex_type);
        llvm::FunctionType *func_type = llvm::FunctionType::get(complex_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(complex_type, nullptr, "local_" + std::to_string(i));

        int arg_idx = 0;
        for (auto &arg : func->args()) {
            if (arg_idx < total_locals) builder.CreateStore(&arg, local_allocas[arg_idx]);
            ++arg_idx;
        }

        // Helper lambdas for complex operations
        auto extract_real = [&](llvm::Value *c) -> llvm::Value* {
            return builder.CreateExtractValue(c, {0}, "real");
        };
        auto extract_imag = [&](llvm::Value *c) -> llvm::Value* {
            return builder.CreateExtractValue(c, {1}, "imag");
        };
        auto make_complex = [&](llvm::Value *real, llvm::Value *imag) -> llvm::Value* {
            llvm::Value *c = llvm::UndefValue::get(complex_type);
            c = builder.CreateInsertValue(c, real, {0}, "c_real");
            c = builder.CreateInsertValue(c, imag, {1}, "c_imag");
            return c;
        };

        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(complex_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < complex_constants.size()) {
                    auto &cc = complex_constants[instr.arg];
                    llvm::Value *real = llvm::ConstantFP::get(f32_type, cc.first);
                    llvm::Value *imag = llvm::ConstantFP::get(f32_type, cc.second);
                    stack.push_back(make_complex(real, imag));
                }
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    
                    llvm::Value *ar = extract_real(lhs);
                    llvm::Value *ai = extract_imag(lhs);
                    llvm::Value *br = extract_real(rhs);
                    llvm::Value *bi = extract_imag(rhs);
                    
                    llvm::Value *result_real = nullptr;
                    llvm::Value *result_imag = nullptr;
                    
                    switch (instr.arg) {
                        case 0: // Add: (ar+br, ai+bi)
                            result_real = builder.CreateFAdd(ar, br, "add_real");
                            result_imag = builder.CreateFAdd(ai, bi, "add_imag");
                            break;
                        case 10: // Sub: (ar-br, ai-bi)
                            result_real = builder.CreateFSub(ar, br, "sub_real");
                            result_imag = builder.CreateFSub(ai, bi, "sub_imag");
                            break;
                        case 5: { // Mul: (ar*br - ai*bi, ar*bi + ai*br)
                            llvm::Value *ar_br = builder.CreateFMul(ar, br);
                            llvm::Value *ai_bi = builder.CreateFMul(ai, bi);
                            llvm::Value *ar_bi = builder.CreateFMul(ar, bi);
                            llvm::Value *ai_br = builder.CreateFMul(ai, br);
                            result_real = builder.CreateFSub(ar_br, ai_bi, "mul_real");
                            result_imag = builder.CreateFAdd(ar_bi, ai_br, "mul_imag");
                            break;
                        }
                        case 11: { // Div: (ar*br + ai*bi, ai*br - ar*bi) / (br*br + bi*bi)
                            llvm::Value *ar_br = builder.CreateFMul(ar, br);
                            llvm::Value *ai_bi = builder.CreateFMul(ai, bi);
                            llvm::Value *ai_br = builder.CreateFMul(ai, br);
                            llvm::Value *ar_bi = builder.CreateFMul(ar, bi);
                            llvm::Value *br_br = builder.CreateFMul(br, br);
                            llvm::Value *bi_bi = builder.CreateFMul(bi, bi);
                            llvm::Value *denom = builder.CreateFAdd(br_br, bi_bi, "denom");
                            llvm::Value *num_real = builder.CreateFAdd(ar_br, ai_bi);
                            llvm::Value *num_imag = builder.CreateFSub(ai_br, ar_bi);
                            result_real = builder.CreateFDiv(num_real, denom, "div_real");
                            result_imag = builder.CreateFDiv(num_imag, denom, "div_imag");
                            break;
                        }
                        default:
                            result_real = ar;
                            result_imag = ai;
                    }
                    stack.push_back(make_complex(result_real, result_imag));
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) {
                    builder.CreateRet(stack.back());
                } else {
                    llvm::Value *zero = llvm::ConstantFP::get(f32_type, 0.0f);
                    builder.CreateRet(make_complex(zero, zero));
                }
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < complex_constants.size()) {
                    auto &cc = complex_constants[instr.arg];
                    llvm::Value *real = llvm::ConstantFP::get(f32_type, cc.first);
                    llvm::Value *imag = llvm::ConstantFP::get(f32_type, cc.second);
                    builder.CreateRet(make_complex(real, imag));
                } else {
                    llvm::Value *zero = llvm::ConstantFP::get(f32_type, 0.0f);
                    builder.CreateRet(make_complex(zero, zero));
                }
            }
        }

        if (!builder.GetInsertBlock()->getTerminator()) {
            llvm::Value *zero = llvm::ConstantFP::get(f32_type, 0.0f);
            builder.CreateRet(make_complex(zero, zero));
        }

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Optional<f64> Mode Compilation (Nullable Float64)
    // =========================================================================
    bool JITCore::compile_optional_f64_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse constants - track which are None vs float
        struct OptionalConst {
            bool has_value;
            double value;
        };
        std::vector<OptionalConst> optional_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            if (const_obj.is_none()) {
                optional_constants.push_back({false, 0.0});
            } else if (nb::isinstance<nb::float_>(const_obj)) {
                optional_constants.push_back({true, nb::cast<double>(const_obj)});
            } else if (nb::isinstance<nb::int_>(const_obj)) {
                optional_constants.push_back({true, static_cast<double>(nb::cast<int64_t>(const_obj))});
            } else {
                optional_constants.push_back({false, 0.0});
            }
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Optional<f64> type: {i64, f64} for (has_value, value)
        // Uses i64 to match C++ struct ABI (int64_t has_value, double value)
        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);
        llvm::Type *f64_type = llvm::Type::getDoubleTy(*local_context);
        llvm::StructType *optional_type = llvm::StructType::get(*local_context, {i64_type, f64_type}, false);
        llvm::Type *ptr_type = llvm::PointerType::get(optional_type, 0);

        // Function signature: void fn(OptionalF64* out, OptionalF64* a, OptionalF64* b, ...)
        std::vector<llvm::Type *> param_types;
        param_types.push_back(ptr_type);  // output pointer
        for (int i = 0; i < param_count; ++i) {
            param_types.push_back(ptr_type);  // input pointers
        }
        llvm::FunctionType *func_type = llvm::FunctionType::get(builder.getVoidTy(), param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        // Get output pointer and input pointers from function args
        auto arg_iter = func->arg_begin();
        llvm::Value *out_ptr = &*arg_iter++;

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(optional_type, nullptr, "local_" + std::to_string(i));

        // Load input pointers into local variables
        int arg_idx = 0;
        while (arg_iter != func->arg_end() && arg_idx < param_count) {
            llvm::Value *input_ptr = &*arg_iter++;
            llvm::Value *input_val = builder.CreateLoad(optional_type, input_ptr);
            if (arg_idx < total_locals) {
                builder.CreateStore(input_val, local_allocas[arg_idx]);
            }
            ++arg_idx;
        }

        // Helper lambdas for optional operations
        auto extract_has_value = [&](llvm::Value *opt) -> llvm::Value* {
            return builder.CreateExtractValue(opt, {0}, "has_value");
        };
        auto extract_value = [&](llvm::Value *opt) -> llvm::Value* {
            return builder.CreateExtractValue(opt, {1}, "value");
        };
        auto make_some = [&](llvm::Value *value) -> llvm::Value* {
            llvm::Value *opt = llvm::UndefValue::get(optional_type);
            opt = builder.CreateInsertValue(opt, llvm::ConstantInt::get(i64_type, 1), {0});  // has_value = 1
            opt = builder.CreateInsertValue(opt, value, {1});
            return opt;
        };
        auto make_none = [&]() -> llvm::Value* {
            llvm::Value *opt = llvm::UndefValue::get(optional_type);
            opt = builder.CreateInsertValue(opt, llvm::ConstantInt::get(i64_type, 0), {0});  // has_value = 0
            opt = builder.CreateInsertValue(opt, llvm::ConstantFP::get(f64_type, 0.0), {1});
            return opt;
        };

        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(optional_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(optional_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(optional_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < optional_constants.size()) {
                    auto &oc = optional_constants[instr.arg];
                    if (oc.has_value) {
                        stack.push_back(make_some(llvm::ConstantFP::get(f64_type, oc.value)));
                    } else {
                        stack.push_back(make_none());
                    }
                }
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                // For optional types: if either operand is None, result is None
                // Otherwise perform the operation on the values
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    
                    llvm::Value *lhs_has = extract_has_value(lhs);
                    llvm::Value *rhs_has = extract_has_value(rhs);
                    llvm::Value *both_have = builder.CreateAnd(lhs_has, rhs_has, "both_have");
                    
                    llvm::Value *lhs_val = extract_value(lhs);
                    llvm::Value *rhs_val = extract_value(rhs);
                    
                    llvm::Value *result_val = nullptr;
                    switch (instr.arg) {
                        case 0: result_val = builder.CreateFAdd(lhs_val, rhs_val); break;
                        case 10: result_val = builder.CreateFSub(lhs_val, rhs_val); break;
                        case 5: result_val = builder.CreateFMul(lhs_val, rhs_val); break;
                        case 11: result_val = builder.CreateFDiv(lhs_val, rhs_val); break;
                        default: result_val = lhs_val;
                    }
                    
                    // Build result optional: {both_have, result_val}
                    llvm::Value *result = llvm::UndefValue::get(optional_type);
                    result = builder.CreateInsertValue(result, both_have, {0});
                    result = builder.CreateInsertValue(result, result_val, {1});
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) {
                    builder.CreateStore(stack.back(), out_ptr);
                } else {
                    builder.CreateStore(make_none(), out_ptr);
                }
                builder.CreateRetVoid();
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < optional_constants.size()) {
                    auto &oc = optional_constants[instr.arg];
                    if (oc.has_value) {
                        builder.CreateStore(make_some(llvm::ConstantFP::get(f64_type, oc.value)), out_ptr);
                    } else {
                        builder.CreateStore(make_none(), out_ptr);
                    }
                } else {
                    builder.CreateStore(make_none(), out_ptr);
                }
                builder.CreateRetVoid();
            }
        }

        if (!builder.GetInsertBlock()->getTerminator()) {
            builder.CreateStore(make_none(), out_ptr);
            builder.CreateRetVoid();
        }

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Ptr Mode Compilation (Array Access)
    // =========================================================================
    bool JITCore::compile_ptr_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        // Parse constants as doubles
        std::vector<double> float_constants;
        for (size_t i = 0; i < py_constants.size(); ++i) {
            nb::object const_obj = py_constants[i];
            if (nb::isinstance<nb::float_>(const_obj))
                float_constants.push_back(nb::cast<double>(const_obj));
            else if (nb::isinstance<nb::int_>(const_obj))
                float_constants.push_back(static_cast<double>(nb::cast<int64_t>(const_obj)));
            else
                float_constants.push_back(0.0);
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Types: ptr for array, i64 for indices, double for elements
        llvm::Type *ptr_type = llvm::PointerType::get(*local_context, 0);
        llvm::Type *i64_type = llvm::Type::getInt64Ty(*local_context);
        llvm::Type *f64_type = llvm::Type::getDoubleTy(*local_context);

        // Function takes ptr as first arg, remaining are i64
        std::vector<llvm::Type *> param_types;
        param_types.push_back(ptr_type); // First param: array pointer
        for (int i = 1; i < param_count; ++i)
            param_types.push_back(i64_type); // Remaining: indices/sizes

        llvm::FunctionType *func_type = llvm::FunctionType::get(f64_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas_ptr;

        // Create allocas for locals - first is ptr, rest are i64
        local_allocas_ptr[0] = builder.CreateAlloca(ptr_type, nullptr, "local_ptr_0");
        for (int i = 1; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(i64_type, nullptr, "local_" + std::to_string(i));

        // Store function arguments
        int arg_idx = 0;
        for (auto &arg : func->args()) {
            if (arg_idx == 0) {
                builder.CreateStore(&arg, local_allocas_ptr[0]);
            } else if (arg_idx < total_locals && local_allocas.count(arg_idx)) {
                builder.CreateStore(&arg, local_allocas[arg_idx]);
            }
            ++arg_idx;
        }

        // Code generation
        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (instr.arg == 0) {
                    // Load ptr
                    stack.push_back(builder.CreateLoad(ptr_type, local_allocas_ptr[0]));
                } else if (local_allocas.count(instr.arg)) {
                    stack.push_back(builder.CreateLoad(i64_type, local_allocas[instr.arg]));
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (idx1 == 0)
                    stack.push_back(builder.CreateLoad(ptr_type, local_allocas_ptr[0]));
                else if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(i64_type, local_allocas[idx1]));
                if (idx2 == 0)
                    stack.push_back(builder.CreateLoad(ptr_type, local_allocas_ptr[0]));
                else if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(i64_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::LOAD_CONST) {
                if (instr.arg < float_constants.size())
                    stack.push_back(llvm::ConstantFP::get(f64_type, float_constants[instr.arg]));
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && instr.arg > 0 && local_allocas.count(instr.arg)) {
                    llvm::Value *val = stack.back();
                    // Convert to i64 if needed for index storage
                    if (val->getType()->isDoubleTy())
                        val = builder.CreateFPToSI(val, i64_type);
                    else if (val->getType()->isPointerTy())
                        val = builder.CreatePtrToInt(val, i64_type);
                    builder.CreateStore(val, local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_SUBSCR) {
                // Array element access: arr[i]
                if (stack.size() >= 2) {
                    llvm::Value *idx = stack.back(); stack.pop_back();
                    llvm::Value *arr = stack.back(); stack.pop_back();
                    
                    // Convert idx to i64 if needed
                    if (idx->getType()->isDoubleTy())
                        idx = builder.CreateFPToSI(idx, i64_type);
                    
                    // GEP to get pointer to element
                    llvm::Value *elem_ptr = builder.CreateGEP(f64_type, arr, idx, "elem_ptr");
                    // Load the element
                    llvm::Value *elem = builder.CreateLoad(f64_type, elem_ptr, "elem");
                    stack.push_back(elem);
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    
                    // Check if this is pointer arithmetic (ptr + int or ptr - int)
                    bool lhs_is_ptr = lhs->getType()->isPointerTy();
                    bool rhs_is_ptr = rhs->getType()->isPointerTy();
                    
                    if (lhs_is_ptr && !rhs_is_ptr) {
                        // Pointer arithmetic: ptr + offset or ptr - offset
                        // Convert rhs to i64 if needed
                        llvm::Value *offset = rhs;
                        if (offset->getType()->isDoubleTy())
                            offset = builder.CreateFPToSI(offset, i64_type);
                        
                        llvm::Value *result = nullptr;
                        switch (instr.arg) {
                            case 0: // Add: ptr + offset
                                result = builder.CreateGEP(builder.getInt8Ty(), lhs, offset, "ptr_add");
                                break;
                            case 10: // Sub: ptr - offset (use negative offset)
                                offset = builder.CreateNeg(offset, "neg_offset");
                                result = builder.CreateGEP(builder.getInt8Ty(), lhs, offset, "ptr_sub");
                                break;
                            default:
                                result = lhs;
                        }
                        stack.push_back(result);
                    } else if (!lhs_is_ptr && !rhs_is_ptr) {
                        // Regular numeric operation
                        // Ensure both are f64
                        if (lhs->getType()->isIntegerTy())
                            lhs = builder.CreateSIToFP(lhs, f64_type);
                        if (rhs->getType()->isIntegerTy())
                            rhs = builder.CreateSIToFP(rhs, f64_type);
                        
                        llvm::Value *result = nullptr;
                        switch (instr.arg) {
                            case 0: result = builder.CreateFAdd(lhs, rhs); break;
                            case 10: result = builder.CreateFSub(lhs, rhs); break;
                            case 5: result = builder.CreateFMul(lhs, rhs); break;
                            case 11: result = builder.CreateFDiv(lhs, rhs); break;
                            default: result = lhs;
                        }
                        stack.push_back(result);
                    } else {
                        // Both are pointers or rhs is ptr - convert to integers for arithmetic
                        llvm::Value *lhs_int = lhs_is_ptr ? builder.CreatePtrToInt(lhs, i64_type) : lhs;
                        llvm::Value *rhs_int = rhs_is_ptr ? builder.CreatePtrToInt(rhs, i64_type) : rhs;
                        
                        if (lhs_int->getType()->isDoubleTy())
                            lhs_int = builder.CreateFPToSI(lhs_int, i64_type);
                        if (rhs_int->getType()->isDoubleTy())
                            rhs_int = builder.CreateFPToSI(rhs_int, i64_type);
                        
                        llvm::Value *result = nullptr;
                        switch (instr.arg) {
                            case 0: result = builder.CreateAdd(lhs_int, rhs_int); break;
                            case 10: result = builder.CreateSub(lhs_int, rhs_int); break;
                            case 5: result = builder.CreateMul(lhs_int, rhs_int); break;
                            default: result = lhs_int;
                        }
                        
                        // If original lhs was ptr, convert result back to ptr
                        if (lhs_is_ptr)
                            result = builder.CreateIntToPtr(result, ptr_type);
                        
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) {
                    llvm::Value *ret = stack.back();
                    if (ret->getType()->isIntegerTy())
                        ret = builder.CreateSIToFP(ret, f64_type);
                    builder.CreateRet(ret);
                } else {
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));
                }
            }
            else if (instr.opcode == op::RETURN_CONST) {
                if (instr.arg < float_constants.size())
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, float_constants[instr.arg]));
                else
                    builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));
            }
        }

        if (!builder.GetInsertBlock()->getTerminator())
            builder.CreateRet(llvm::ConstantFP::get(f64_type, 0.0));

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Vec4f Mode Compilation (SSE SIMD)
    // =========================================================================
    // Uses ptr-based ABI: void fn(float* out, float* a, float* b)
    // Internally loads to <4 x float>, does SIMD ops, stores result
    bool JITCore::compile_vec4f_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Types
        llvm::Type *void_type = llvm::Type::getVoidTy(*local_context);
        llvm::Type *ptr_type = llvm::PointerType::get(*local_context, 0);
        llvm::Type *f32 = llvm::Type::getFloatTy(*local_context);
        llvm::FixedVectorType *vec4f_type = llvm::FixedVectorType::get(f32, 4);

        // Function signature: void fn(ptr out, ptr a, ptr b)
        // param_count=2 means a+b, we add out as first hidden param
        std::vector<llvm::Type *> param_types;
        param_types.push_back(ptr_type);  // out
        for (int i = 0; i < param_count; ++i)
            param_types.push_back(ptr_type);  // inputs

        llvm::FunctionType *func_type = llvm::FunctionType::get(void_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        // Get function args
        auto args = func->arg_begin();
        llvm::Value *out_ptr = &*args++;
        std::vector<llvm::Value*> input_ptrs;
        for (int i = 0; i < param_count; ++i) {
            input_ptrs.push_back(&*args++);
        }

        // Stack for bytecode ops
        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(vec4f_type, nullptr, "local_" + std::to_string(i));

        // Load input vectors into local allocas (treating params as vec4f)
        for (int i = 0; i < param_count && i < total_locals; ++i) {
            llvm::Value *vec = builder.CreateAlignedLoad(vec4f_type, input_ptrs[i], llvm::MaybeAlign(16), "input_" + std::to_string(i));
            builder.CreateStore(vec, local_allocas[i]);
        }

        llvm::Value *result_vec = nullptr;

        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(vec4f_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(vec4f_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(vec4f_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *res = nullptr;
                    switch (instr.arg) {
                        case 0: res = builder.CreateFAdd(lhs, rhs); break;
                        case 10: res = builder.CreateFSub(lhs, rhs); break;
                        case 5: res = builder.CreateFMul(lhs, rhs); break;
                        case 11: res = builder.CreateFDiv(lhs, rhs); break;
                        default: res = lhs;
                    }
                    stack.push_back(res);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) result_vec = stack.back();
            }
        }

        // Store result to output pointer
        if (result_vec) {
            builder.CreateAlignedStore(result_vec, out_ptr, llvm::MaybeAlign(16));
        } else {
            builder.CreateAlignedStore(llvm::ConstantAggregateZero::get(vec4f_type), out_ptr, llvm::MaybeAlign(16));
        }
        builder.CreateRetVoid();

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

        compiled_functions.insert(name);
        return true;
    }

    // =========================================================================
    // Vec8i Mode Compilation (AVX SIMD)
    // =========================================================================
    // Uses ptr-based ABI: void fn(int32_t* out, int32_t* a, int32_t* b)
    // Internally loads to <8 x i32>, does SIMD ops, stores result
    bool JITCore::compile_vec8i_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count, int total_locals)
    {
        if (!jit) return false;
        if (compiled_functions.count(name) > 0) return true;

        std::vector<Instruction> instructions;
        for (size_t i = 0; i < py_instructions.size(); ++i) {
            nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
            Instruction instr;
            instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
            instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
            instr.argval = nb::cast<int32_t>(instr_dict["argval"]);
            instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
            instructions.push_back(instr);
        }

        auto local_context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(name, *local_context);
        llvm::IRBuilder<> builder(*local_context);

        // Types
        llvm::Type *void_type = llvm::Type::getVoidTy(*local_context);
        llvm::Type *ptr_type = llvm::PointerType::get(*local_context, 0);
        llvm::Type *i32 = llvm::Type::getInt32Ty(*local_context);
        llvm::FixedVectorType *vec8i_type = llvm::FixedVectorType::get(i32, 8);

        // Function signature: void fn(ptr out, ptr a, ptr b)
        std::vector<llvm::Type *> param_types;
        param_types.push_back(ptr_type);  // out
        for (int i = 0; i < param_count; ++i)
            param_types.push_back(ptr_type);  // inputs

        llvm::FunctionType *func_type = llvm::FunctionType::get(void_type, param_types, false);
        llvm::Function *func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, module.get());

        llvm::BasicBlock *entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);

        // Get function args
        auto args = func->arg_begin();
        llvm::Value *out_ptr = &*args++;
        std::vector<llvm::Value*> input_ptrs;
        for (int i = 0; i < param_count; ++i) {
            input_ptrs.push_back(&*args++);
        }

        // Stack for bytecode ops
        std::vector<llvm::Value *> stack;
        std::unordered_map<int, llvm::AllocaInst *> local_allocas;
        for (int i = 0; i < total_locals; ++i)
            local_allocas[i] = builder.CreateAlloca(vec8i_type, nullptr, "local_" + std::to_string(i));

        // Load input vectors into local allocas
        for (int i = 0; i < param_count && i < total_locals; ++i) {
            llvm::Value *vec = builder.CreateAlignedLoad(vec8i_type, input_ptrs[i], llvm::MaybeAlign(32), "input_" + std::to_string(i));
            builder.CreateStore(vec, local_allocas[i]);
        }

        llvm::Value *result_vec = nullptr;

        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto &instr = instructions[i];
            
            if (instr.opcode == op::RESUME || instr.opcode == op::NOP || instr.opcode == op::CACHE) {
                // No-op
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg))
                    stack.push_back(builder.CreateLoad(vec8i_type, local_allocas[instr.arg]));
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                int idx1 = (instr.arg >> 4) & 0xF;
                int idx2 = instr.arg & 0xF;
                if (local_allocas.count(idx1))
                    stack.push_back(builder.CreateLoad(vec8i_type, local_allocas[idx1]));
                if (local_allocas.count(idx2))
                    stack.push_back(builder.CreateLoad(vec8i_type, local_allocas[idx2]));
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty() && local_allocas.count(instr.arg)) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    llvm::Value *rhs = stack.back(); stack.pop_back();
                    llvm::Value *lhs = stack.back(); stack.pop_back();
                    llvm::Value *res = nullptr;
                    switch (instr.arg) {
                        case 0: res = builder.CreateAdd(lhs, rhs); break;
                        case 10: res = builder.CreateSub(lhs, rhs); break;
                        case 5: res = builder.CreateMul(lhs, rhs); break;
                        case 2: res = builder.CreateSDiv(lhs, rhs); break;
                        default: res = lhs;
                    }
                    stack.push_back(res);
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!stack.empty()) result_vec = stack.back();
            }
        }

        // Store result to output pointer
        if (result_vec) {
            builder.CreateAlignedStore(result_vec, out_ptr, llvm::MaybeAlign(32));
        } else {
            builder.CreateAlignedStore(llvm::ConstantAggregateZero::get(vec8i_type), out_ptr, llvm::MaybeAlign(32));
        }
        builder.CreateRetVoid();

        if (dump_ir) {
            std::string ir_str;
            llvm::raw_string_ostream ir_stream(ir_str);
            module->print(ir_stream, nullptr);
            last_ir = ir_stream.str();
        }

        optimize_module(*module, func);
        auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
        if (err) return false;

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
        // Debug flag for tracing generator execution
        // Set to true to enable runtime trace output
        const bool DEBUG_GENERATOR = true;
        
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
            // These effects should match Python's dis.stack_effect() exactly
            if (instr.opcode == op::LOAD_CONST || instr.opcode == op::LOAD_FAST ||
                instr.opcode == op::LOAD_FAST_CHECK || instr.opcode == op::PUSH_NULL) {
                simulated_depth++;
            } else if (instr.opcode == op::LOAD_ATTR) {
                // LOAD_ATTR: effect is 0 for normal attr load, +1 for method load (low bit set)
                bool is_method = (instr.arg & 1) != 0;
                if (is_method) simulated_depth++;
                // Normal attr load: pops object, pushes attr = net 0
            } else if (instr.opcode == op::LOAD_GLOBAL) {
                // LOAD_GLOBAL: pushes 1 value, +1 more if low bit is set (push NULL for method)
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
                // END_FOR: In Python 3.13, only pops the iterator (effect -1)
                // FOR_ITER jumps directly on exhaustion, no NULL is pushed
                if (simulated_depth > 0) simulated_depth--;
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
            } else if (instr.opcode == op::CLEANUP_THROW) {
                // CLEANUP_THROW: pops 3 values, pushes 1 result
                if (simulated_depth >= 3) simulated_depth -= 2;  // Net: -2
            } else if (instr.opcode == op::BUILD_MAP) {
                // BUILD_MAP: pops 2*arg items (key-value pairs), pushes 1 dict
                if (simulated_depth >= static_cast<size_t>(instr.arg * 2)) {
                    simulated_depth -= (instr.arg * 2);
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::BUILD_SET) {
                // BUILD_SET: pops arg items, pushes 1 set
                if (simulated_depth >= static_cast<size_t>(instr.arg)) {
                    simulated_depth -= instr.arg;
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::UNPACK_SEQUENCE) {
                // UNPACK_SEQUENCE: pops 1, pushes arg items
                if (simulated_depth > 0) simulated_depth--;
                simulated_depth += instr.arg;
            } else if (instr.opcode == op::UNPACK_EX) {
                // UNPACK_EX: pops 1, pushes (count_before + 1 + count_after)
                int count_before = instr.arg & 0xFF;
                int count_after = (instr.arg >> 8) & 0xFF;
                if (simulated_depth > 0) simulated_depth--;
                simulated_depth += (count_before + 1 + count_after);
            } else if (instr.opcode == op::CALL_KW) {
                // CALL_KW: pops (callable + self_or_null + args + kwnames), pushes 1
                int num_args = instr.arg;
                if (simulated_depth >= static_cast<size_t>(num_args + 3)) {
                    simulated_depth -= (num_args + 3);
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::CALL_FUNCTION_EX) {
                // CALL_FUNCTION_EX: pops 3 or 4 (callable, null, args, [kwargs]), pushes 1
                bool has_kwargs = (instr.arg & 1) != 0;
                size_t pop_count = has_kwargs ? 4 : 3;
                if (simulated_depth >= pop_count) {
                    simulated_depth -= pop_count;
                } else {
                    simulated_depth = 0;
                }
                simulated_depth++;
            } else if (instr.opcode == op::STORE_GLOBAL || instr.opcode == op::STORE_ATTR) {
                // STORE_GLOBAL: pops 1 value
                // STORE_ATTR: pops 2 (value and object)
                if (instr.opcode == op::STORE_GLOBAL) {
                    if (simulated_depth > 0) simulated_depth--;
                } else {
                    if (simulated_depth >= 2) simulated_depth -= 2;
                }
            } else if (instr.opcode == op::LIST_APPEND || instr.opcode == op::SET_ADD) {
                // LIST_APPEND/SET_ADD: pops 1 item (list/set stays on stack)
                if (simulated_depth > 0) simulated_depth--;
            } else if (instr.opcode == op::MAP_ADD) {
                // MAP_ADD: pops 2 (key, value), dict stays on stack
                if (simulated_depth >= 2) simulated_depth -= 2;
            } else if (instr.opcode == op::DELETE_SUBSCR) {
                // DELETE_SUBSCR: pops 2 (key and container)
                if (simulated_depth >= 2) simulated_depth -= 2;
            } else if (instr.opcode == op::LOAD_DEREF || instr.opcode == op::LOAD_CLOSURE) {
                // LOAD_DEREF/LOAD_CLOSURE: pushes 1
                simulated_depth++;
            } else if (instr.opcode == op::STORE_DEREF) {
                // STORE_DEREF: pops 1
                if (simulated_depth > 0) simulated_depth--;
            } else if (instr.opcode == op::COPY_FREE_VARS || instr.opcode == op::MAKE_CELL) {
                // No stack effect
            } else if (instr.opcode == op::IMPORT_NAME) {
                // IMPORT_NAME: pops 2 (fromlist, level), pushes 1 (module)
                if (simulated_depth >= 2) simulated_depth--;
            } else if (instr.opcode == op::IMPORT_FROM) {
                // IMPORT_FROM: module stays on stack, pushes 1 attribute
                simulated_depth++;
            } else if (instr.opcode == op::MAKE_FUNCTION) {
                // MAKE_FUNCTION: pops 1 (code), pushes 1 (function)
                // Net: 0
            } else if (instr.opcode == op::SET_FUNCTION_ATTRIBUTE) {
                // SET_FUNCTION_ATTRIBUTE: pops 2 (func, value), pushes 1 (func)
                if (simulated_depth >= 2) simulated_depth--;
            } else if (instr.opcode == op::LIST_EXTEND || instr.opcode == op::SET_UPDATE ||
                       instr.opcode == op::DICT_UPDATE || instr.opcode == op::DICT_MERGE) {
                // These pop 1 iterable/dict, list/set/dict stays on stack
                if (simulated_depth > 0) simulated_depth--;
            } else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                // LOAD_FAST_LOAD_FAST: pushes 2 values
                simulated_depth += 2;
            } else if (instr.opcode == op::STORE_FAST_STORE_FAST) {
                // STORE_FAST_STORE_FAST: pops 2 values
                if (simulated_depth >= 2) simulated_depth -= 2;
            } else if (instr.opcode == op::STORE_FAST_LOAD_FAST) {
                // STORE_FAST_LOAD_FAST: pops 1, pushes 1 (net: 0)
            } else if (instr.opcode == op::LOAD_FAST_AND_CLEAR) {
                // LOAD_FAST_AND_CLEAR: pushes 1
                simulated_depth++;
            } else if (instr.opcode == op::CONTAINS_OP || instr.opcode == op::IS_OP) {
                // These pop 2, push 1
                if (simulated_depth >= 2) simulated_depth--;
            } else if (instr.opcode == op::BUILD_SLICE) {
                // BUILD_SLICE: pops 2 or 3, pushes 1
                int argc = instr.arg;
                if (simulated_depth >= static_cast<size_t>(argc)) {
                    simulated_depth -= argc;
                }
                simulated_depth++;
            } else if (instr.opcode == op::BINARY_SLICE) {
                // BINARY_SLICE: pops 3 (obj, start, stop), pushes 1
                if (simulated_depth >= 3) simulated_depth -= 2;
            } else if (instr.opcode == op::STORE_SLICE) {
                // STORE_SLICE: pops 4 (value, obj, start, stop)
                if (simulated_depth >= 4) simulated_depth -= 4;
            } else if (instr.opcode == op::TO_BOOL || instr.opcode == op::UNARY_NEGATIVE ||
                       instr.opcode == op::UNARY_NOT || instr.opcode == op::UNARY_INVERT) {
                // Unary ops: 1 in, 1 out (no change)
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
        
        // Compute actual total_locals based on simulated stack depth
        // The Python side passes co_stacksize, but our simulation may find different requirements
        int actual_total_locals = nlocals + static_cast<int>(max_stack_depth);
        if (actual_total_locals < total_locals) {
            actual_total_locals = total_locals;  // Use the larger of the two
        }
        
        // Store the computed total_locals for get_generator_callable
        generator_total_locals[name] = actual_total_locals;
        
        size_t max_stack_slots = static_cast<size_t>(actual_total_locals - nlocals);

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
        builder.CreateCall(py_xincref_func, {py_none});
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
                        builder.CreateCall(py_xdecref_func, {val});
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

        // Identify pure exception handler offsets (only reachable via exception, not normal jumps)
        // These blocks should NOT be entered via fallthrough during linear code generation
        std::unordered_set<int> pure_exception_handler_offsets;
        for (const auto &exc_entry : exception_table)
        {
            pure_exception_handler_offsets.insert(exc_entry.target);
        }
        // Remove offsets that are also normal jump targets
        for (int target : jump_targets)
        {
            pure_exception_handler_offsets.erase(target);
        }
        
        // For pure exception handlers, add a terminator (return NULL to propagate exception)
        // This handles the case where the exception handler does cleanup and re-raises
        for (int exc_offset : pure_exception_handler_offsets)
        {
            if (offset_blocks.count(exc_offset))
            {
                llvm::BasicBlock *exc_block = offset_blocks[exc_offset];
                builder.SetInsertPoint(exc_block);
                // Set state to -2 (exception occurred) and return NULL
                builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
            }
        }
        
        // Reset builder to start block for main code generation
        builder.SetInsertPoint(state_0);

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
        
        // Track if we're currently in a reachable code section
        bool in_unreachable_section = false;
        
        // Helper lambda to emit debug trace call
        auto emit_debug_trace = [&](int offset, const char* opname, size_t stack_depth, llvm::Value* value = nullptr) {
            if (!DEBUG_GENERATOR) return;
            
            // Create string constant for opname
            llvm::Value *opname_str = builder.CreateGlobalStringPtr(opname);
            llvm::Value *offset_val = llvm::ConstantInt::get(i32_type, offset);
            llvm::Value *depth_val = llvm::ConstantInt::get(i32_type, static_cast<int>(stack_depth));
            llvm::Value *value_ptr = value ? value : llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
            
            builder.CreateCall(jit_debug_trace_func, {offset_val, opname_str, depth_val, value_ptr});
        };
        
        for (size_t i = start_idx; i < instructions.size(); ++i)
        {
            const auto &instr = instructions[i];
            
            // If this is a pure exception handler offset that we've never reached via normal flow,
            // we should NOT generate code for it during linear iteration.
            // Instead, it will only be reached via exception branches.
            if (pure_exception_handler_offsets.count(instr.offset))
            {
                // Check if we've already initialized this block (meaning it was reached via exception)
                if (initialized_blocks.find(offset_blocks[instr.offset]) == initialized_blocks.end())
                {
                    // Mark as unreachable - we'll skip until we hit a reachable block
                    in_unreachable_section = true;
                }
            }
            
            // Check if this offset is a normal jump target - need new block
            if (offset_blocks.count(instr.offset) && !pure_exception_handler_offsets.count(instr.offset))
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
                            // Use XINCREF to handle NULL values safely (e.g., from LOAD_FAST_AND_CLEAR)
                            builder.CreateCall(py_xincref_func, {val});
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
                        // For exception handlers, use the depth from the exception table
                        // For normal jump targets, use the target_stack_depth map
                        size_t expected_depth = 0;
                        if (exception_handler_depth.count(instr.offset)) {
                            expected_depth = exception_handler_depth[instr.offset];
                        } else if (target_stack_depth.count(instr.offset)) {
                            expected_depth = target_stack_depth[instr.offset];
                        }
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
                
                // If we reached a normal jump target, we're back in reachable code
                in_unreachable_section = false;
            }
            
            // Skip code generation for unreachable sections (pure exception handlers)
            if (in_unreachable_section)
            {
                continue;
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
                emit_debug_trace(instr.offset, "POP_TOP (before)", stack.size());
                
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    // Show what we're popping
                    emit_debug_trace(instr.offset, "POP_TOP popping", stack.size(), val);
                    
                    stack.pop_back();
                    // Use XDECREF to safely handle NULL values (e.g., from LOAD_FAST_AND_CLEAR)
                    builder.CreateCall(py_xdecref_func, {val});
                    
                    emit_debug_trace(instr.offset, "POP_TOP (after)", stack.size());
                }
            }
            else if (instr.opcode == op::LOAD_FAST || instr.opcode == op::LOAD_FAST_CHECK)
            {
                emit_debug_trace(instr.offset, "LOAD_FAST (before)", stack.size());
                
                llvm::Value *val = load_local(instr.arg);
                builder.CreateCall(py_xincref_func, {val});
                stack.push_back(val);
                
                emit_debug_trace(instr.offset, "LOAD_FAST (after)", stack.size(), val);
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST)
            {
                // Python 3.13: Pushes co_varnames[arg>>4] then co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                
                llvm::Value *val1 = load_local(first_local);
                builder.CreateCall(py_xincref_func, {val1});
                stack.push_back(val1);
                
                llvm::Value *val2 = load_local(second_local);
                builder.CreateCall(py_xincref_func, {val2});
                stack.push_back(val2);
            }
            else if (instr.opcode == op::STORE_FAST)
            {
                emit_debug_trace(instr.offset, ("STORE_FAST " + std::to_string(instr.arg) + " (before)").c_str(), stack.size());
                
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    
                    // Show what we're storing
                    emit_debug_trace(instr.offset, ("STORE_FAST " + std::to_string(instr.arg) + " storing").c_str(), stack.size(), val);

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
                    builder.CreateCall(py_xdecref_func, {old_val});
                    builder.CreateBr(after_decref);

                    builder.SetInsertPoint(after_decref);
                    store_local(instr.arg, val);
                    
                    emit_debug_trace(instr.offset, ("STORE_FAST " + std::to_string(instr.arg) + " (after)").c_str(), stack.size());
                }
            }
            // ========== STORE_FAST_STORE_FAST ==========
            else if (instr.opcode == op::STORE_FAST_STORE_FAST)
            {
                // Python 3.13: Stores STACK[-1] into co_varnames[arg>>4] and STACK[-2] into co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;

                if (stack.size() >= 2)
                {
                    llvm::Value *first_val = stack.back();
                    stack.pop_back();
                    llvm::Value *second_val = stack.back();
                    stack.pop_back();

                    llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));

                    // Store first value with decref of old
                    llvm::Value *old_val1 = load_local(first_local);
                    llvm::Value *is_null1 = builder.CreateICmpEQ(old_val1, null_ptr);
                    llvm::BasicBlock *decref_block1 = llvm::BasicBlock::Create(*local_context, "decref_ssss1", func);
                    llvm::BasicBlock *store_block1 = llvm::BasicBlock::Create(*local_context, "store_ssss1", func);
                    builder.CreateCondBr(is_null1, store_block1, decref_block1);
                    builder.SetInsertPoint(decref_block1);
                    builder.CreateCall(py_xdecref_func, {old_val1});
                    builder.CreateBr(store_block1);
                    builder.SetInsertPoint(store_block1);
                    store_local(first_local, first_val);

                    // Store second value with decref of old
                    llvm::Value *old_val2 = load_local(second_local);
                    llvm::Value *is_null2 = builder.CreateICmpEQ(old_val2, null_ptr);
                    llvm::BasicBlock *decref_block2 = llvm::BasicBlock::Create(*local_context, "decref_ssss2", func);
                    llvm::BasicBlock *store_block2 = llvm::BasicBlock::Create(*local_context, "store_ssss2", func);
                    builder.CreateCondBr(is_null2, store_block2, decref_block2);
                    builder.SetInsertPoint(decref_block2);
                    builder.CreateCall(py_xdecref_func, {old_val2});
                    builder.CreateBr(store_block2);
                    builder.SetInsertPoint(store_block2);
                    store_local(second_local, second_val);
                    current_block = store_block2;
                }
            }
            // ========== STORE_FAST_LOAD_FAST ==========
            else if (instr.opcode == op::STORE_FAST_LOAD_FAST)
            {
                // Python 3.13: Stores TOS into co_varnames[arg>>4], then loads co_varnames[arg&15]
                int store_local_idx = instr.arg >> 4;
                int load_local_idx = instr.arg & 0xF;

                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();

                    llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));

                    // Decref and store
                    llvm::Value *old_val = load_local(store_local_idx);
                    llvm::Value *is_null = builder.CreateICmpEQ(old_val, null_ptr);
                    llvm::BasicBlock *decref_block = llvm::BasicBlock::Create(*local_context, "decref_sflf", func);
                    llvm::BasicBlock *store_block = llvm::BasicBlock::Create(*local_context, "store_sflf", func);
                    builder.CreateCondBr(is_null, store_block, decref_block);
                    builder.SetInsertPoint(decref_block);
                    builder.CreateCall(py_xdecref_func, {old_val});
                    builder.CreateBr(store_block);
                    builder.SetInsertPoint(store_block);
                    store_local(store_local_idx, val);
                    current_block = store_block;

                    // Load the other local
                    llvm::Value *loaded = load_local(load_local_idx);
                    builder.CreateCall(py_xincref_func, {loaded});
                    stack.push_back(loaded);
                }
            }
            // ========== LOAD_FAST_AND_CLEAR ==========
            else if (instr.opcode == op::LOAD_FAST_AND_CLEAR)
            {
                // Push local value (or NULL) and clear the slot
                llvm::Value *val = load_local(instr.arg);
                builder.CreateCall(py_xincref_func, {val});
                stack.push_back(val);
                store_local(instr.arg, llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
            }
            else if (instr.opcode == op::LOAD_CONST)
            {
                if (instr.arg < obj_constants.size() && obj_constants[instr.arg] != nullptr)
                {
                    llvm::Value *const_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(obj_constants[instr.arg]));
                    llvm::Value *py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                    builder.CreateCall(py_xincref_func, {py_obj});
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

                    builder.CreateCall(py_xdecref_func, {lhs});
                    builder.CreateCall(py_xdecref_func, {rhs});
                    
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

                    builder.CreateCall(py_xdecref_func, {lhs});
                    builder.CreateCall(py_xdecref_func, {rhs});

                    // Convert int result to Python bool
                    llvm::Value *true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *is_true = builder.CreateICmpNE(result, llvm::ConstantInt::get(i32_type, 0));
                    llvm::Value *bool_result = builder.CreateSelect(is_true,
                        builder.CreateIntToPtr(true_ptr, ptr_type),
                        builder.CreateIntToPtr(false_ptr, ptr_type));
                    builder.CreateCall(py_xincref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== CONTAINS_OP ==========
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

                    // PySequence_Contains returns 1 if contains, 0 if not, -1 on error
                    llvm::Value *result = builder.CreateCall(py_sequence_contains_func, {container, value}, "contains");

                    if (invert)
                    {
                        // 'not in': invert the result (1->0, 0->1)
                        result = builder.CreateXor(result, llvm::ConstantInt::get(result->getType(), 1), "not_in");
                    }

                    // Decref consumed operands
                    builder.CreateCall(py_xdecref_func, {value});
                    builder.CreateCall(py_xdecref_func, {container});

                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *is_true = builder.CreateICmpSGT(result, llvm::ConstantInt::get(result->getType(), 0));
                    llvm::Value *bool_result = builder.CreateSelect(is_true, py_true, py_false);
                    builder.CreateCall(py_xincref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== IS_OP ==========
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

                    // Pointer identity comparison
                    llvm::Value *is_same = builder.CreateICmpEQ(lhs, rhs, "is");

                    if (invert)
                    {
                        is_same = builder.CreateNot(is_same, "is_not");
                    }

                    // Decref consumed operands
                    builder.CreateCall(py_xdecref_func, {lhs});
                    builder.CreateCall(py_xdecref_func, {rhs});

                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *bool_result = builder.CreateSelect(is_same, py_true, py_false);
                    builder.CreateCall(py_xincref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== TO_BOOL ==========
            else if (instr.opcode == op::TO_BOOL)
            {
                // Converts TOS to a boolean
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    
                    // PyObject_IsTrue returns 1 for true, 0 for false, -1 for error
                    llvm::Value *is_true = builder.CreateCall(py_object_istrue_func, {val}, "is_true");
                    builder.CreateCall(py_xdecref_func, {val});
                    
                    // Convert to Python bool
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value *cmp = builder.CreateICmpSGT(is_true, llvm::ConstantInt::get(i32_type, 0));
                    llvm::Value *bool_result = builder.CreateSelect(cmp, py_true, py_false);
                    builder.CreateCall(py_xincref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== UNARY_NOT ==========
            else if (instr.opcode == op::UNARY_NOT)
            {
                // Logical NOT: TOS = not TOS
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    
                    // PyObject_IsTrue returns 1 for true, 0 for false, -1 for error
                    llvm::Value *is_true = builder.CreateCall(py_object_istrue_func, {val}, "is_true");
                    builder.CreateCall(py_xdecref_func, {val});
                    
                    // Negate: true becomes false, false becomes true
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    // If is_true > 0, result is False; else True
                    llvm::Value *cmp = builder.CreateICmpSGT(is_true, llvm::ConstantInt::get(i32_type, 0));
                    llvm::Value *bool_result = builder.CreateSelect(cmp, py_false, py_true);
                    builder.CreateCall(py_xincref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            // ========== UNARY_NEGATIVE ==========
            else if (instr.opcode == op::UNARY_NEGATIVE)
            {
                // Negate TOS: TOS = -TOS
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    
                    llvm::Value *result = builder.CreateCall(py_number_negative_func, {val}, "neg");
                    builder.CreateCall(py_xdecref_func, {val});
                    stack.push_back(result);
                }
            }
            // ========== UNARY_INVERT ==========
            else if (instr.opcode == op::UNARY_INVERT)
            {
                // Bitwise invert TOS: TOS = ~TOS
                if (!stack.empty())
                {
                    llvm::Value *val = stack.back();
                    stack.pop_back();
                    
                    llvm::Value *result = builder.CreateCall(py_number_invert_func, {val}, "invert");
                    builder.CreateCall(py_xdecref_func, {val});
                    stack.push_back(result);
                }
            }
            // ========== BUILD_SLICE ==========
            else if (instr.opcode == op::BUILD_SLICE)
            {
                // Build a slice object
                // arg=2: slice(start, stop), arg=3: slice(start, stop, step)
                int argc = instr.arg;
                llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                
                if (argc == 2 && stack.size() >= 2)
                {
                    llvm::Value *stop = stack.back(); stack.pop_back();
                    llvm::Value *start = stack.back(); stack.pop_back();
                    
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none}, "slice2");
                    
                    builder.CreateCall(py_xdecref_func, {start});
                    builder.CreateCall(py_xdecref_func, {stop});
                    
                    stack.push_back(slice);
                }
                else if (argc == 3 && stack.size() >= 3)
                {
                    llvm::Value *step = stack.back(); stack.pop_back();
                    llvm::Value *stop = stack.back(); stack.pop_back();
                    llvm::Value *start = stack.back(); stack.pop_back();
                    
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, step}, "slice3");
                    
                    builder.CreateCall(py_xdecref_func, {start});
                    builder.CreateCall(py_xdecref_func, {stop});
                    builder.CreateCall(py_xdecref_func, {step});
                    
                    stack.push_back(slice);
                }
            }
            // ========== BINARY_SLICE ==========
            else if (instr.opcode == op::BINARY_SLICE)
            {
                // TOS = TOS2[TOS1:TOS]  (slice from TOS1 to TOS)
                if (stack.size() >= 3)
                {
                    llvm::Value *stop = stack.back(); stack.pop_back();
                    llvm::Value *start = stack.back(); stack.pop_back();
                    llvm::Value *obj = stack.back(); stack.pop_back();
                    
                    // Build slice object
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none}, "slice");
                    
                    // Get item with slice
                    llvm::Value *result = builder.CreateCall(py_object_getitem_func, {obj, slice}, "sliced");
                    
                    // Decref consumed values
                    builder.CreateCall(py_xdecref_func, {slice});
                    builder.CreateCall(py_xdecref_func, {obj});
                    builder.CreateCall(py_xdecref_func, {start});
                    builder.CreateCall(py_xdecref_func, {stop});
                    
                    stack.push_back(result);
                }
            }
            // ========== STORE_SLICE ==========
            else if (instr.opcode == op::STORE_SLICE)
            {
                // TOS2[TOS1:TOS] = TOS3
                if (stack.size() >= 4)
                {
                    llvm::Value *stop = stack.back(); stack.pop_back();
                    llvm::Value *start = stack.back(); stack.pop_back();
                    llvm::Value *obj = stack.back(); stack.pop_back();
                    llvm::Value *value = stack.back(); stack.pop_back();
                    
                    // Build slice object
                    llvm::Value *py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value *py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value *slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none}, "slice");
                    
                    // Set item with slice
                    builder.CreateCall(py_object_setitem_func, {obj, slice, value});
                    
                    // Decref consumed values
                    builder.CreateCall(py_xdecref_func, {slice});
                    builder.CreateCall(py_xdecref_func, {obj});
                    builder.CreateCall(py_xdecref_func, {start});
                    builder.CreateCall(py_xdecref_func, {stop});
                    builder.CreateCall(py_xdecref_func, {value});
                }
            }
            // ========== NOP ==========
            else if (instr.opcode == op::NOP)
            {
                // No operation - do nothing
            }
            // ========== CACHE ==========
            else if (instr.opcode == op::CACHE)
            {
                // Cache slot - do nothing (used by adaptive interpreter)
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
                    builder.CreateCall(py_xincref_func, {result_phi});

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
                    builder.CreateCall(py_xdecref_func, {obj});

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
                        builder.CreateCall(py_xdecref_func, {value});
                    }
                    
                    builder.CreateCall(py_xdecref_func, {keys_tuple});
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
                    builder.CreateCall(py_xdecref_func, {key});
                    builder.CreateCall(py_xdecref_func, {value});
                    // container is borrowed (it stays alive)
                    builder.CreateCall(py_xdecref_func, {container});
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
                    builder.CreateCall(py_xdecref_func, {key});
                    builder.CreateCall(py_xdecref_func, {container});

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
                    builder.CreateCall(py_xdecref_func, {args_tuple});

                    // Decref callable
                    builder.CreateCall(py_xdecref_func, {callable});

                    // Handle self_or_null decref
                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_xdecref_func, {self_or_null});
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
                emit_debug_trace(instr.offset, "YIELD_VALUE (before)", stack.size());
                
                // This is the core of generator support!
                // We must spill the remaining stack to persistent storage before yielding
                if (!stack.empty())
                {
                    llvm::Value *yield_val = stack.back();
                    stack.pop_back();
                    
                    // Debug: trace the value being yielded
                    emit_debug_trace(instr.offset, "YIELD_VALUE yielding", stack.size(), yield_val);
                    
                    // SPILL STACK: Save remaining stack values to locals[stack_base + j]
                    // This persists them across the yield/resume boundary
                    for (size_t j = 0; j < stack.size(); ++j)
                    {
                        llvm::Value *val = stack[j];
                        // Use XINCREF to handle NULL values safely (e.g., from LOAD_FAST_AND_CLEAR)
                        builder.CreateCall(py_xincref_func, {val});
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
                            builder.CreateCall(py_xincref_func, {restored_val});
                            stack.push_back(restored_val);
                            
                            // Decref the stored reference and clear the slot
                            builder.CreateCall(py_xdecref_func, {restored_val});
                            builder.CreateStore(
                                llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                                slot_ptr);
                        }
                        
                        // Push the sent value onto the stack
                        builder.CreateCall(py_xincref_func, {sent_value});
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
                    builder.CreateCall(py_xincref_func, {py_none});
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
                    builder.CreateCall(py_xincref_func, {py_obj});
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
                        builder.CreateCall(py_xincref_func, {val});
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
                        builder.CreateCall(py_xincref_func, {val});
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
                        builder.CreateCall(py_xincref_func, {val});
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
                    builder.CreateCall(py_xdecref_func, {cond});

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
                        builder.CreateCall(py_xincref_func, {val});
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
                    builder.CreateCall(py_xdecref_func, {cond});

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
                        builder.CreateCall(py_xincref_func, {val});
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
                emit_debug_trace(instr.offset, "FOR_ITER (before)", stack.size());
                
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
                        builder.CreateCall(py_xincref_func, {val});
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
                emit_debug_trace(instr.offset, "END_FOR (before)", stack.size());
                
                // END_FOR: In Python 3.13, only pops the iterator (stack effect -1)
                // FOR_ITER jumps directly to END_FOR on exhaustion, no NULL is pushed
                if (!stack.empty())
                {
                    llvm::Value *iter = stack.back();
                    // Show what we're popping
                    emit_debug_trace(instr.offset, "END_FOR popping", stack.size(), iter);
                    
                    stack.pop_back();
                    builder.CreateCall(py_xdecref_func, {iter});
                    
                    emit_debug_trace(instr.offset, "END_FOR (after)", stack.size());
                }
            }
            else if (instr.opcode == op::GET_ITER)
            {
                if (!stack.empty())
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();
                    llvm::Value *iter = builder.CreateCall(py_object_getiter_func, {obj});
                    builder.CreateCall(py_xdecref_func, {obj});
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

                    builder.CreateCall(py_xdecref_func, {val});

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

                    builder.CreateCall(py_xdecref_func, {val});

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
                    builder.CreateCall(py_xincref_func, {val});
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
                        module->getOrInsertFunction("JITGetAwaitable", helper_type).getCallee());
                    
                    llvm::Value *awaitable = builder.CreateCall(get_awaitable_helper, {obj});
                    
                    // Decref the original object if different
                    builder.CreateCall(py_xdecref_func, {obj});
                    
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
                    builder.CreateCall(py_xdecref_func, {value});
                    
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
                    
                    // Handle PYGEN_RETURN - sub-iterator finished
                    // We handle END_SEND cleanup inline here rather than jumping to it,
                    // because the stack state differs between PYGEN_RETURN and normal path
                    builder.SetInsertPoint(return_block);
                    
                    // For PYGEN_RETURN: receiver is still on stack, result is the return value
                    // Decref receiver (sub-iterator is done)
                    builder.CreateCall(py_xdecref_func, {receiver});
                    
                    // Pop receiver from compile-time stack (we're in return_block, 
                    // but this affects the logical stack state for the target)
                    // After this, result will be the top of stack
                    
                    // Find the instruction AFTER END_SEND to jump to
                    // END_SEND is at offset 'target', we need the next instruction
                    int target = instr.argval;  // Jump target (END_SEND offset)
                    int after_end_send = target + 2;  // Next instruction offset
                    
                    // Store result to the stack persistence slot (replacing receiver)
                    // The compile-time stack currently has: [..., receiver]
                    // receiver is at index stack.size() - 1
                    // After PYGEN_RETURN, END_SEND would leave: [..., result]
                    // So we store at receiver's position
                    size_t receiver_slot = stack.size() - 1;
                    llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, stack_base + receiver_slot);
                    llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                    builder.CreateCall(py_xincref_func, {result});
                    builder.CreateStore(result, slot_ptr);
                    
                    // Create/get block for the instruction after END_SEND
                    if (!offset_blocks.count(after_end_send))
                    {
                        offset_blocks[after_end_send] = llvm::BasicBlock::Create(
                            *local_context, "after_end_send_" + std::to_string(after_end_send), func);
                    }
                    
                    // Record the expected stack depth at the target
                    // After END_SEND: receiver is gone, result is on top
                    // Stack depth is same as before SEND started the loop (receiver position)
                    if (!target_stack_depth.count(after_end_send))
                    {
                        target_stack_depth[after_end_send] = stack.size();  // result replaces receiver
                    }
                    
                    builder.CreateBr(offset_blocks[after_end_send]);
                    
                    // Handle error
                    builder.SetInsertPoint(error_block);
                    // Clean up receiver before returning error
                    builder.CreateCall(py_xdecref_func, {receiver});
                    // NOTE: Don't modify compile-time stack here - error path returns immediately
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
                    builder.CreateCall(py_xdecref_func, {receiver});
                    
                    // Push result back
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CLEANUP_THROW)
            {
                // CLEANUP_THROW: Handles an exception raised during throw()/close()
                // If STACK[-1] is StopIteration, pop 3 values and push its .value
                // Otherwise, re-raise STACK[-1]
                // STACK: [..., exc_type_or_val?, sub_iter, exc] -> [..., value] or reraise
                if (stack.size() >= 3)
                {
                    llvm::Value *exc = stack.back();
                    stack.pop_back();
                    llvm::Value *sub_iter = stack.back();
                    stack.pop_back();
                    llvm::Value *last_sent_val = stack.back();
                    stack.pop_back();
                    
                    // Check if exc is a StopIteration
                    // Get the StopIteration type
                    llvm::Value *stop_iteration_type = llvm::ConstantInt::get(i64_type, 
                        reinterpret_cast<uint64_t>(PyExc_StopIteration));
                    llvm::Value *stop_iter_ptr = builder.CreateIntToPtr(stop_iteration_type, ptr_type);
                    
                    // Get the type of exc
                    llvm::Value *exc_type = builder.CreateCall(py_object_type_func, {exc}, "exc_type");
                    
                    // Check if it's a StopIteration instance
                    llvm::Value *is_stop_iter = builder.CreateCall(py_exception_matches_func,
                        {exc_type, stop_iter_ptr}, "is_stop_iter");
                    builder.CreateCall(py_xdecref_func, {exc_type});
                    
                    llvm::BasicBlock *stop_iter_block = llvm::BasicBlock::Create(
                        *local_context, "cleanup_throw_stop_" + std::to_string(i), func);
                    llvm::BasicBlock *reraise_block = llvm::BasicBlock::Create(
                        *local_context, "cleanup_throw_reraise_" + std::to_string(i), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "cleanup_throw_cont_" + std::to_string(i), func);
                    
                    llvm::Value *is_match = builder.CreateICmpNE(is_stop_iter,
                        llvm::ConstantInt::get(i32_type, 0));
                    builder.CreateCondBr(is_match, stop_iter_block, reraise_block);
                    
                    // Handle StopIteration - extract .value and push it
                    builder.SetInsertPoint(stop_iter_block);
                    // Get the .value attribute from StopIteration
                    llvm::Value *value_attr = builder.CreateCall(
                        py_object_getattr_func, 
                        {exc, builder.CreateGlobalStringPtr("value")},
                        "stop_iter_value");
                    // If .value is NULL, use Py_None
                    llvm::Value *is_null = builder.CreateICmpEQ(value_attr, 
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    // Clear any error from failed attribute access
                    builder.CreateCall(py_err_clear_func, {});
                    llvm::Value *py_none = builder.CreateIntToPtr(
                        llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None)), ptr_type);
                    llvm::Value *result_val = builder.CreateSelect(is_null, py_none, value_attr);
                    builder.CreateCall(py_xincref_func, {result_val});
                    // Decref the exception and sub_iter and last_sent_val
                    builder.CreateCall(py_xdecref_func, {exc});
                    builder.CreateCall(py_xdecref_func, {sub_iter});
                    builder.CreateCall(py_xdecref_func, {last_sent_val});
                    builder.CreateBr(continue_block);
                    
                    // Handle reraise - restore exception and return NULL
                    builder.SetInsertPoint(reraise_block);
                    // Re-raise the exception by restoring it and returning NULL
                    builder.CreateCall(py_err_restore_func, {
                        builder.CreateCall(py_object_type_func, {exc}),
                        exc,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0))
                    });
                    builder.CreateCall(py_xdecref_func, {sub_iter});
                    builder.CreateCall(py_xdecref_func, {last_sent_val});
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    
                    // Continue with the result
                    builder.SetInsertPoint(continue_block);
                    current_block = continue_block;
                    stack.push_back(result_val);
                }
            }
            // ========== Async Iteration Opcodes (for async for loops) ==========
            else if (instr.opcode == op::GET_AITER)
            {
                // GET_AITER: Get async iterator from object
                // STACK: [..., obj] -> [..., aiter]
                // Implements: STACK[-1] = STACK[-1].__aiter__()
                if (!stack.empty())
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();
                    
                    // Call our JITGetAIter helper (wraps PyObject_GetAIter)
                    llvm::Value *aiter = builder.CreateCall(jit_get_aiter_func, {obj}, "aiter");
                    
                    // Decref original object
                    builder.CreateCall(py_xdecref_func, {obj});
                    
                    // Check for error (NULL return)
                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                        *local_context, "get_aiter_error_" + std::to_string(i), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "get_aiter_cont_" + std::to_string(i), func);
                    
                    llvm::Value *is_null = builder.CreateICmpEQ(aiter,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    builder.CreateCondBr(is_null, error_block, continue_block);
                    
                    // Error path - return NULL to propagate exception
                    builder.SetInsertPoint(error_block);
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    
                    // Continue path
                    builder.SetInsertPoint(continue_block);
                    current_block = continue_block;
                    stack.push_back(aiter);
                }
            }
            else if (instr.opcode == op::GET_ANEXT)
            {
                // GET_ANEXT: Get next awaitable from async iterator
                // STACK: [..., aiter] -> [..., aiter, awaitable]
                // Implements: STACK.append(get_awaitable(STACK[-1].__anext__()))
                if (!stack.empty())
                {
                    llvm::Value *aiter = stack.back();
                    // Don't pop - aiter stays on stack for next iteration
                    
                    // Call our JITGetANext helper (calls __anext__, returns awaitable)
                    llvm::Value *awaitable = builder.CreateCall(jit_get_anext_func, {aiter}, "anext");
                    
                    // Check for error (NULL return)
                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                        *local_context, "get_anext_error_" + std::to_string(i), func);
                    llvm::BasicBlock *continue_block = llvm::BasicBlock::Create(
                        *local_context, "get_anext_cont_" + std::to_string(i), func);
                    
                    llvm::Value *is_null = builder.CreateICmpEQ(awaitable,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    builder.CreateCondBr(is_null, error_block, continue_block);
                    
                    // Error path - check if it's StopAsyncIteration
                    builder.SetInsertPoint(error_block);
                    // For now, just propagate the error
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    
                    // Continue path
                    builder.SetInsertPoint(continue_block);
                    current_block = continue_block;
                    stack.push_back(awaitable);
                }
            }
            else if (instr.opcode == op::END_ASYNC_FOR)
            {
                // END_ASYNC_FOR: End of async for loop iteration
                // STACK: [..., aiter, exc] -> [...]
                // If exc is StopAsyncIteration, clear it and continue
                // Otherwise, re-raise the exception
                if (stack.size() >= 2)
                {
                    llvm::Value *exc = stack.back();
                    stack.pop_back();
                    llvm::Value *aiter = stack.back();
                    stack.pop_back();
                    
                    // Call our JITEndAsyncFor helper
                    llvm::Value *result = builder.CreateCall(jit_end_async_for_func, {exc}, "end_async_for");
                    
                    // Decref both values
                    builder.CreateCall(py_xdecref_func, {exc});
                    builder.CreateCall(py_xdecref_func, {aiter});
                    
                    // Check result: 1 = success (StopAsyncIteration), 0 = error (propagate)
                    llvm::BasicBlock *success_block = llvm::BasicBlock::Create(
                        *local_context, "end_async_success_" + std::to_string(i), func);
                    llvm::BasicBlock *error_block = llvm::BasicBlock::Create(
                        *local_context, "end_async_error_" + std::to_string(i), func);
                    
                    llvm::Value *is_success = builder.CreateICmpNE(result,
                        llvm::ConstantInt::get(i32_type, 0));
                    builder.CreateCondBr(is_success, success_block, error_block);
                    
                    // Error path - propagate exception
                    builder.SetInsertPoint(error_block);
                    builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                    builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                    
                    // Success path - loop ends normally
                    builder.SetInsertPoint(success_block);
                    current_block = success_block;
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

                    builder.CreateCall(py_xdecref_func, {actual_type});
                    builder.CreateCall(py_xdecref_func, {exc_type});

                    llvm::Value *is_match = builder.CreateICmpNE(match_result,
                                                                 llvm::ConstantInt::get(i32_type, 0), "is_match");
                    llvm::Value *py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value *py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value *py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value *py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);

                    llvm::Value *result = builder.CreateSelect(is_match, py_true, py_false, "match_bool");
                    builder.CreateCall(py_xincref_func, {result});
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
                        builder.CreateCall(py_xdecref_func, {arg});
                        {
                            llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(Py_None));
                            result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                            builder.CreateCall(py_xincref_func, {result});
                        }
                        break;
                    case 3: // INTRINSIC_STOPITERATION_ERROR
                        // Handle StopIteration - just decref and push None
                        builder.CreateCall(py_xdecref_func, {arg});
                        {
                            llvm::Value *py_none_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(Py_None));
                            result = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                            builder.CreateCall(py_xincref_func, {result});
                        }
                        break;
                    case 4: // INTRINSIC_ASYNC_GEN_WRAP
                        // For async generators - just pass through for now
                        result = arg;  // Transfer ownership
                        break;
                    case 5: // INTRINSIC_UNARY_POSITIVE
                        result = builder.CreateCall(py_number_positive_func, {arg});
                        builder.CreateCall(py_xdecref_func, {arg});
                        check_error_and_branch_gen(instr.offset, result, "unary_positive");
                        break;
                    case 6: // INTRINSIC_LIST_TO_TUPLE
                    {
                        llvm::FunctionType *list_as_tuple_type = llvm::FunctionType::get(
                            ptr_type, {ptr_type}, false);
                        llvm::FunctionCallee list_as_tuple_func = module->getOrInsertFunction(
                            "PyList_AsTuple", list_as_tuple_type);
                        result = builder.CreateCall(list_as_tuple_func, {arg});
                        builder.CreateCall(py_xdecref_func, {arg});
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
                        builder.CreateCall(py_xdecref_func, {arg});
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
                        
                        builder.CreateCall(py_xdecref_func, {typevar_class});
                        builder.CreateCall(py_xdecref_func, {typing_mod});
                        builder.CreateCall(py_xdecref_func, {arg});
                        
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
                        
                        builder.CreateCall(py_xdecref_func, {paramspec_class});
                        builder.CreateCall(py_xdecref_func, {typing_mod});
                        builder.CreateCall(py_xdecref_func, {arg});
                        
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
                        
                        builder.CreateCall(py_xdecref_func, {typevartuple_class});
                        builder.CreateCall(py_xdecref_func, {typing_mod});
                        builder.CreateCall(py_xdecref_func, {arg});
                        
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
                        
                        builder.CreateCall(py_xdecref_func, {typealias_class});
                        builder.CreateCall(py_xdecref_func, {typing_mod});
                        builder.CreateCall(py_xdecref_func, {arg});
                        
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
                        
                        builder.CreateCall(py_xdecref_func, {mod_dict});
                        builder.CreateCall(py_xdecref_func, {locals});
                        builder.CreateCall(py_xdecref_func, {arg});
                        
                        llvm::FunctionType *err_clear_type = llvm::FunctionType::get(
                            llvm::Type::getVoidTy(*local_context), {}, false);
                        llvm::FunctionCallee err_clear_func = module->getOrInsertFunction(
                            "PyErr_Clear", err_clear_type);
                        builder.CreateCall(err_clear_func, {});
                        
                        llvm::Value *py_none_ptr_ret = llvm::ConstantInt::get(
                            i64_type, reinterpret_cast<uint64_t>(Py_None));
                        result = builder.CreateIntToPtr(py_none_ptr_ret, ptr_type);
                        builder.CreateCall(py_xincref_func, {result});
                        break;
                    }
                    default:
                    {
                        // Unknown intrinsic - raise error and transition to error state
                        builder.CreateCall(py_xdecref_func, {arg});
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
            // ========== BUILD_MAP ==========
            else if (instr.opcode == op::BUILD_MAP)
            {
                // Build a dictionary from arg key-value pairs
                int count = instr.arg;
                llvm::Value *new_dict = builder.CreateCall(py_dict_new_func, {}, "new_dict");

                std::vector<std::pair<llvm::Value *, llvm::Value *>> pairs;
                for (int j = 0; j < count; ++j)
                {
                    if (stack.size() >= 2)
                    {
                        llvm::Value *value = stack.back();
                        stack.pop_back();
                        llvm::Value *key = stack.back();
                        stack.pop_back();
                        pairs.push_back({key, value});
                    }
                }

                for (int j = count - 1; j >= 0; --j)
                {
                    llvm::Value *key = pairs[j].first;
                    llvm::Value *value = pairs[j].second;
                    builder.CreateCall(py_dict_setitem_func, {new_dict, key, value});
                    builder.CreateCall(py_xdecref_func, {key});
                    builder.CreateCall(py_xdecref_func, {value});
                }

                stack.push_back(new_dict);
            }
            // ========== BUILD_SET ==========
            else if (instr.opcode == op::BUILD_SET)
            {
                int count = instr.arg;
                llvm::Value *null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                llvm::Value *new_set = builder.CreateCall(py_set_new_func, {null_ptr}, "new_set");

                std::vector<llvm::Value *> items;
                for (int j = 0; j < count; ++j)
                {
                    if (!stack.empty())
                    {
                        items.push_back(stack.back());
                        stack.pop_back();
                    }
                }

                for (int j = count - 1; j >= 0; --j)
                {
                    llvm::Value *item = items[j];
                    builder.CreateCall(py_set_add_func, {new_set, item});
                    builder.CreateCall(py_xdecref_func, {item});
                }

                stack.push_back(new_set);
            }
            // ========== UNPACK_SEQUENCE ==========
            else if (instr.opcode == op::UNPACK_SEQUENCE)
            {
                int count = instr.arg;
                if (!stack.empty())
                {
                    llvm::Value *sequence = stack.back();
                    stack.pop_back();

                    std::vector<llvm::Value *> unpacked;
                    for (int j = 0; j < count; ++j)
                    {
                        llvm::Value *idx = llvm::ConstantInt::get(i64_type, j);
                        llvm::Value *item = builder.CreateCall(py_sequence_getitem_func, {sequence, idx}, "unpack_item");
                        check_error_and_branch_gen(instr.offset, item, "unpack_seq");
                        unpacked.push_back(item);
                    }

                    for (int j = count - 1; j >= 0; --j)
                    {
                        stack.push_back(unpacked[j]);
                    }

                    builder.CreateCall(py_xdecref_func, {sequence});
                }
            }
            // ========== UNPACK_EX ==========
            else if (instr.opcode == op::UNPACK_EX)
            {
                int count_before = instr.arg & 0xFF;
                int count_after = (instr.arg >> 8) & 0xFF;

                if (!stack.empty())
                {
                    llvm::Value *sequence = stack.back();
                    stack.pop_back();

                    llvm::Value *seq_len = builder.CreateCall(py_sequence_size_func, {sequence}, "seq_len");

                    std::vector<llvm::Value *> before_items;
                    for (int j = 0; j < count_before; ++j)
                    {
                        llvm::Value *idx = llvm::ConstantInt::get(i64_type, j);
                        llvm::Value *item = builder.CreateCall(py_sequence_getitem_func, {sequence, idx}, "before_item");
                        check_error_and_branch_gen(instr.offset, item, "unpack_ex_before");
                        before_items.push_back(item);
                    }

                    std::vector<llvm::Value *> after_items;
                    for (int j = count_after; j > 0; --j)
                    {
                        llvm::Value *neg_idx = llvm::ConstantInt::get(i64_type, -static_cast<int64_t>(j));
                        llvm::Value *item = builder.CreateCall(py_sequence_getitem_func, {sequence, neg_idx}, "after_item");
                        check_error_and_branch_gen(instr.offset, item, "unpack_ex_after");
                        after_items.push_back(item);
                    }

                    llvm::Value *middle_start = llvm::ConstantInt::get(i64_type, count_before);
                    llvm::Value *after_count_val = llvm::ConstantInt::get(i64_type, count_after);
                    llvm::Value *middle_end = builder.CreateSub(seq_len, after_count_val, "middle_end");
                    llvm::Value *middle_list = builder.CreateCall(py_sequence_getslice_func,
                                                                  {sequence, middle_start, middle_end}, "middle_list");
                    check_error_and_branch_gen(instr.offset, middle_list, "unpack_ex_middle");

                    for (int j = static_cast<int>(after_items.size()) - 1; j >= 0; --j)
                    {
                        stack.push_back(after_items[j]);
                    }
                    stack.push_back(middle_list);
                    for (int j = static_cast<int>(before_items.size()) - 1; j >= 0; --j)
                    {
                        stack.push_back(before_items[j]);
                    }

                    builder.CreateCall(py_xdecref_func, {sequence});
                }
            }
            // ========== CALL_KW ==========
            else if (instr.opcode == op::CALL_KW)
            {
                int num_args = instr.arg;
                if (stack.size() >= static_cast<size_t>(num_args + 3))
                {
                    llvm::Value *kwnames = stack.back();
                    stack.pop_back();

                    size_t base = stack.size() - num_args - 2;
                    llvm::Value *callable = stack[base];
                    llvm::Value *self_or_null = stack[base + 1];

                    std::vector<llvm::Value *> args;
                    for (int ai = 0; ai < num_args; ++ai)
                    {
                        args.push_back(stack[base + 2 + ai]);
                    }
                    stack.erase(stack.begin() + base, stack.end());

                    llvm::ArrayType *args_array_type = llvm::ArrayType::get(ptr_type, num_args);
                    llvm::Value *args_array = builder.CreateAlloca(args_array_type, nullptr, "args_array");

                    for (int ai = 0; ai < num_args; ++ai)
                    {
                        llvm::Value *indices[] = {
                            llvm::ConstantInt::get(i64_type, 0),
                            llvm::ConstantInt::get(i64_type, ai)};
                        llvm::Value *elem_ptr = builder.CreateGEP(args_array_type, args_array, indices, "arg_ptr");
                        builder.CreateStore(args[ai], elem_ptr);
                    }

                    llvm::Value *first_indices[] = {
                        llvm::ConstantInt::get(i64_type, 0),
                        llvm::ConstantInt::get(i64_type, 0)};
                    llvm::Value *args_ptr = builder.CreateGEP(args_array_type, args_array, first_indices, "args_ptr");

                    llvm::Value *nargs_val = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value *result = builder.CreateCall(jit_call_with_kwargs_func,
                                                             {callable, args_ptr, nargs_val, kwnames}, "call_kw_result");

                    builder.CreateCall(py_xdecref_func, {kwnames});
                    builder.CreateCall(py_xdecref_func, {callable});

                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self_kw_gen", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self_kw_gen", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_xdecref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);
                    current_block = after_decref_self;

                    check_error_and_branch_gen(instr.offset, result, "call_kw");
                    stack.push_back(result);
                }
            }
            // ========== CALL_FUNCTION_EX ==========
            else if (instr.opcode == op::CALL_FUNCTION_EX)
            {
                bool has_kwargs = (instr.arg & 1) != 0;
                size_t required = has_kwargs ? 4 : 3;

                if (stack.size() >= required)
                {
                    llvm::Value *kwargs = nullptr;
                    if (has_kwargs)
                    {
                        kwargs = stack.back();
                        stack.pop_back();
                    }

                    llvm::Value *args_seq = stack.back();
                    stack.pop_back();
                    llvm::Value *self_or_null = stack.back();
                    stack.pop_back();
                    llvm::Value *callable = stack.back();
                    stack.pop_back();

                    llvm::Value *args_tuple = builder.CreateCall(py_sequence_tuple_func, {args_seq}, "args_as_tuple");
                    builder.CreateCall(py_xdecref_func, {args_seq});

                    llvm::Value *kwargs_arg = kwargs;
                    if (!kwargs_arg)
                    {
                        kwargs_arg = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    }

                    llvm::Value *result = builder.CreateCall(py_object_call_func,
                                                             {callable, args_tuple, kwargs_arg}, "call_ex_result");

                    builder.CreateCall(py_xdecref_func, {args_tuple});
                    if (has_kwargs && kwargs)
                    {
                        builder.CreateCall(py_xdecref_func, {kwargs});
                    }
                    builder.CreateCall(py_xdecref_func, {callable});

                    llvm::Value *null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value *has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self_ex");

                    llvm::BasicBlock *decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self_ex_gen", func);
                    llvm::BasicBlock *after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self_ex_gen", func);

                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);

                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_xdecref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);

                    builder.SetInsertPoint(after_decref_self);
                    current_block = after_decref_self;

                    check_error_and_branch_gen(instr.offset, result, "call_function_ex");
                    stack.push_back(result);
                }
            }
            // ========== STORE_GLOBAL ==========
            else if (instr.opcode == op::STORE_GLOBAL)
            {
                int name_idx = instr.arg;
                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    llvm::Value *name_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "name_obj");

                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals_dict = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals_dict");

                    builder.CreateCall(py_dict_setitem_func, {globals_dict, name_obj, value});
                    builder.CreateCall(py_xdecref_func, {value});
                }
            }
            // ========== STORE_ATTR ==========
            else if (instr.opcode == op::STORE_ATTR)
            {
                int name_idx = instr.arg;
                if (stack.size() >= 2 && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *obj = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);

                    builder.CreateCall(py_object_setattr_func, {obj, attr_name, value});
                    builder.CreateCall(py_xdecref_func, {obj});
                    builder.CreateCall(py_xdecref_func, {value});
                }
            }
            // ========== LIST_APPEND ==========
            else if (instr.opcode == op::LIST_APPEND)
            {
                emit_debug_trace(instr.offset, "LIST_APPEND (before)", stack.size());
                
                if (!stack.empty())
                {
                    llvm::Value *item = stack.back();
                    stack.pop_back();

                    int list_index = instr.arg;
                    if (list_index > 0 && static_cast<size_t>(list_index) <= stack.size())
                    {
                        llvm::Value *list = stack[stack.size() - list_index];
                        
                        // Debug: trace the list before append
                        emit_debug_trace(instr.offset, "LIST_APPEND list", stack.size(), list);
                        emit_debug_trace(instr.offset, "LIST_APPEND item", stack.size(), item);
                        
                        builder.CreateCall(py_list_append_func, {list, item});
                        builder.CreateCall(py_xdecref_func, {item});
                        
                        emit_debug_trace(instr.offset, "LIST_APPEND (after)", stack.size());
                    }
                }
            }
            // ========== SET_ADD ==========
            else if (instr.opcode == op::SET_ADD)
            {
                if (!stack.empty())
                {
                    llvm::Value *item = stack.back();
                    stack.pop_back();

                    int set_index = instr.arg;
                    if (set_index > 0 && static_cast<size_t>(set_index) <= stack.size())
                    {
                        llvm::Value *set = stack[stack.size() - set_index];
                        builder.CreateCall(py_set_add_func, {set, item});
                        builder.CreateCall(py_xdecref_func, {item});
                    }
                }
            }
            // ========== MAP_ADD ==========
            else if (instr.opcode == op::MAP_ADD)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();
                    llvm::Value *key = stack.back();
                    stack.pop_back();

                    int map_index = instr.arg;
                    if (map_index > 0 && static_cast<size_t>(map_index) <= stack.size())
                    {
                        llvm::Value *dict = stack[stack.size() - map_index];
                        builder.CreateCall(py_dict_setitem_func, {dict, key, value});
                        builder.CreateCall(py_xdecref_func, {key});
                        builder.CreateCall(py_xdecref_func, {value});
                    }
                }
            }
            // ========== DELETE_SUBSCR ==========
            else if (instr.opcode == op::DELETE_SUBSCR)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *key = stack.back();
                    stack.pop_back();
                    llvm::Value *container = stack.back();
                    stack.pop_back();

                    builder.CreateCall(py_object_delitem_func, {container, key});
                    builder.CreateCall(py_xdecref_func, {key});
                    builder.CreateCall(py_xdecref_func, {container});
                }
            }
            // ========== COPY_FREE_VARS ==========
            else if (instr.opcode == op::COPY_FREE_VARS)
            {
                int num_free_vars = instr.arg;
                for (int j = 0; j < num_free_vars && j < static_cast<int>(closure_cells.size()); ++j)
                {
                    if (closure_cells[j] != nullptr)
                    {
                        int slot = nlocals + j;
                        llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, slot);
                        llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                        llvm::Value *cell_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(closure_cells[j]));
                        llvm::Value *cell_obj = builder.CreateIntToPtr(cell_ptr, ptr_type);
                        builder.CreateStore(cell_obj, slot_ptr);
                    }
                }
            }
            // ========== LOAD_DEREF ==========
            else if (instr.opcode == op::LOAD_DEREF)
            {
                int slot = instr.arg;
                llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, slot);
                llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                llvm::Value *cell = builder.CreateLoad(ptr_type, slot_ptr, "load_cell");
                llvm::Value *contents = builder.CreateCall(py_cell_get_func, {cell}, "cell_contents");
                check_error_and_branch_gen(instr.offset, contents, "load_deref");
                stack.push_back(contents);
            }
            // ========== STORE_DEREF ==========
            else if (instr.opcode == op::STORE_DEREF)
            {
                int slot = instr.arg;
                if (!stack.empty())
                {
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, slot);
                    llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                    llvm::Value *cell = builder.CreateLoad(ptr_type, slot_ptr, "store_cell");
                    builder.CreateCall(py_cell_set_func, {cell, value});
                }
            }
            // ========== MAKE_CELL ==========
            else if (instr.opcode == op::MAKE_CELL)
            {
                int slot = instr.arg;
                
                // Get or create PyCell_New function
                llvm::FunctionType *py_cell_new_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
                llvm::FunctionCallee py_cell_new_func_local = module->getOrInsertFunction("PyCell_New", py_cell_new_type);
                
                // Load current value from slot (may be parameter)
                llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, slot);
                llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                llvm::Value *initial_value = builder.CreateLoad(ptr_type, slot_ptr, "initial_cell_value");
                
                // Handle NULL initial value
                llvm::Value *is_null = builder.CreateICmpEQ(initial_value,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));
                llvm::Value *cell_value = builder.CreateSelect(is_null,
                    llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                    initial_value);
                
                llvm::Value *new_cell = builder.CreateCall(py_cell_new_func_local, {cell_value}, "new_cell");
                builder.CreateStore(new_cell, slot_ptr);
            }
            // ========== LOAD_CLOSURE ==========
            else if (instr.opcode == op::LOAD_CLOSURE)
            {
                int slot = instr.arg;
                llvm::Value *slot_idx = llvm::ConstantInt::get(i64_type, slot);
                llvm::Value *slot_ptr = builder.CreateGEP(ptr_type, locals_array, slot_idx);
                llvm::Value *cell = builder.CreateLoad(ptr_type, slot_ptr, "load_closure");
                builder.CreateCall(py_xincref_func, {cell});
                stack.push_back(cell);
            }
            // ========== IMPORT_NAME ==========
            else if (instr.opcode == op::IMPORT_NAME)
            {
                int name_idx = instr.arg;
                if (stack.size() >= 2 && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *fromlist = stack.back();
                    stack.pop_back();
                    llvm::Value *level_obj = stack.back();
                    stack.pop_back();

                    llvm::Value *name_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *name = builder.CreateIntToPtr(name_ptr, ptr_type, "module_name");

                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals = builder.CreateIntToPtr(globals_ptr_val, ptr_type, "globals");

                    llvm::Value *locals_null = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));

                    llvm::Value *level_int = builder.CreateCall(py_long_aslong_func, {level_obj});
                    llvm::Value *level_trunc = builder.CreateTrunc(level_int, i32_type);
                    builder.CreateCall(py_xdecref_func, {level_obj});

                    llvm::Value *module = builder.CreateCall(py_import_importmodule_func,
                        {name, globals, locals_null, fromlist, level_trunc}, "imported_module");

                    builder.CreateCall(py_xdecref_func, {fromlist});
                    check_error_and_branch_gen(instr.offset, module, "import_name");
                    stack.push_back(module);
                }
            }
            // ========== IMPORT_FROM ==========
            else if (instr.opcode == op::IMPORT_FROM)
            {
                int name_idx = instr.arg;
                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size()))
                {
                    llvm::Value *module = stack.back();

                    llvm::Value *attr_name_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(name_objects[name_idx]));
                    llvm::Value *attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type, "attr_name");

                    llvm::Value *attr = builder.CreateCall(py_object_getattr_func, {module, attr_name}, "imported_attr");
                    check_error_and_branch_gen(instr.offset, attr, "import_from");
                    stack.push_back(attr);
                }
            }
            // ========== MAKE_FUNCTION ==========
            else if (instr.opcode == op::MAKE_FUNCTION)
            {
                if (!stack.empty())
                {
                    llvm::Value *code_obj = stack.back();
                    stack.pop_back();

                    llvm::Value *globals_ptr_val = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(globals_dict_ptr));
                    llvm::Value *globals = builder.CreateIntToPtr(globals_ptr_val, ptr_type);

                    llvm::Value *func_obj = builder.CreateCall(py_function_new_func, {code_obj, globals});
                    builder.CreateCall(py_xdecref_func, {code_obj});
                    check_error_and_branch_gen(instr.offset, func_obj, "make_function");
                    stack.push_back(func_obj);
                }
            }
            // ========== SET_FUNCTION_ATTRIBUTE ==========
            else if (instr.opcode == op::SET_FUNCTION_ATTRIBUTE)
            {
                if (stack.size() >= 2)
                {
                    llvm::Value *py_func = stack.back();
                    stack.pop_back();
                    llvm::Value *value = stack.back();
                    stack.pop_back();

                    int flag = instr.arg;
                    llvm::Value *result = nullptr;

                    if (flag == 0x01)
                    {
                        result = builder.CreateCall(py_function_set_defaults_func, {py_func, value});
                    }
                    else if (flag == 0x02)
                    {
                        result = builder.CreateCall(py_function_set_kwdefaults_func, {py_func, value});
                    }
                    else if (flag == 0x04)
                    {
                        result = builder.CreateCall(py_function_set_annotations_func, {py_func, value});
                    }
                    else if (flag == 0x08)
                    {
                        result = builder.CreateCall(py_function_set_closure_func, {py_func, value});
                    }

                    if (result)
                    {
                        llvm::Value *is_error = builder.CreateICmpSLT(result, llvm::ConstantInt::get(i32_type, 0));
                        llvm::BasicBlock *error_bb = llvm::BasicBlock::Create(*local_context, "set_func_attr_error_gen", func);
                        llvm::BasicBlock *continue_bb = llvm::BasicBlock::Create(*local_context, "set_func_attr_continue_gen", func);
                        builder.CreateCondBr(is_error, error_bb, continue_bb);

                        builder.SetInsertPoint(error_bb);
                        builder.CreateCall(py_xdecref_func, {py_func});
                        builder.CreateStore(llvm::ConstantInt::get(i32_type, -2), state_ptr);
                        builder.CreateRet(llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)));

                        builder.SetInsertPoint(continue_bb);
                        current_block = continue_bb;
                    }

                    stack.push_back(py_func);
                }
            }
            // ========== LIST_EXTEND ==========
            else if (instr.opcode == op::LIST_EXTEND)
            {
                if (!stack.empty())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();

                    int list_index = instr.arg;
                    if (list_index > 0 && static_cast<size_t>(list_index) <= stack.size())
                    {
                        llvm::Value *list = stack[stack.size() - list_index];
                        builder.CreateCall(py_list_extend_func, {list, iterable});
                        builder.CreateCall(py_xdecref_func, {iterable});
                    }
                }
            }
            // ========== SET_UPDATE ==========
            else if (instr.opcode == op::SET_UPDATE)
            {
                if (!stack.empty())
                {
                    llvm::Value *iterable = stack.back();
                    stack.pop_back();

                    int set_index = instr.arg;
                    if (set_index > 0 && static_cast<size_t>(set_index) <= stack.size())
                    {
                        llvm::Value *set = stack[stack.size() - set_index];
                        builder.CreateCall(py_set_update_func, {set, iterable});
                        builder.CreateCall(py_xdecref_func, {iterable});
                    }
                }
            }
            // ========== DICT_UPDATE ==========
            else if (instr.opcode == op::DICT_UPDATE)
            {
                if (!stack.empty())
                {
                    llvm::Value *update_dict = stack.back();
                    stack.pop_back();

                    int dict_index = instr.arg;
                    if (dict_index > 0 && static_cast<size_t>(dict_index) <= stack.size())
                    {
                        llvm::Value *dict = stack[stack.size() - dict_index];
                        builder.CreateCall(py_dict_update_func, {dict, update_dict});
                        builder.CreateCall(py_xdecref_func, {update_dict});
                    }
                }
            }
            // ========== DICT_MERGE ==========
            else if (instr.opcode == op::DICT_MERGE)
            {
                if (!stack.empty())
                {
                    llvm::Value *update_dict = stack.back();
                    stack.pop_back();

                    int dict_index = instr.arg;
                    if (dict_index > 0 && static_cast<size_t>(dict_index) <= stack.size())
                    {
                        llvm::Value *dict = stack[stack.size() - dict_index];
                        builder.CreateCall(py_dict_merge_func, {dict, update_dict, llvm::ConstantInt::get(i32_type, 1)});
                        builder.CreateCall(py_xdecref_func, {update_dict});
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
            llvm::errs() << "=== GENERATOR VERIFICATION FAILED ===\n" << verify_err << "\n=== END ERROR ===\n";
            llvm::errs().flush();
            return false;  // Return false on verification failure
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
        
        // Use the actual computed total_locals from compilation if available
        int actual_total_locals = total_locals;
        auto it = generator_total_locals.find(name);
        if (it != generator_total_locals.end()) {
            actual_total_locals = it->second;
        }
        Py_ssize_t num_locals = static_cast<Py_ssize_t>(actual_total_locals);

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
            const char* awaiting_type = Py_TYPE(coro->awaiting)->tp_name;
            bool is_gen_or_coro = (strcmp(awaiting_type, "generator") == 0 || 
                                   strcmp(awaiting_type, "coroutine") == 0);
            if (is_gen_or_coro) {
                // Native coroutine or generator - use send
                PyObject* send_meth = PyObject_GetAttrString(coro->awaiting, "send");
                if (send_meth != NULL) {
                    result = PyObject_CallFunctionObjArgs(send_meth, value, NULL);
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

// =========================================================================
// Inline C Compiler Implementation
// =========================================================================
#ifdef JUSTJIT_HAS_CLANG

    // RAII wrapper for clang diagnostics (LLVM 18+ API)
    class DiagnosticsRAII {
    public:
        DiagnosticsRAII() 
            : diag_opts_(new clang::DiagnosticOptions()),
              diag_printer_(new clang::TextDiagnosticPrinter(llvm::errs(), diag_opts_.get())),
              diag_id_(new clang::DiagnosticIDs()),
              diags_(diag_id_, diag_opts_, diag_printer_.get(), false)
        {
        }

        ~DiagnosticsRAII() {
            // Resources cleaned up automatically
        }

        clang::DiagnosticsEngine& get() { return diags_; }

    private:
        llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_opts_;
        std::unique_ptr<clang::TextDiagnosticPrinter> diag_printer_;
        llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_id_;
        clang::DiagnosticsEngine diags_;
    };

    // =========================================================================
    // JITCallable - Native Python callable for JIT-compiled C functions
    // =========================================================================
    // Uses Python C API directly instead of nanobind for pure native callable
    // =========================================================================
    
    // Signature type encoding - supports multiple integer sizes
    enum class JITCallableReturnType { INT64, INT32, DOUBLE, VOID, PTR };
    
    // Per-parameter type encoding (4 bits per param, 8 params max = 32 bits)
    // Values: 0=INT64, 1=INT32, 2=DOUBLE, 3=FLOAT, 4=PTR
    enum JITParamTypeCode { 
        PARAM_INT64 = 0,
        PARAM_INT32 = 1, 
        PARAM_DOUBLE = 2, 
        PARAM_FLOAT = 3, 
        PARAM_PTR = 4 
    };
    
    // JITCallable object struct
    struct JITCallableObject {
        PyObject_HEAD
        uint64_t func_ptr;                  // Pointer to JIT-compiled function
        JITCallableReturnType return_type;  // Return type
        int param_count;                    // Number of parameters
        uint32_t param_type_mask;           // Per-param types (4 bits each)
        char* name;                         // Function name (for repr)
        bool is_varargs;                    // For warning purposes
        bool is_struct_ret;                 // For error message
    };

    
    // Forward declarations
    static void JITCallable_dealloc(JITCallableObject* self);
    static PyObject* JITCallable_repr(JITCallableObject* self);
    static PyObject* JITCallable_call(JITCallableObject* self, PyObject* args, PyObject* kwargs);
    
    // Python type object for JIT callables
    static PyTypeObject JITCallable_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "justjit.JITCallable",              // tp_name
        sizeof(JITCallableObject),          // tp_basicsize
        0,                                  // tp_itemsize
        (destructor)JITCallable_dealloc,    // tp_dealloc
        0,                                  // tp_vectorcall_offset
        0,                                  // tp_getattr
        0,                                  // tp_setattr
        0,                                  // tp_as_async
        (reprfunc)JITCallable_repr,         // tp_repr
        0,                                  // tp_as_number
        0,                                  // tp_as_sequence  
        0,                                  // tp_as_mapping
        0,                                  // tp_hash
        (ternaryfunc)JITCallable_call,      // tp_call
        0,                                  // tp_str
        0,                                  // tp_getattro
        0,                                  // tp_setattro
        0,                                  // tp_as_buffer
        Py_TPFLAGS_DEFAULT,                 // tp_flags
        "JIT-compiled C callable",          // tp_doc
    };
    
    // Deallocate callable object
    static void JITCallable_dealloc(JITCallableObject* self) {
        if (self->name) {
            PyMem_Free(self->name);
        }
        Py_TYPE(self)->tp_free((PyObject*)self);
    }
    
    // Repr for callable
    static PyObject* JITCallable_repr(JITCallableObject* self) {
        return PyUnicode_FromFormat("<justjit.JITCallable '%s' at %p>", 
                                    self->name ? self->name : "?", 
                                    (void*)self->func_ptr);
    }
    
    // Call implementation - invokes JIT-compiled function
    static PyObject* JITCallable_call(JITCallableObject* self, PyObject* args, PyObject* kwargs) {
        Py_ssize_t nargs = PyTuple_Size(args);
        if (nargs != self->param_count) {
            PyErr_Format(PyExc_TypeError, "%s() takes %d argument(s) but %zd were given",
                        self->name ? self->name : "function", self->param_count, nargs);
            return NULL;
        }
        
        // Check for unsupported function types
        if (self->is_struct_ret) {
            PyErr_Format(PyExc_NotImplementedError, 
                "%s() returns a struct which is not supported. "
                "Consider returning via pointer parameter instead.",
                self->name ? self->name : "function");
            return NULL;
        }
        
        // Warn about varargs (only on first call could log)
        // For now, varargs will just fail at call time if wrong args used
        
        // Parse arguments based on per-param types from param_type_mask
        // Each param uses 4 bits: 0=INT64, 1=INT32, 2=DOUBLE, 3=FLOAT, 4=PTR
        int64_t iargs[8] = {0};
        double dargs[8] = {0};
        
        for (Py_ssize_t i = 0; i < nargs && i < 8; i++) {
            PyObject* arg = PyTuple_GetItem(args, i);
            int param_type = (self->param_type_mask >> (i * 4)) & 0xF;
            
            // Always populate BOTH arrays
            // For wrappers: store double bits in iargs for proper passing
            if (PyFloat_Check(arg)) {
                dargs[i] = PyFloat_AsDouble(arg);
                // Bit-cast double to int64 for wrapper functions
                memcpy(&iargs[i], &dargs[i], sizeof(double));
            } else if (PyLong_Check(arg)) {
                iargs[i] = PyLong_AsLongLong(arg);
                dargs[i] = (double)iargs[i];
            } else if (arg == Py_None) {
                iargs[i] = 0;
                dargs[i] = 0.0;
            } else {
                // Try capsule for pointer types
                void* ptr = PyCapsule_GetPointer(arg, NULL);
                if (ptr) {
                    iargs[i] = reinterpret_cast<int64_t>(ptr);
                    dargs[i] = 0.0;
                } else {
                    PyErr_Clear();
                    PyErr_Format(PyExc_TypeError, "argument %zd must be int, float, or None", i);
                    return NULL;
                }
            }
        }

        
        // Determine if all params are same type for optimized calling
        bool all_int = true, all_float = true;
        for (int i = 0; i < self->param_count && i < 8; i++) {
            int param_type = (self->param_type_mask >> (i * 4)) & 0xF;
            if (param_type == PARAM_DOUBLE || param_type == PARAM_FLOAT) {
                all_int = false;
            } else {
                all_float = false;
            }
        }

        

        // Call based on return type and param type patterns
        // Note: Mixed int/float params require the same positional slots
        // We use all_float to decide whether to use double casts
        
        if (self->return_type == JITCallableReturnType::INT64) {
            int64_t result;
            if (all_float) {
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<int64_t(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<int64_t(*)(double)>(self->func_ptr)(dargs[0]); break;
                    case 2: result = reinterpret_cast<int64_t(*)(double,double)>(self->func_ptr)(dargs[0], dargs[1]); break;
                    case 3: result = reinterpret_cast<int64_t(*)(double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2]); break;
                    case 4: result = reinterpret_cast<int64_t(*)(double,double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            } else {
                // All int or mixed - use int signature
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<int64_t(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<int64_t(*)(int64_t)>(self->func_ptr)(iargs[0]); break;
                    case 2: result = reinterpret_cast<int64_t(*)(int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1]); break;
                    case 3: result = reinterpret_cast<int64_t(*)(int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2]); break;
                    case 4: result = reinterpret_cast<int64_t(*)(int64_t,int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2], iargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            }
            return PyLong_FromLongLong(result);
        }
        else if (self->return_type == JITCallableReturnType::INT32) {
            int32_t result;
            if (all_float) {
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<int32_t(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<int32_t(*)(double)>(self->func_ptr)(dargs[0]); break;
                    case 2: result = reinterpret_cast<int32_t(*)(double,double)>(self->func_ptr)(dargs[0], dargs[1]); break;
                    case 3: result = reinterpret_cast<int32_t(*)(double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2]); break;
                    case 4: result = reinterpret_cast<int32_t(*)(double,double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            } else {
                // All int or mixed - use int signature
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<int32_t(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<int32_t(*)(int64_t)>(self->func_ptr)(iargs[0]); break;
                    case 2: result = reinterpret_cast<int32_t(*)(int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1]); break;
                    case 3: result = reinterpret_cast<int32_t(*)(int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2]); break;
                    case 4: result = reinterpret_cast<int32_t(*)(int64_t,int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2], iargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            }
            return PyLong_FromLong(result);
        }
        else if (self->return_type == JITCallableReturnType::DOUBLE) {
            double result;
            if (all_int) {
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<double(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<double(*)(int64_t)>(self->func_ptr)(iargs[0]); break;
                    case 2: result = reinterpret_cast<double(*)(int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1]); break;
                    case 3: result = reinterpret_cast<double(*)(int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2]); break;
                    case 4: result = reinterpret_cast<double(*)(int64_t,int64_t,int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1], iargs[2], iargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            } else {
                // All float or mixed - use double signature
                switch (self->param_count) {
                    case 0: result = reinterpret_cast<double(*)()>(self->func_ptr)(); break;
                    case 1: result = reinterpret_cast<double(*)(double)>(self->func_ptr)(dargs[0]); break;
                    case 2: result = reinterpret_cast<double(*)(double,double)>(self->func_ptr)(dargs[0], dargs[1]); break;
                    case 3: result = reinterpret_cast<double(*)(double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2]); break;
                    case 4: result = reinterpret_cast<double(*)(double,double,double,double)>(self->func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            }
            return PyFloat_FromDouble(result);
        }
        else if (self->return_type == JITCallableReturnType::VOID) {
            if (all_float) {
                switch (self->param_count) {
                    case 0: reinterpret_cast<void(*)()>(self->func_ptr)(); break;
                    case 1: reinterpret_cast<void(*)(double)>(self->func_ptr)(dargs[0]); break;
                    case 2: reinterpret_cast<void(*)(double,double)>(self->func_ptr)(dargs[0], dargs[1]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            } else {
                switch (self->param_count) {
                    case 0: reinterpret_cast<void(*)()>(self->func_ptr)(); break;
                    case 1: reinterpret_cast<void(*)(int64_t)>(self->func_ptr)(iargs[0]); break;
                    case 2: reinterpret_cast<void(*)(int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1]); break;
                    default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
                }
            }
            Py_RETURN_NONE;
        }
        else if (self->return_type == JITCallableReturnType::PTR) {
            void* result;
            switch (self->param_count) {
                case 0: result = reinterpret_cast<void*(*)()>(self->func_ptr)(); break;
                case 1: result = reinterpret_cast<void*(*)(int64_t)>(self->func_ptr)(iargs[0]); break;
                case 2: result = reinterpret_cast<void*(*)(int64_t,int64_t)>(self->func_ptr)(iargs[0], iargs[1]); break;
                default: PyErr_SetString(PyExc_NotImplementedError, "too many parameters"); return NULL;
            }
            return PyLong_FromUnsignedLongLong(reinterpret_cast<uint64_t>(result));
        }
        
        PyErr_SetString(PyExc_RuntimeError, "Unknown return type");
        return NULL;

    }
    
    // Create a new JITCallable object
    static PyObject* JITCallable_New(uint64_t func_ptr, JITCallableReturnType ret_type, 
                                     int param_count, uint32_t param_type_mask, 
                                     const char* name, bool is_varargs, bool is_struct_ret) {
        // Ensure type is ready
        static bool type_ready = false;
        if (!type_ready) {
            if (PyType_Ready(&JITCallable_Type) < 0) {
                return NULL;
            }
            type_ready = true;
        }
        
        JITCallableObject* self = PyObject_New(JITCallableObject, &JITCallable_Type);
        if (!self) return NULL;
        
        self->func_ptr = func_ptr;
        self->return_type = ret_type;
        self->param_count = param_count;
        self->param_type_mask = param_type_mask;
        self->is_varargs = is_varargs;
        self->is_struct_ret = is_struct_ret;
        
        if (name) {
            size_t len = strlen(name) + 1;
            self->name = (char*)PyMem_Malloc(len);
            if (self->name) {
                memcpy(self->name, name, len);
            }
        } else {
            self->name = NULL;
        }
        
        return (PyObject*)self;
    }


    InlineCCompiler::InlineCCompiler(JITCore* jit_core)
        : jit_core_(jit_core)
    {
    }

    InlineCCompiler::~InlineCCompiler()
    {
        // RAII: resources cleaned up automatically
    }

    void InlineCCompiler::add_include_path(const std::string& path)
    {
        include_paths_.push_back(path);
    }

    std::string InlineCCompiler::generate_variable_declarations(nb::dict captured_vars)
    {
        std::stringstream ss;

        // Use Python C API for iteration to avoid nanobind cast issues
        PyObject* py_dict = captured_vars.ptr();
        PyObject* key;
        PyObject* value_obj;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(py_dict, &pos, &key, &value_obj)) {
            // Get key as string
            PyObject* key_str = PyObject_Str(key);
            if (!key_str) continue;
            const char* name_cstr = PyUnicode_AsUTF8(key_str);
            if (!name_cstr) {
                Py_DECREF(key_str);
                continue;
            }
            std::string name(name_cstr);
            Py_DECREF(key_str);
            
            // Wrap value for type checking
            nb::object value = nb::borrow<nb::object>(value_obj);

            // Determine C type from Python type
            // Check bool BEFORE int since Python bool is a subclass of int
            if (nb::isinstance<nb::bool_>(value)) {
                int val = PyObject_IsTrue(value.ptr());
                ss << "int " << name << " = " << val << ";\n";
            }
            else if (nb::isinstance<nb::int_>(value)) {
                // Use Python C API directly to avoid nanobind cast issues
                long long val = PyLong_AsLongLong(value.ptr());
                ss << "long long " << name << " = " << val << "LL;\n";
            }
            else if (nb::isinstance<nb::float_>(value)) {
                // Use Python C API directly
                double val = PyFloat_AsDouble(value.ptr());
                ss << "double " << name << " = " << val << ";\n";
            }
            else if (nb::isinstance<nb::str>(value)) {
                std::string val = nb::cast<std::string>(value);
                // Escape special characters in string
                std::string escaped;
                for (char c : val) {
                    if (c == '\\') escaped += "\\\\";
                    else if (c == '"') escaped += "\\\"";
                    else if (c == '\n') escaped += "\\n";
                    else if (c == '\r') escaped += "\\r";
                    else if (c == '\t') escaped += "\\t";
                    else escaped += c;
                }
                ss << "const char* " << name << " = \"" << escaped << "\";\n";
            }
            else if (value.is_none()) {
                ss << "void* " << name << " = (void*)0;\n";
            }
            else if (nb::isinstance<nb::list>(value)) {
                // For lists, provide both:
                // 1. The PyObject* pointer for Python API access (jit_list_size, etc.)
                // 2. Optionally, a C array for fast element access if homogeneous
                PyObject* ptr = value.ptr();
                Py_INCREF(ptr);
                ss << "void* " << name << " = (void*)" << reinterpret_cast<uintptr_t>(ptr) << "ULL;\n";
                
                // Also generate C array version for homogeneous numeric lists
                nb::list lst = nb::cast<nb::list>(value);
                if (lst.size() > 0) {
                    nb::object first = lst[0];
                    if (nb::isinstance<nb::int_>(first)) {
                        ss << "long long " << name << "_arr[] = {";
                        for (size_t i = 0; i < lst.size(); i++) {
                            if (i > 0) ss << ", ";
                            ss << nb::cast<int64_t>(lst[i]);
                        }
                        ss << "};\n";
                        ss << "long long " << name << "_len = " << lst.size() << ";\n";
                    }
                    else if (nb::isinstance<nb::float_>(first)) {
                        ss << "double " << name << "_arr[] = {";
                        for (size_t i = 0; i < lst.size(); i++) {
                            if (i > 0) ss << ", ";
                            ss << nb::cast<double>(lst[i]);
                        }
                        ss << "};\n";
                        ss << "long long " << name << "_len = " << lst.size() << ";\n";
                    }
                }
            }

            else if (nb::isinstance<nb::tuple>(value)) {
                // For tuples, same as lists
                nb::tuple tpl = nb::cast<nb::tuple>(value);
                if (tpl.size() > 0) {
                    nb::object first = tpl[0];
                    if (nb::isinstance<nb::int_>(first)) {
                        ss << "long long " << name << "[] = {";
                        for (size_t i = 0; i < tpl.size(); i++) {
                            if (i > 0) ss << ", ";
                            ss << nb::cast<int64_t>(tpl[i]);
                        }
                        ss << "};\n";
                        ss << "long long " << name << "_len = " << tpl.size() << ";\n";
                    }
                    else if (nb::isinstance<nb::float_>(first)) {
                        ss << "double " << name << "[] = {";
                        for (size_t i = 0; i < tpl.size(); i++) {
                            if (i > 0) ss << ", ";
                            ss << nb::cast<double>(tpl[i]);
                        }
                        ss << "};\n";
                        ss << "long long " << name << "_len = " << tpl.size() << ";\n";
                    }
                }
            }
            // NumPy arrays - detect via buffer protocol and generate direct pointer access
            // Uses buffer protocol at capture time to get pointer addresses as constants
            else if (PyObject_CheckBuffer(value.ptr())) {
                PyObject* ptr = value.ptr();
                
                // Get buffer info to determine element type and data pointer
                Py_buffer view;
                if (PyObject_GetBuffer(ptr, &view, PyBUF_SIMPLE | PyBUF_FORMAT) == 0) {
                    // Get the actual data pointer and size NOW (at capture time)
                    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(view.buf);
                    Py_ssize_t data_len = view.len / view.itemsize;
                    
                    ss << "// NumPy array: " << name << " (captured at compile time)\n";
                    ss << "long long " << name << "_len = " << data_len << "LL;\n";
                    
                    // Generate typed pointer based on format
                    const char* fmt = view.format ? view.format : "B";
                    if (fmt[0] == 'd') {  // float64/double
                        ss << "double* " << name << " = (double*)" << data_ptr << "ULL;\n";
                    } else if (fmt[0] == 'f') {  // float32
                        ss << "float* " << name << " = (float*)" << data_ptr << "ULL;\n";
                    } else if (fmt[0] == 'l' || fmt[0] == 'q') {  // long/longlong
                        ss << "long long* " << name << " = (long long*)" << data_ptr << "ULL;\n";
                    } else if (fmt[0] == 'i') {  // int32
                        ss << "int* " << name << " = (int*)" << data_ptr << "ULL;\n";
                    } else {
                        // Default to void* for unknown types
                        ss << "void* " << name << " = (void*)" << data_ptr << "ULL;\n";
                    }
                    
                    PyBuffer_Release(&view);
                } else {
                    // Fallback if buffer info failed - use original name
                    Py_INCREF(ptr);
                    ss << "void* " << name << " = (void*)" << reinterpret_cast<uintptr_t>(ptr) << "ULL;\n";
                }
            }
            // For other generic objects, pass as PyObject pointer
            // Use the original variable name so C code can access it directly
            else {
                PyObject* ptr = value.ptr();
                // Keep a reference to prevent Python from garbage collecting
                Py_INCREF(ptr);
                ss << "void* " << name << " = (void*)" << reinterpret_cast<uintptr_t>(ptr) << "ULL;\n";
            }
        }


        return ss.str();
    }

    nb::dict InlineCCompiler::extract_exported_variables(llvm::Module* module)
    {
        nb::dict result;

        // Extract global variables from the module
        for (auto& gv : module->globals()) {
            if (gv.hasInitializer() && !gv.isConstant()) {
                std::string name = gv.getName().str();
                // Skip internal variables
                if (name.empty() || name[0] == '.') continue;

                // Get initializer value
                llvm::Constant* init = gv.getInitializer();
                if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(init)) {
                    result[name.c_str()] = nb::int_(ci->getSExtValue());
                }
                else if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(init)) {
                    result[name.c_str()] = nb::float_(cf->getValueAPF().convertToDouble());
                }
            }
        }

        return result;
    }

    nb::dict InlineCCompiler::compile_and_execute(
        const std::string& code,
        const std::string& lang,
        nb::dict captured_vars)
    {
        // Generate variable declarations from captured Python vars
        std::string var_decls = generate_variable_declarations(captured_vars);

        // Build complete C code with extern declarations for RAII helpers
        std::string full_code = R"(
// ============================================================================
// JustJIT Python-C Interop API
// These functions allow inline C/C++ code to interact with Python objects
// ============================================================================

// GIL Management
extern void* jit_gil_acquire(void);
extern void jit_gil_release(void* guard);
extern void* jit_gil_release_begin(void);
extern void jit_gil_release_end(void* save);

// NumPy Buffer Access
extern void* jit_buffer_new(void* arr);
extern void jit_buffer_free(void* buf);
extern void* jit_buffer_data(void* buf);
extern long long jit_buffer_size(void* buf);

// Type Conversions
extern long long jit_py_to_long(void* obj);
extern double jit_py_to_double(void* obj);
extern const char* jit_py_to_string(void* obj);
extern void* jit_long_to_py(long long val);
extern void* jit_double_to_py(double val);
extern void* jit_string_to_py(const char* val);

// Python Function Call
extern void* jit_call_python(void* func, void* args);

// List Operations
extern void* jit_list_new(long long size);
extern long long jit_list_size(void* list);
extern void* jit_list_get(void* list, long long index);
extern int jit_list_set(void* list, long long index, void* item);
extern int jit_list_append(void* list, void* item);

// Dict Operations
extern void* jit_dict_new(void);
extern void* jit_dict_get(void* dict, const char* key);
extern void* jit_dict_get_obj(void* dict, void* key);
extern int jit_dict_set(void* dict, const char* key, void* val);
extern int jit_dict_set_obj(void* dict, void* key, void* val);
extern int jit_dict_del(void* dict, const char* key);
extern void* jit_dict_keys(void* dict);

// Tuple Operations
extern void* jit_tuple_new(long long size);
extern void* jit_tuple_get(void* tuple, long long index);
extern int jit_tuple_set(void* tuple, long long index, void* item);

// Object Attribute/Method Access
extern void* jit_getattr(void* obj, const char* name);
extern int jit_setattr(void* obj, const char* name, void* val);
extern int jit_hasattr(void* obj, const char* name);
extern void* jit_call_method(void* obj, const char* method, void* args);
extern void* jit_call_method0(void* obj, const char* method);

// Reference Counting
extern void jit_incref(void* obj);
extern void jit_decref(void* obj);

// Module Import
extern void* jit_import(const char* name);

// Sequence/Iterator Operations
extern long long jit_len(void* obj);
extern void* jit_getitem(void* obj, long long index);
extern int jit_setitem(void* obj, long long index, void* val);
extern void* jit_getitem_obj(void* obj, void* key);
extern int jit_setitem_obj(void* obj, void* key, void* val);

// Type Checking
extern int jit_is_list(void* obj);
extern int jit_is_dict(void* obj);
extern int jit_is_tuple(void* obj);
extern int jit_is_int(void* obj);
extern int jit_is_float(void* obj);
extern int jit_is_str(void* obj);
extern int jit_is_none(void* obj);
extern int jit_is_callable(void* obj);

// Constants
extern void* jit_none(void);
extern void* jit_true(void);
extern void* jit_false(void);

// Error Handling
extern int jit_error_occurred(void);
extern void jit_error_clear(void);
extern void jit_error_print(void);

// Enhanced Callbacks - Call Python functions with arguments
extern void* jit_call1(void* func, void* arg);
extern void* jit_call2(void* func, void* arg1, void* arg2);
extern void* jit_call3(void* func, void* arg1, void* arg2, void* arg3);
extern void* jit_call_method1(void* obj, const char* method, void* arg);
extern void* jit_call_method2(void* obj, const char* method, void* arg1, void* arg2);

// Argument Builders
extern void* jit_build_args1(void* arg);
extern void* jit_build_args2(void* arg1, void* arg2);
extern void* jit_build_args3(void* arg1, void* arg2, void* arg3);
extern void* jit_build_int_args1(long long v1);
extern void* jit_build_int_args2(long long v1, long long v2);
extern void* jit_build_float_args1(double v1);
extern void* jit_build_float_args2(double v1, double v2);

// Iterator Support
extern void* jit_get_iter(void* obj);
extern void* jit_iter_next(void* iter);
extern int jit_iter_check(void* obj);

// Bytes Support
extern void* jit_bytes_new(const char* data, long long len);
extern const char* jit_bytes_data(void* bytes);
extern long long jit_bytes_len(void* bytes);

// Simplified Python Expression Evaluation
extern void* jit_py_eval(const char* expr, void* locals);
extern void* jit_py_exec(const char* code, void* locals);

// ============================================================================
// Convenience Macros for Simplified API
// ============================================================================
#define py_eval(expr) jit_py_eval(expr, 0)
#define py_exec(code) jit_py_exec(code, 0)
#define py_import(name) jit_import(name)
#define py_call0(obj, method) jit_call_method0(obj, method)
#define py_call1(obj, method, arg) jit_call_method1(obj, method, arg)
#define py_attr(obj, name) jit_getattr(obj, name)

// ============================================================================
// RAII Scope Guards using __attribute__((cleanup))
// These macros ensure automatic cleanup when variables go out of scope
// Works with GCC, Clang, and MSVC (via extensions)
// ============================================================================

// Cleanup function for PyObject pointers
static inline void __jit_pyobj_cleanup(void** ptr) {
    if (*ptr) jit_decref(*ptr);
}

// Cleanup function for buffer handles
static inline void __jit_buffer_cleanup(void** ptr) {
    if (*ptr) jit_buffer_free(*ptr);
}

// Cleanup function for GIL release state
static inline void __jit_gil_cleanup(void** ptr) {
    if (*ptr) jit_gil_release_end(*ptr);
}

// ============================================================================
// Scoped macros for automatic resource management
// ============================================================================

// MSVC doesn't support __attribute__((cleanup)), use different approach
#if defined(_MSC_VER)
// On MSVC, users must manually free - provide helper functions
#define JIT_SCOPED_PYOBJ(name, expr) void* name = (expr)
#define JIT_SCOPED_BUFFER(name, arr) void* name##_buf = jit_buffer_new(arr)
#define JIT_SCOPED_BUFFER_DATA(name, type) ((type*)jit_buffer_data(name##_buf))
#else
// GCC/Clang: automatic cleanup at scope exit
#define JIT_SCOPED_PYOBJ(name, expr) \
    void* name __attribute__((cleanup(__jit_pyobj_cleanup))) = (expr)

#define JIT_SCOPED_BUFFER(name, arr) \
    void* name##_buf __attribute__((cleanup(__jit_buffer_cleanup))) = jit_buffer_new(arr)

#define JIT_SCOPED_BUFFER_DATA(name, type) ((type*)jit_buffer_data(name##_buf))
#endif

// GIL management with automatic reacquisition
#if defined(_MSC_VER)
#define JIT_NOGIL_BEGIN void* __jit_gil_save = jit_gil_release_begin()
#define JIT_NOGIL_END jit_gil_release_end(__jit_gil_save)
#else
#define JIT_NOGIL_SCOPE \
    void* __jit_gil_save __attribute__((cleanup(__jit_gil_cleanup))) = jit_gil_release_begin()
#define JIT_NOGIL_BEGIN void* __jit_gil_save = jit_gil_release_begin()
#define JIT_NOGIL_END jit_gil_release_end(__jit_gil_save)
#endif

// ============================================================================
// Convenience helpers - simple data access (caller manages lifetime)
// Use JIT_SCOPED_BUFFER for automatic cleanup instead
// ============================================================================

// Quick buffer access - returns pointer, buffer handle stored separately
// Usage: 
//   JIT_SCOPED_BUFFER(arr, pyobj);          // creates arr_buf with cleanup
//   double* data = JIT_SCOPED_BUFFER_DATA(arr, double);
typedef struct { void* buf; void* data; } JitBuffer;

static inline JitBuffer jit_open_buffer_double(void* pyobj) {
    JitBuffer b;
    b.buf = jit_buffer_new(pyobj);
    b.data = b.buf ? jit_buffer_data(b.buf) : 0;
    return b;
}

static inline JitBuffer jit_open_buffer_int(void* pyobj) {
    JitBuffer b;
    b.buf = jit_buffer_new(pyobj);
    b.data = b.buf ? jit_buffer_data(b.buf) : 0;
    return b;
}

static inline JitBuffer jit_open_buffer_float(void* pyobj) {
    JitBuffer b;
    b.buf = jit_buffer_new(pyobj);
    b.data = b.buf ? jit_buffer_data(b.buf) : 0;
    return b;
}

static inline void jit_close_buffer(JitBuffer* b) {
    if (b->buf) jit_buffer_free(b->buf);
    b->buf = 0;
    b->data = 0;
}

// Legacy helpers (may leak if not manually freed - prefer JIT_SCOPED_BUFFER)
static inline double* buffer_double(void* pyobj) {
    void* buf = jit_buffer_new(pyobj);
    return buf ? (double*)jit_buffer_data(buf) : 0;
}

static inline long long* buffer_int(void* pyobj) {
    void* buf = jit_buffer_new(pyobj);
    return buf ? (long long*)jit_buffer_data(buf) : 0;
}

static inline float* buffer_float(void* pyobj) {
    void* buf = jit_buffer_new(pyobj);
    return buf ? (float*)jit_buffer_data(buf) : 0;
}

)" + var_decls + "\n" + code;

        // Create a local context for this compilation
        auto local_context = std::make_unique<llvm::LLVMContext>();

        // Generate temp file path
        static std::atomic<int> counter{0};
        int id = counter++;
        std::string temp_dir = ".";
        if (const char* tmp = std::getenv("TEMP")) {
            temp_dir = tmp;
        } else if (const char* tmp2 = std::getenv("TMP")) {
            temp_dir = tmp2;
        }
        std::string src_file = temp_dir + "/justjit_" + std::to_string(id) + 
                               (lang == "c++" ? ".cpp" : ".c");

        // Write source code to temp file
        {
            std::ofstream out(src_file);
            if (!out) {
                throw std::runtime_error("CError: Failed to create temp source file: " + src_file);
            }
            out << full_code;
        }

        // =====================================================================
        // Simple CompilerInstance approach with environment variable detection
        // Run from Developer Command Prompt for automatic MSVC path detection
        // =====================================================================
        
        // Build command line args
        std::vector<std::string> args_storage;
        std::vector<const char*> args;
        
        args_storage.push_back("-x");
        if (lang == "c++") {
            args_storage.push_back("c++");
            args_storage.push_back("-std=c++17");
        } else {
            args_storage.push_back("c");
            args_storage.push_back("-std=c11");
        }
        args_storage.push_back("-O2");
        
        // Windows SDK headers require Microsoft extensions (__declspec, etc.)
        #ifdef _WIN32
        args_storage.push_back("-fms-extensions");
        args_storage.push_back("-D_CRT_SECURE_NO_WARNINGS");
        #endif
        
        // Add user-specified include paths
        for (const auto& path : include_paths_) {
            args_storage.push_back("-I" + path);
        }
        
        // =====================================================================
        // Platform-aware header search order
        // Windows: Use Windows SDK (complete C library, no musl to avoid conflicts)
        // Linux:   Use musl for portability, then system headers
        // macOS:   Use macOS SDK
        // =====================================================================
        
        #ifdef _WIN32
        // On Windows: Try Windows SDK headers first (they have complete C library)
        #ifdef JUSTJIT_WINSDK_UCRT_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_WINSDK_UCRT_DIR);
        #endif
        #ifdef JUSTJIT_WINSDK_SHARED_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_WINSDK_SHARED_DIR);
        #endif
        #ifdef JUSTJIT_WINSDK_UM_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_WINSDK_UM_DIR);
        #endif
        #ifdef JUSTJIT_MSVC_INCLUDE_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_MSVC_INCLUDE_DIR);
        #endif
        
        // Fallback 1: INCLUDE env var from Developer Command Prompt
        #if !defined(JUSTJIT_WINSDK_UCRT_DIR)
        {
            bool found_include = false;
            if (const char* inc_env = std::getenv("INCLUDE")) {
                std::string include_str(inc_env);
                std::stringstream ss(include_str);
                std::string path;
                while (std::getline(ss, path, ';')) {
                    if (!path.empty()) {
                        args_storage.push_back("-isystem");
                        args_storage.push_back(path);
                        found_include = true;
                    }
                }
            }
            
            // Fallback 2: Use embedded musl if no SDK/INCLUDE available
            // This enables pure C code to work without any dev tools installed
            #ifdef JUSTJIT_EMBEDDED_LIBC_DIR
            if (!found_include) {
                args_storage.push_back("-isystem");
                args_storage.push_back(JUSTJIT_EMBEDDED_LIBC_DIR);
            }
            #endif
        }
        #endif
        
        #else
        // On Linux/macOS: Use embedded musl libc headers for portability
        #ifdef JUSTJIT_EMBEDDED_LIBC_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_EMBEDDED_LIBC_DIR);
        #endif
        
        // Then system headers as fallback for platform-specific code
        #ifdef JUSTJIT_LINUX_INCLUDE_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_LINUX_INCLUDE_DIR);
        #endif
        
        #ifdef JUSTJIT_MACOS_SDK_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_MACOS_SDK_DIR);
        #endif
        #endif
        
        // Use embedded Clang resource headers (stddef.h, stdint.h, stdarg.h, etc.)
        // These are provided by the Clang installation and found at CMake configure time
        #ifdef JUSTJIT_CLANG_RESOURCE_DIR
        args_storage.push_back("-isystem");
        args_storage.push_back(JUSTJIT_CLANG_RESOURCE_DIR);
        #endif
        
        // Use embedded libc++ for C++ standard library (cmath, cstdlib, etc.)
        #ifdef JUSTJIT_LIBCXX_DIR
        if (lang == "c++") {
            args_storage.push_back("-stdlib=libc++");
            args_storage.push_back("-isystem");
            args_storage.push_back(JUSTJIT_LIBCXX_DIR);
        }
        #endif

        
        args_storage.push_back(src_file);
        
        // Convert to const char* array
        for (const auto& arg : args_storage) {
            args.push_back(arg.c_str());
        }

        // Create compiler instance
        clang::CompilerInstance compiler;
        
        // Create diagnostics (LLVM 18+ API)
        auto diag_opts = llvm::makeIntrusiveRefCnt<clang::DiagnosticOptions>();
        clang::TextDiagnosticPrinter* diag_printer = 
            new clang::TextDiagnosticPrinter(llvm::errs(), diag_opts.get());
        compiler.createDiagnostics(diag_printer, true);
        
        // Create invocation and parse args
        clang::CompilerInvocation::CreateFromArgs(
            compiler.getInvocation(),
            args,
            compiler.getDiagnostics()
        );

        // Set up target (LLVM 18+ uses shared_ptr for TargetOptions)
        std::string target_triple = llvm::sys::getDefaultTargetTriple();
        auto target_opts = std::make_shared<clang::TargetOptions>();
        target_opts->Triple = target_triple;
        compiler.setTarget(clang::TargetInfo::CreateTargetInfo(
            compiler.getDiagnostics(), target_opts));

        // Create file manager and source manager
        compiler.createFileManager();
        compiler.createSourceManager(compiler.getFileManager());

        // Use local_context for the action
        clang::EmitLLVMOnlyAction action(local_context.get());

        bool success = compiler.ExecuteAction(action);
        
        // Cleanup temp file
        std::remove(src_file.c_str());

        if (!success) {
            throw std::runtime_error("CError: Failed to compile inline C code");
        }

        // Get the generated module
        std::unique_ptr<llvm::Module> module = action.takeModule();
        if (!module) {
            throw std::runtime_error("CError: No module generated");
        }
        
        // Capture IR for dump_ir functionality
        std::string ir_str;
        llvm::raw_string_ostream ir_stream(ir_str);
        module->print(ir_stream, nullptr);
        last_ir_ = ir_stream.str();

        // Extract function names before adding to JIT (using Python C API)
        PyObject* result_dict = PyDict_New();
        if (!result_dict) {
            throw std::runtime_error("CError: Failed to create result dict");
        }
        
        // Create list of function names
        PyObject* func_list = PyList_New(0);
        if (!func_list) {
            Py_DECREF(result_dict);
            throw std::runtime_error("CError: Failed to create function list");
        }
        
        // Store function info for later callable creation
        // ParamType enum for per-parameter type tracking
        enum class ParamType { INT64, INT32, DOUBLE, FLOAT, PTR };
        
        struct FuncInfo {
            std::string name;           // Original function name (for export key)
            std::string symbol_name;    // Name to lookup (may be wrapper)
            int param_count;
            bool is_double_ret;
            bool is_void_ret;
            bool is_int32_ret;      // true for i32, i16, i8, i1 return types
            bool is_ptr_ret;        // true for pointer return types
            bool is_struct_ret;     // true for struct return (unsupported)
            bool is_varargs;        // true for varargs functions
            std::vector<ParamType> param_types;  // Per-parameter type tracking
        };

        std::vector<FuncInfo> functions_to_export;
        
        for (auto& func : module->functions()) {
            if (!func.isDeclaration() && !func.getName().empty()) {
                std::string func_name = func.getName().str();
                // Skip internal/system functions
                if (func_name[0] != '_' && func_name.find("jit_") != 0 && 
                    func_name.find("buffer_") != 0 && func_name.find("gil_") != 0) {
                    
                    PyObject* py_name = PyUnicode_FromString(func_name.c_str());
                    if (py_name) {
                        PyList_Append(func_list, py_name);
                        Py_DECREF(py_name);
                    }
                    
                    // Detect signature from LLVM types
                    FuncInfo info;
                    info.name = func_name;
                    info.symbol_name = func_name;  // Default: same as name, may be changed if wrapper created
                    info.param_count = func.arg_size();

                    info.is_varargs = func.isVarArg();
                    
                    llvm::Type* ret_type = func.getReturnType();
                    info.is_void_ret = ret_type->isVoidTy();
                    info.is_double_ret = ret_type->isDoubleTy() || ret_type->isFloatTy();
                    info.is_ptr_ret = ret_type->isPointerTy();
                    info.is_struct_ret = ret_type->isStructTy();
                    
                    // Detect small integer return types (i32, i16, i8, i1)
                    info.is_int32_ret = false;
                    if (ret_type->isIntegerTy()) {
                        unsigned bit_width = ret_type->getIntegerBitWidth();
                        if (bit_width <= 32) {
                            info.is_int32_ret = true;
                        }
                    }
                    
                    // Track each parameter's type
                    for (auto& arg : func.args()) {
                        llvm::Type* arg_type = arg.getType();
                        if (arg_type->isDoubleTy()) {
                            info.param_types.push_back(ParamType::DOUBLE);
                        } else if (arg_type->isFloatTy()) {
                            info.param_types.push_back(ParamType::FLOAT);
                        } else if (arg_type->isPointerTy()) {
                            info.param_types.push_back(ParamType::PTR);
                        } else if (arg_type->isIntegerTy()) {
                            unsigned bit_width = arg_type->getIntegerBitWidth();
                            if (bit_width <= 32) {
                                info.param_types.push_back(ParamType::INT32);
                            } else {
                                info.param_types.push_back(ParamType::INT64);
                            }
                        } else {
                            info.param_types.push_back(ParamType::INT64);  // Default
                        }
                    }

                    
                    functions_to_export.push_back(info);
                }
            }
        }
        PyDict_SetItemString(result_dict, "functions", func_list);
        Py_DECREF(func_list);
        
        // Generate LLVM IR wrappers for mixed-type functions
        // This enables proper calling convention for mixed int/float params
        llvm::LLVMContext& ctx = module->getContext();
        llvm::IRBuilder<> builder(ctx);
        for (auto& info : functions_to_export) {
            // Check if this function has mixed types or float params
            bool has_int = false, has_float_or_double = false, has_float32 = false;
            for (const auto& pt : info.param_types) {
                if (pt == ParamType::DOUBLE || pt == ParamType::FLOAT) {
                    has_float_or_double = true;
                    if (pt == ParamType::FLOAT) {
                        has_float32 = true;  // 32-bit float needs wrapper
                    }
                } else {
                    has_int = true;
                }
            }

            
            // Generate wrapper if: mixed types OR has float32 params (need double→float conversion)
            bool needs_wrapper = (has_int && has_float_or_double) || has_float32;
            if (needs_wrapper && info.param_count > 0 && info.param_count <= 4) {

                llvm::Function* orig_func = module->getFunction(info.name);
                if (!orig_func) continue;
                
                // Create wrapper function type: all params as int64, return double for float
                std::vector<llvm::Type*> wrapper_param_types(info.param_count, 
                    llvm::Type::getInt64Ty(ctx));
                // Use double return type if original returns float (for consistent calling)
                llvm::Type* wrapper_ret_type = orig_func->getReturnType();
                if (wrapper_ret_type->isFloatTy()) {
                    wrapper_ret_type = llvm::Type::getDoubleTy(ctx);
                }
                llvm::FunctionType* wrapper_type = llvm::FunctionType::get(
                    wrapper_ret_type, wrapper_param_types, false);

                
                std::string wrapper_name = "__jit_wrap_" + info.name;
                llvm::Function* wrapper_func = llvm::Function::Create(
                    wrapper_type, llvm::Function::ExternalLinkage, wrapper_name, module.get());
                
                // Create entry block
                llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", wrapper_func);
                builder.SetInsertPoint(entry);
                
                // Convert args: reinterpret int64 bits to double/float where needed
                // LLVM doesn't allow bitcast between int and float, so use alloca+store+load
                std::vector<llvm::Value*> converted_args;
                auto arg_it = wrapper_func->arg_begin();
                for (size_t i = 0; i < info.param_types.size(); i++, arg_it++) {
                    llvm::Value* arg_val = &*arg_it;
                    
                    if (info.param_types[i] == ParamType::DOUBLE) {
                        // Reinterpret int64 bits as double via memory (opaque pointer compatible)
                        llvm::AllocaInst* alloca = builder.CreateAlloca(
                            llvm::Type::getInt64Ty(ctx), nullptr, "int_slot");
                        builder.CreateStore(arg_val, alloca);
                        // With opaque pointers, just load with target type directly
                        llvm::Value* converted = builder.CreateLoad(
                            llvm::Type::getDoubleTy(ctx), alloca, "as_double");
                        converted_args.push_back(converted);
                    } else if (info.param_types[i] == ParamType::FLOAT) {
                        // Python passes doubles, need to convert double→float
                        // First reinterpret int64 bits as double, then fptrunc to float
                        llvm::AllocaInst* alloca = builder.CreateAlloca(
                            llvm::Type::getInt64Ty(ctx), nullptr, "int_slot");
                        builder.CreateStore(arg_val, alloca);
                        llvm::Value* as_double = builder.CreateLoad(
                            llvm::Type::getDoubleTy(ctx), alloca, "as_double");
                        // Convert double to float
                        llvm::Value* converted = builder.CreateFPTrunc(as_double,
                            llvm::Type::getFloatTy(ctx), "as_float");
                        converted_args.push_back(converted);

                    } else {
                        // Integer or pointer type
                        llvm::Type* expected = orig_func->getArg(i)->getType();
                        
                        if (expected->isPointerTy()) {
                            // Convert int64 to pointer via inttoptr
                            arg_val = builder.CreateIntToPtr(arg_val, expected);
                        } else if (expected != arg_val->getType() && expected->isIntegerTy()) {
                            // Integer truncation/extension
                            unsigned expected_bits = expected->getIntegerBitWidth();
                            arg_val = builder.CreateTrunc(arg_val, expected);
                        }
                        converted_args.push_back(arg_val);
                    }

                }
                
                // Call original function
                llvm::Value* call_result = builder.CreateCall(orig_func, converted_args);
                
                // Return result (extend float to double for consistent calling)
                if (orig_func->getReturnType()->isVoidTy()) {
                    builder.CreateRetVoid();
                } else if (orig_func->getReturnType()->isFloatTy()) {
                    // Extend float return to double
                    llvm::Value* extended = builder.CreateFPExt(call_result, 
                        llvm::Type::getDoubleTy(ctx), "ret_as_double");
                    builder.CreateRet(extended);
                } else {
                    builder.CreateRet(call_result);
                }
                
                // Update info to use wrapper for symbol lookup (keep original name for export key)
                info.symbol_name = wrapper_name;
                // Mark all params as INT64 since wrapper takes all int64
                for (auto& pt : info.param_types) {
                    pt = ParamType::INT64;
                }
            }
        }


        // Add to JIT (same pattern as other compile functions)
        auto err = jit_core_->jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context))
        );

        if (err) {
            Py_DECREF(result_dict);
            std::string err_str;
            llvm::raw_string_ostream os(err_str);
            os << err;
            throw std::runtime_error("CError: Failed to add IR to JIT: " + err_str);
        }

        // Create callables for each exported function and add to result dict
        for (const auto& info : functions_to_export) {
            // Lookup using symbol_name (may be wrapper for mixed-type functions)
            uint64_t func_ptr = jit_core_->lookup_symbol(info.symbol_name);
            if (func_ptr == 0) continue;
            

            // Determine return type for JITCallable
            JITCallableReturnType ret_type;
            
            if (info.is_void_ret) {
                ret_type = JITCallableReturnType::VOID;
            } else if (info.is_struct_ret) {
                // Struct returns will error at call time with helpful message
                ret_type = JITCallableReturnType::INT64;  // Placeholder
            } else if (info.is_ptr_ret) {
                ret_type = JITCallableReturnType::PTR;
            } else if (info.is_double_ret) {
                ret_type = JITCallableReturnType::DOUBLE;
            } else if (info.is_int32_ret) {
                ret_type = JITCallableReturnType::INT32;
            } else {
                ret_type = JITCallableReturnType::INT64;
            }
            
            // Build param_type_mask from per-param types (4 bits each, up to 8 params)
            uint32_t param_type_mask = 0;
            for (size_t i = 0; i < info.param_types.size() && i < 8; i++) {
                int type_code;
                switch (info.param_types[i]) {
                    case ParamType::INT64: type_code = PARAM_INT64; break;
                    case ParamType::INT32: type_code = PARAM_INT32; break;
                    case ParamType::DOUBLE: type_code = PARAM_DOUBLE; break;
                    case ParamType::FLOAT: type_code = PARAM_FLOAT; break;
                    case ParamType::PTR: type_code = PARAM_PTR; break;
                    default: type_code = PARAM_INT64; break;
                }
                param_type_mask |= (type_code << (i * 4));
            }
            
            // Create callable with per-param types
            PyObject* callable = JITCallable_New(func_ptr, ret_type, info.param_count, 
                                                  param_type_mask, info.name.c_str(),
                                                  info.is_varargs, info.is_struct_ret);
            if (callable) {
                PyDict_SetItemString(result_dict, info.name.c_str(), callable);
                Py_DECREF(callable);
            }
        }


        // Return the result dict (steal reference)
        return nb::steal<nb::dict>(result_dict);
    }

    nb::object InlineCCompiler::get_c_callable(const std::string& name, const std::string& signature)
    {
        // Look up the symbol in the JIT
        uint64_t func_ptr = jit_core_->lookup_symbol(name);
        if (func_ptr == 0) {
            throw std::runtime_error("CError: Symbol not found: " + name);
        }

        // Parse signature like "int(int,int)" or "double(double)" or "void*(void*,long long)"
        // Find the return type (before '(')
        size_t paren_pos = signature.find('(');
        if (paren_pos == std::string::npos) {
            throw std::runtime_error("CError: Invalid signature format: " + signature);
        }

        std::string return_type = signature.substr(0, paren_pos);
        std::string params = signature.substr(paren_pos + 1);
        
        // Remove trailing ')'
        if (!params.empty() && params.back() == ')') {
            params.pop_back();
        }

        // Count parameters and build param_type_mask
        int param_count = 0;
        uint32_t param_type_mask = 0;
        
        if (!params.empty()) {
            // Split by comma and process each param
            std::stringstream ss(params);
            std::string param;
            int idx = 0;
            while (std::getline(ss, param, ',') && idx < 8) {
                param_count++;
                // Trim whitespace
                size_t start = param.find_first_not_of(" \t");
                size_t end = param.find_last_not_of(" \t");
                if (start != std::string::npos) {
                    param = param.substr(start, end - start + 1);
                }
                
                // Determine type code
                int type_code = PARAM_INT64;  // default
                if (param.find("double") != std::string::npos) {
                    type_code = PARAM_DOUBLE;
                } else if (param.find("float") != std::string::npos) {
                    type_code = PARAM_FLOAT;
                } else if (param.find('*') != std::string::npos) {
                    type_code = PARAM_PTR;
                }
                
                param_type_mask |= (type_code << (idx * 4));
                idx++;
            }
        }

        // Determine return type enum
        JITCallableReturnType ret_type;
        if (return_type == "int" || return_type == "long" || return_type == "long long") {
            ret_type = JITCallableReturnType::INT64;
        } else if (return_type == "double" || return_type == "float") {
            ret_type = JITCallableReturnType::DOUBLE;
        } else if (return_type == "void") {
            ret_type = JITCallableReturnType::VOID;
        } else if (!return_type.empty() && return_type.back() == '*') {
            ret_type = JITCallableReturnType::PTR;
        } else {
            throw std::runtime_error("CError: Unsupported return type: " + return_type);
        }

        // Create native Python callable using JITCallable_New
        PyObject* callable = JITCallable_New(func_ptr, ret_type, param_count, 
                                              param_type_mask, name.c_str(), 
                                              false, false);
        if (!callable) {
            throw std::runtime_error("CError: Failed to create JITCallable");
        }
        
        return nb::steal<nb::object>(callable);
    }


#endif // JUSTJIT_HAS_CLANG

} // namespace justjit


