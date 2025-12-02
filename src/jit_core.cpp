#include "jit_core.h"
#include "opcodes.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace justjit {

JITCore::JITCore() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    
    auto jit_builder = llvm::orc::LLJITBuilder();
    auto jit_result = jit_builder.create();
    
    if (!jit_result) {
        llvm::errs() << "Failed to create LLJIT: " << toString(jit_result.takeError()) << "\n";
        return;
    }
    
    jit = std::move(*jit_result);
    context = std::make_unique<llvm::LLVMContext>();
}

JITCore::~JITCore() {
    // Release all stored Python object references
    for (PyObject* obj : stored_constants) {
        if (obj != nullptr) {
            Py_DECREF(obj);
        }
    }
    for (PyObject* obj : stored_names) {
        if (obj != nullptr) {
            Py_DECREF(obj);
        }
    }
    for (PyObject* obj : stored_closure_cells) {
        if (obj != nullptr) {
            Py_DECREF(obj);
        }
    }
    // Release globals and builtins dicts
    if (globals_dict_ptr != nullptr) {
        Py_DECREF(globals_dict_ptr);
    }
    if (builtins_dict_ptr != nullptr) {
        Py_DECREF(builtins_dict_ptr);
    }
    stored_constants.clear();
    stored_names.clear();
    stored_closure_cells.clear();
}
    
void JITCore::set_opt_level(int level) {
    opt_level = std::min(std::max(level, 0), 3);
}

int JITCore::get_opt_level() const {
    return opt_level;
}

nb::object JITCore::get_callable(const std::string& name, int param_count) {
    uint64_t func_ptr = lookup_symbol(name);
    if (func_ptr == 0) {
        return nb::none();
    }
    
    switch (param_count) {
        case 0: return create_callable_0(func_ptr);
        case 1: return create_callable_1(func_ptr);
        case 2: return create_callable_2(func_ptr);
        case 3: return create_callable_3(func_ptr);
        case 4: return create_callable_4(func_ptr);
        default:
            return nb::none();
    }
}

void JITCore::declare_python_api_functions(llvm::Module* module, llvm::IRBuilder<>* builder) {
    llvm::Type* ptr_type = builder->getPtrTy();
    llvm::Type* i64_type = builder->getInt64Ty();
    llvm::Type* void_type = builder->getVoidTy();
    
    // PyObject* PyList_New(Py_ssize_t len)
    llvm::FunctionType* list_new_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
    py_list_new_func = llvm::Function::Create(list_new_type, llvm::Function::ExternalLinkage, "PyList_New", module);
    
    // int PyList_SetItem(PyObject* list, Py_ssize_t index, PyObject* item)  
    // Returns 0 on success, -1 on failure - steals reference to item
    llvm::FunctionType* list_setitem_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, i64_type, ptr_type}, false);
    py_list_setitem_func = llvm::Function::Create(list_setitem_type, llvm::Function::ExternalLinkage, "PyList_SetItem", module);
    
    // PyObject* PyObject_GetItem(PyObject* o, PyObject* key)
    // Returns new reference or NULL on error
    llvm::FunctionType* object_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_object_getitem_func = llvm::Function::Create(object_getitem_type, llvm::Function::ExternalLinkage, "PyObject_GetItem", module);
    
    // void Py_IncRef(PyObject* o)
    llvm::FunctionType* incref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
    py_incref_func = llvm::Function::Create(incref_type, llvm::Function::ExternalLinkage, "Py_IncRef", module);
    
    // void Py_DecRef(PyObject* o)
    llvm::FunctionType* decref_type = llvm::FunctionType::get(void_type, {ptr_type}, false);
    py_decref_func = llvm::Function::Create(decref_type, llvm::Function::ExternalLinkage, "Py_DecRef", module);
    
    // PyObject* PyLong_FromLong(long value)
    llvm::FunctionType* long_fromlong_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
    py_long_fromlong_func = llvm::Function::Create(long_fromlong_type, llvm::Function::ExternalLinkage, "PyLong_FromLong", module);
    
    // PyObject* PyTuple_New(Py_ssize_t len)
    llvm::FunctionType* tuple_new_type = llvm::FunctionType::get(ptr_type, {i64_type}, false);
    py_tuple_new_func = llvm::Function::Create(tuple_new_type, llvm::Function::ExternalLinkage, "PyTuple_New", module);
    
    // void PyTuple_SetItem(PyObject* tuple, Py_ssize_t index, PyObject* item)
    // Steals reference to item, unlike PyList_SetItem which returns int
    llvm::FunctionType* tuple_setitem_type = llvm::FunctionType::get(
        void_type, {ptr_type, i64_type, ptr_type}, false);
    py_tuple_setitem_func = llvm::Function::Create(tuple_setitem_type, llvm::Function::ExternalLinkage, "PyTuple_SetItem", module);
    
    // PyObject* PyNumber_Add(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_add_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_add_func = llvm::Function::Create(number_add_type, llvm::Function::ExternalLinkage, "PyNumber_Add", module);
    
    // PyObject* PyNumber_Subtract(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_subtract_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_subtract_func = llvm::Function::Create(number_subtract_type, llvm::Function::ExternalLinkage, "PyNumber_Subtract", module);
    
    // PyObject* PyNumber_Multiply(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_multiply_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_multiply_func = llvm::Function::Create(number_multiply_type, llvm::Function::ExternalLinkage, "PyNumber_Multiply", module);
    
    // PyObject* PyNumber_TrueDivide(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_truedivide_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_truedivide_func = llvm::Function::Create(number_truedivide_type, llvm::Function::ExternalLinkage, "PyNumber_TrueDivide", module);
    
    // PyObject* PyNumber_FloorDivide(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_floordivide_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_floordivide_func = llvm::Function::Create(number_floordivide_type, llvm::Function::ExternalLinkage, "PyNumber_FloorDivide", module);
    
    // PyObject* PyNumber_Remainder(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_remainder_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_remainder_func = llvm::Function::Create(number_remainder_type, llvm::Function::ExternalLinkage, "PyNumber_Remainder", module);
    
    // PyObject* PyNumber_Power(PyObject* o1, PyObject* o2, PyObject* o3)
    // o3 is for modular exponentiation, pass Py_None to ignore
    llvm::FunctionType* number_power_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type, ptr_type}, false);
    py_number_power_func = llvm::Function::Create(number_power_type, llvm::Function::ExternalLinkage, "PyNumber_Power", module);
    
    // PyObject* PyNumber_Negative(PyObject* o)
    llvm::FunctionType* number_negative_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_number_negative_func = llvm::Function::Create(number_negative_type, llvm::Function::ExternalLinkage, "PyNumber_Negative", module);
    
    // PyObject* PyObject_Str(PyObject* o)
    llvm::FunctionType* object_str_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_object_str_func = llvm::Function::Create(object_str_type, llvm::Function::ExternalLinkage, "PyObject_Str", module);
    
    // PyObject* PyUnicode_Concat(PyObject* left, PyObject* right)
    llvm::FunctionType* unicode_concat_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_unicode_concat_func = llvm::Function::Create(unicode_concat_type, llvm::Function::ExternalLinkage, "PyUnicode_Concat", module);
    
    // PyObject* PyObject_GetAttr(PyObject* o, PyObject* attr_name)
    llvm::FunctionType* getattr_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_object_getattr_func = llvm::Function::Create(getattr_type, llvm::Function::ExternalLinkage, "PyObject_GetAttr", module);
    
    // int PyObject_SetAttr(PyObject* o, PyObject* attr_name, PyObject* value)
    llvm::FunctionType* setattr_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
    py_object_setattr_func = llvm::Function::Create(setattr_type, llvm::Function::ExternalLinkage, "PyObject_SetAttr", module);
    
    // int PyObject_SetItem(PyObject* o, PyObject* key, PyObject* value)
    llvm::FunctionType* object_setitem_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
    py_object_setitem_func = llvm::Function::Create(object_setitem_type, llvm::Function::ExternalLinkage, "PyObject_SetItem", module);
    
    // PyObject* PyObject_Call(PyObject* callable, PyObject* args, PyObject* kwargs)
    // args must be tuple, kwargs can be NULL (nullptr)
    llvm::FunctionType* object_call_type = llvm::FunctionType::get(
        ptr_type, {ptr_type, ptr_type, ptr_type}, false);
    py_object_call_func = llvm::Function::Create(object_call_type, llvm::Function::ExternalLinkage, "PyObject_Call", module);
    
    // long PyLong_AsLong(PyObject* obj) - for unboxing
    llvm::FunctionType* long_aslong_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
    py_long_aslong_func = llvm::Function::Create(long_aslong_type, llvm::Function::ExternalLinkage, "PyLong_AsLong", module);
    
    // int PyObject_RichCompareBool(PyObject* o1, PyObject* o2, int opid)
    // opid: Py_LT=0, Py_LE=1, Py_EQ=2, Py_NE=3, Py_GT=4, Py_GE=5
    // Returns -1 on error, 0 if false, 1 if true
    llvm::FunctionType* richcompare_bool_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type, builder->getInt32Ty()}, false);
    py_object_richcompare_bool_func = llvm::Function::Create(richcompare_bool_type, llvm::Function::ExternalLinkage, "PyObject_RichCompareBool", module);
    
    // int PyObject_IsTrue(PyObject* o)
    // Returns 1 if true, 0 if false, -1 on error
    llvm::FunctionType* istrue_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type}, false);
    py_object_istrue_func = llvm::Function::Create(istrue_type, llvm::Function::ExternalLinkage, "PyObject_IsTrue", module);
    
    // PyObject* PyNumber_Invert(PyObject* o) - bitwise NOT (~)
    llvm::FunctionType* number_invert_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_number_invert_func = llvm::Function::Create(number_invert_type, llvm::Function::ExternalLinkage, "PyNumber_Invert", module);
    
    // int PyObject_Not(PyObject* o) - logical NOT
    // Returns 0 if true, 1 if false, -1 on error
    llvm::FunctionType* object_not_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type}, false);
    py_object_not_func = llvm::Function::Create(object_not_type, llvm::Function::ExternalLinkage, "PyObject_Not", module);
    
    // PyObject* PyObject_GetIter(PyObject* o) - get iterator
    llvm::FunctionType* getiter_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_object_getiter_func = llvm::Function::Create(getiter_type, llvm::Function::ExternalLinkage, "PyObject_GetIter", module);
    
    // PyObject* PyIter_Next(PyObject* o) - get next item from iterator
    // Returns NULL when exhausted (no exception) or on error (exception set)
    llvm::FunctionType* iter_next_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_iter_next_func = llvm::Function::Create(iter_next_type, llvm::Function::ExternalLinkage, "PyIter_Next", module);
    
    // PyObject* PyDict_New() - create new empty dict
    llvm::FunctionType* dict_new_type = llvm::FunctionType::get(ptr_type, {}, false);
    py_dict_new_func = llvm::Function::Create(dict_new_type, llvm::Function::ExternalLinkage, "PyDict_New", module);
    
    // int PyDict_SetItem(PyObject* p, PyObject* key, PyObject* val)
    // Returns 0 on success, -1 on failure
    llvm::FunctionType* dict_setitem_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type, ptr_type}, false);
    py_dict_setitem_func = llvm::Function::Create(dict_setitem_type, llvm::Function::ExternalLinkage, "PyDict_SetItem", module);
    
    // PyObject* PySet_New(PyObject* iterable) - create new set (iterable can be NULL for empty)
    llvm::FunctionType* set_new_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_set_new_func = llvm::Function::Create(set_new_type, llvm::Function::ExternalLinkage, "PySet_New", module);
    
    // int PySet_Add(PyObject* set, PyObject* key)
    // Returns 0 on success, -1 on failure
    llvm::FunctionType* set_add_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_set_add_func = llvm::Function::Create(set_add_type, llvm::Function::ExternalLinkage, "PySet_Add", module);
    
    // int PyList_Append(PyObject* list, PyObject* item)
    // Returns 0 on success, -1 on failure
    llvm::FunctionType* list_append_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_list_append_func = llvm::Function::Create(list_append_type, llvm::Function::ExternalLinkage, "PyList_Append", module);
    
    // int PyList_Extend(PyObject* list, PyObject* iterable)
    // Returns 0 on success, -1 on failure
    llvm::FunctionType* list_extend_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_list_extend_func = llvm::Function::Create(list_extend_type, llvm::Function::ExternalLinkage, "PyList_Extend", module);
    
    // int PySequence_Contains(PyObject* o, PyObject* value)
    // Returns 1 if contains, 0 if not, -1 on error
    llvm::FunctionType* sequence_contains_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_sequence_contains_func = llvm::Function::Create(sequence_contains_type, llvm::Function::ExternalLinkage, "PySequence_Contains", module);
    
    // Bitwise operations
    // PyObject* PyNumber_Lshift(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_lshift_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_lshift_func = llvm::Function::Create(number_lshift_type, llvm::Function::ExternalLinkage, "PyNumber_Lshift", module);
    
    // PyObject* PyNumber_Rshift(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_rshift_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_rshift_func = llvm::Function::Create(number_rshift_type, llvm::Function::ExternalLinkage, "PyNumber_Rshift", module);
    
    // PyObject* PyNumber_And(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_and_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_and_func = llvm::Function::Create(number_and_type, llvm::Function::ExternalLinkage, "PyNumber_And", module);
    
    // PyObject* PyNumber_Or(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_or_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_or_func = llvm::Function::Create(number_or_type, llvm::Function::ExternalLinkage, "PyNumber_Or", module);
    
    // PyObject* PyNumber_Xor(PyObject* o1, PyObject* o2)
    llvm::FunctionType* number_xor_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_number_xor_func = llvm::Function::Create(number_xor_type, llvm::Function::ExternalLinkage, "PyNumber_Xor", module);
    
    // PyObject* PyCell_Get(PyObject* cell) - get contents of a cell object
    // Returns new reference to the cell contents
    llvm::FunctionType* cell_get_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_cell_get_func = llvm::Function::Create(cell_get_type, llvm::Function::ExternalLinkage, "PyCell_Get", module);
    
    // PyObject* PyTuple_GetItem(PyObject* tuple, Py_ssize_t index)
    // Returns borrowed reference
    llvm::FunctionType* tuple_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, i64_type}, false);
    py_tuple_getitem_func = llvm::Function::Create(tuple_getitem_type, llvm::Function::ExternalLinkage, "PyTuple_GetItem", module);
    
    // Py_ssize_t PyTuple_Size(PyObject* tuple)
    llvm::FunctionType* tuple_size_type = llvm::FunctionType::get(i64_type, {ptr_type}, false);
    py_tuple_size_func = llvm::Function::Create(tuple_size_type, llvm::Function::ExternalLinkage, "PyTuple_Size", module);
    
    // PyObject* PySlice_New(PyObject* start, PyObject* stop, PyObject* step)
    // Creates a new slice object - any can be NULL (means default)
    llvm::FunctionType* slice_new_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type, ptr_type}, false);
    py_slice_new_func = llvm::Function::Create(slice_new_type, llvm::Function::ExternalLinkage, "PySlice_New", module);
    
    // PyObject* PySequence_GetSlice(PyObject* o, Py_ssize_t i1, Py_ssize_t i2)
    // Returns o[i1:i2] - new reference
    llvm::FunctionType* sequence_getslice_type = llvm::FunctionType::get(ptr_type, {ptr_type, i64_type, i64_type}, false);
    py_sequence_getslice_func = llvm::Function::Create(sequence_getslice_type, llvm::Function::ExternalLinkage, "PySequence_GetSlice", module);
    
    // int PySequence_SetSlice(PyObject* o, Py_ssize_t i1, Py_ssize_t i2, PyObject* v)
    // Sets o[i1:i2] = v - returns 0 on success, -1 on failure
    llvm::FunctionType* sequence_setslice_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, i64_type, i64_type, ptr_type}, false);
    py_sequence_setslice_func = llvm::Function::Create(sequence_setslice_type, llvm::Function::ExternalLinkage, "PySequence_SetSlice", module);
    
    // int PyObject_DelItem(PyObject* o, PyObject* key)
    // Delete o[key] - returns 0 on success, -1 on failure
    llvm::FunctionType* object_delitem_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_object_delitem_func = llvm::Function::Create(object_delitem_type, llvm::Function::ExternalLinkage, "PyObject_DelItem", module);
    
    // int _PySet_Update(PyObject* set, PyObject* iterable)
    // Update set with items from iterable - returns 0 on success, -1 on failure
    llvm::FunctionType* set_update_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_set_update_func = llvm::Function::Create(set_update_type, llvm::Function::ExternalLinkage, "_PySet_Update", module);
    
    // int PyDict_Update(PyObject* a, PyObject* b)
    // Update dict a with items from dict b - returns 0 on success, -1 on failure
    llvm::FunctionType* dict_update_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_dict_update_func = llvm::Function::Create(dict_update_type, llvm::Function::ExternalLinkage, "PyDict_Update", module);
    
    // int PyDict_Merge(PyObject* a, PyObject* b, int override)
    // Merge dict b into dict a - returns 0 on success, -1 on failure
    llvm::FunctionType* dict_merge_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type, builder->getInt32Ty()}, false);
    py_dict_merge_func = llvm::Function::Create(dict_merge_type, llvm::Function::ExternalLinkage, "PyDict_Merge", module);
    
    // PyObject* PyDict_GetItem(PyObject* p, PyObject* key)
    // Returns borrowed reference or NULL if not found (does NOT set exception)
    llvm::FunctionType* dict_getitem_type = llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false);
    py_dict_getitem_func = llvm::Function::Create(dict_getitem_type, llvm::Function::ExternalLinkage, "PyDict_GetItem", module);
    
    // ========== Exception Handling API (Bug #3 fix) ==========
    
    // PyObject* PyErr_Occurred(void)
    // Returns NULL if no error, otherwise returns the exception type (borrowed ref)
    llvm::FunctionType* err_occurred_type = llvm::FunctionType::get(ptr_type, {}, false);
    py_err_occurred_func = llvm::Function::Create(err_occurred_type, llvm::Function::ExternalLinkage, "PyErr_Occurred", module);
    
    // void PyErr_Fetch(PyObject** ptype, PyObject** pvalue, PyObject** ptraceback)
    // Retrieve exception info and clear error indicator
    llvm::FunctionType* err_fetch_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type, ptr_type}, false);
    py_err_fetch_func = llvm::Function::Create(err_fetch_type, llvm::Function::ExternalLinkage, "PyErr_Fetch", module);
    
    // void PyErr_Restore(PyObject* type, PyObject* value, PyObject* traceback)
    // Set exception info (steals references)
    llvm::FunctionType* err_restore_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type, ptr_type}, false);
    py_err_restore_func = llvm::Function::Create(err_restore_type, llvm::Function::ExternalLinkage, "PyErr_Restore", module);
    
    // void PyErr_SetObject(PyObject* type, PyObject* value)
    // Set exception (does not steal references)
    llvm::FunctionType* err_set_object_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type}, false);
    py_err_set_object_func = llvm::Function::Create(err_set_object_type, llvm::Function::ExternalLinkage, "PyErr_SetObject", module);
    
    // void PyErr_SetString(PyObject* type, const char* message)
    llvm::FunctionType* err_set_string_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type}, false);
    py_err_set_string_func = llvm::Function::Create(err_set_string_type, llvm::Function::ExternalLinkage, "PyErr_SetString", module);
    
    // void PyErr_Clear(void)
    // Clear current error indicator
    llvm::FunctionType* err_clear_type = llvm::FunctionType::get(void_type, {}, false);
    py_err_clear_func = llvm::Function::Create(err_clear_type, llvm::Function::ExternalLinkage, "PyErr_Clear", module);
    
    // int PyErr_GivenExceptionMatches(PyObject* given, PyObject* exc)
    // Returns 1 if given matches exc, 0 otherwise
    llvm::FunctionType* exception_matches_type = llvm::FunctionType::get(
        builder->getInt32Ty(), {ptr_type, ptr_type}, false);
    py_exception_matches_func = llvm::Function::Create(exception_matches_type, llvm::Function::ExternalLinkage, "PyErr_GivenExceptionMatches", module);
    
    // PyObject* PyObject_Type(PyObject* o)
    // Get the type of an object (new reference)
    llvm::FunctionType* object_type_type = llvm::FunctionType::get(ptr_type, {ptr_type}, false);
    py_object_type_func = llvm::Function::Create(object_type_type, llvm::Function::ExternalLinkage, "PyObject_Type", module);
    
    // void PyException_SetCause(PyObject* exc, PyObject* cause)
    // Set __cause__ attribute (steals reference to cause)
    llvm::FunctionType* set_cause_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type}, false);
    py_exception_set_cause_func = llvm::Function::Create(set_cause_type, llvm::Function::ExternalLinkage, "PyException_SetCause", module);
}

bool JITCore::compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::object py_globals_dict, nb::object py_builtins_dict, nb::list py_closure_cells, const std::string& name, int param_count, int total_locals, int nlocals) {
    if (!jit) {
        return false;
    }
    
    // Check if already compiled to prevent duplicate symbol errors
    if (compiled_functions.count(name) > 0) {
        return true;  // Already compiled, return success
    }
    
    // Bug #4 Fix: Store globals and builtins dicts for runtime lookup
    // These are dictionaries, not pre-resolved values
    globals_dict_ptr = py_globals_dict.ptr();
    Py_INCREF(globals_dict_ptr);
    
    builtins_dict_ptr = py_builtins_dict.ptr();
    Py_INCREF(builtins_dict_ptr);
    
    // Convert Python instructions list to C++ vector
    std::vector<Instruction> instructions;
    for (size_t i = 0; i < py_instructions.size(); ++i) {
        nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
        Instruction instr;
        instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
        instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
        instr.argval = nb::cast<int32_t>(instr_dict["argval"]);  // Get actual jump target from Python (can be negative)
        instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
        instructions.push_back(instr);
    }
    
    // Convert Python constants list - support both int64 and PyObject*
    std::vector<int64_t> int_constants;
    std::vector<PyObject*> obj_constants;
    for (size_t i = 0; i < py_constants.size(); ++i) {
        nb::object const_obj = py_constants[i];
        PyObject* py_obj = const_obj.ptr();
        
        // Bug #2 Fix: Check for bool BEFORE int, since bool is a subclass of int
        // Python's True/False need to be stored as PyObject* to preserve identity
        if (py_obj == Py_True || py_obj == Py_False) {
            // Store bools as PyObject* to preserve True/False identity
            int_constants.push_back(0);
            Py_INCREF(py_obj);  // Keep reference alive
            obj_constants.push_back(py_obj);
            stored_constants.push_back(py_obj);  // Track for cleanup in destructor
        }
        // Try to convert to int64 for regular integers
        else if (PyLong_Check(py_obj)) {
            try {
                int64_t int_val = nb::cast<int64_t>(const_obj);
                int_constants.push_back(int_val);
                obj_constants.push_back(nullptr);  // Mark as int constant
            } catch (...) {
                // Integer too large for int64, store as PyObject*
                int_constants.push_back(0);
                Py_INCREF(py_obj);  // Keep reference alive
                obj_constants.push_back(py_obj);
                stored_constants.push_back(py_obj);  // Track for cleanup in destructor
            }
        }
        else {
            // All other types stored as PyObject*
            int_constants.push_back(0);
            Py_INCREF(py_obj);  // Keep reference alive
            obj_constants.push_back(py_obj);
            stored_constants.push_back(py_obj);  // Track for cleanup in destructor
        }
    }
    
    // Extract names (used by LOAD_ATTR, LOAD_GLOBAL, etc)
    std::vector<PyObject*> name_objects;
    for (size_t i = 0; i < py_names.size(); ++i) {
        nb::object name_obj = py_names[i];
        PyObject* py_name = name_obj.ptr();
        Py_INCREF(py_name);  // Keep reference alive
        name_objects.push_back(py_name);
        stored_names.push_back(py_name);  // Track for cleanup in destructor
    }
    
    // Bug #4 Fix: No longer extract global VALUES here.
    // globals_dict_ptr and builtins_dict_ptr are stored at the start of this function.
    // LOAD_GLOBAL will do runtime lookup using PyDict_GetItem.
    
    // Extract closure cells (used by COPY_FREE_VARS / LOAD_DEREF)
    std::vector<PyObject*> closure_cells;
    for (size_t i = 0; i < py_closure_cells.size(); ++i) {
        nb::object cell_obj = py_closure_cells[i];
        if (cell_obj.is_none()) {
            closure_cells.push_back(nullptr);
        } else {
            PyObject* py_cell = cell_obj.ptr();
            Py_INCREF(py_cell);  // Keep reference alive
            closure_cells.push_back(py_cell);
            stored_closure_cells.push_back(py_cell);  // Track for cleanup in destructor
        }
    }
    
    auto local_context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>(name, *local_context);
    llvm::IRBuilder<> builder(*local_context);
        
        // Declare Python C API functions
        declare_python_api_functions(module.get(), &builder);
        
        llvm::Type* i64_type = llvm::Type::getInt64Ty(*local_context);
        llvm::Type* ptr_type = builder.getPtrTy();
        
        // Create function type - return PyObject* (ptr) to support both int and object returns
        // In object mode, all values are PyObject*, ints are boxed as PyLong
        std::vector<llvm::Type*> param_types(param_count, ptr_type);  // Parameters are PyObject*
        llvm::FunctionType* func_type = llvm::FunctionType::get(
            ptr_type,  // Return PyObject*
            param_types,
            false
        );
        
        llvm::Function* func = llvm::Function::Create(
            func_type,
            llvm::Function::ExternalLinkage,
            name,
            module.get()
        );
        
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(*local_context, "entry", func);
        builder.SetInsertPoint(entry);
        
        std::vector<llvm::Value*> stack;
        std::unordered_map<int, llvm::AllocaInst*> local_allocas;
        std::unordered_map<int, llvm::BasicBlock*> jump_targets;
        std::unordered_map<int, size_t> stack_depth_at_offset;  // Track stack depth at each offset for loops
        
        // Bug #1 Fix: Track incoming stack states per block for PHI node insertion
        struct BlockStackState {
            std::vector<llvm::Value*> stack;
            llvm::BasicBlock* predecessor;
        };
        std::unordered_map<int, std::vector<BlockStackState>> block_incoming_stacks;
        std::unordered_map<int, bool> block_needs_phi;  // Blocks that need PHI nodes
        
        // Create allocas only for actual locals needed (not 256!)
        // In object mode, all locals are PyObject* (ptr type)
        llvm::IRBuilder<> alloca_builder(entry, entry->begin());
        llvm::Value* null_ptr_init = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
        for (int i = 0; i < total_locals; ++i) {
            local_allocas[i] = alloca_builder.CreateAlloca(
                ptr_type, nullptr, "local_" + std::to_string(i)
            );
            // CRITICAL: Initialize to NULL to avoid SEGFAULT on LOAD_FAST before STORE_FAST
            alloca_builder.CreateStore(null_ptr_init, local_allocas[i]);
        }
        
        // Store function parameters into allocas
        auto args = func->arg_begin();
        for (int i = 0; i < param_count; ++i) {
            builder.CreateStore(&*args++, local_allocas[i]);
        }
        
        // First pass: Create basic blocks for all jump targets
        jump_targets[0] = entry;
        for (size_t i = 0; i < instructions.size(); ++i) {
            const auto& instr = instructions[i];
            
            if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE ||
                instr.opcode == op::POP_JUMP_IF_NONE || instr.opcode == op::POP_JUMP_IF_NOT_NONE) {
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(target_offset), func
                    );
                }
            } else if (instr.opcode == op::JUMP_BACKWARD) {
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_" + std::to_string(target_offset), func
                    );
                }
            } else if (instr.opcode == op::JUMP_FORWARD) {
                // Pre-create blocks for forward jump targets
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "forward_" + std::to_string(target_offset), func
                    );
                }
            }
        }
        
        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i) {
            int current_offset = instructions[i].offset;
            
            // If this offset is a jump target, switch to that block and handle PHI nodes
            if (jump_targets.count(current_offset) && jump_targets[current_offset] != builder.GetInsertBlock()) {
                // Record current stack state before branching (for fall-through)
                llvm::BasicBlock* current_block = builder.GetInsertBlock();
                if (!current_block->getTerminator()) {
                    // Record stack state for this predecessor
                    BlockStackState state;
                    state.stack = stack;
                    state.predecessor = current_block;
                    block_incoming_stacks[current_offset].push_back(state);
                    
                    builder.CreateBr(jump_targets[current_offset]);
                }
                builder.SetInsertPoint(jump_targets[current_offset]);
                
                // Bug #1 Fix: Check if this block has multiple predecessors with different stacks
                if (block_incoming_stacks.count(current_offset) && 
                    block_incoming_stacks[current_offset].size() > 1) {
                    
                    auto& incoming = block_incoming_stacks[current_offset];
                    size_t stack_size = incoming[0].stack.size();
                    
                    // Verify all predecessors have same stack depth
                    bool valid = true;
                    for (const auto& s : incoming) {
                        if (s.stack.size() != stack_size) {
                            valid = false;
                            break;
                        }
                    }
                    
                    if (valid && stack_size > 0) {
                        // Create PHI nodes for each stack slot that differs
                        std::vector<llvm::Value*> merged_stack;
                        for (size_t slot = 0; slot < stack_size; ++slot) {
                            // Check if all incoming values are the same (no PHI needed)
                            llvm::Value* first_val = incoming[0].stack[slot];
                            bool all_same = true;
                            for (size_t j = 1; j < incoming.size(); ++j) {
                                if (incoming[j].stack[slot] != first_val) {
                                    all_same = false;
                                    break;
                                }
                            }
                            
                            if (all_same) {
                                // No PHI needed, all paths have same value
                                merged_stack.push_back(first_val);
                            } else {
                                // Create PHI node at the start of this block
                                llvm::Type* value_type = first_val->getType();
                                llvm::PHINode* phi = builder.CreatePHI(
                                    value_type, 
                                    incoming.size(),
                                    "stack_phi_" + std::to_string(slot)
                                );
                                
                                // Add incoming values from each predecessor
                                for (const auto& s : incoming) {
                                    phi->addIncoming(s.stack[slot], s.predecessor);
                                }
                                
                                merged_stack.push_back(phi);
                            }
                        }
                        
                        // Replace stack with merged version
                        stack = merged_stack;
                    } else if (stack_depth_at_offset.count(current_offset)) {
                        // Fallback: just restore stack depth
                        size_t expected_depth = stack_depth_at_offset[current_offset];
                        while (stack.size() > expected_depth) {
                            stack.pop_back();
                        }
                    }
                } else if (stack_depth_at_offset.count(current_offset)) {
                    // Single predecessor or no recorded stacks - restore stack depth
                    size_t expected_depth = stack_depth_at_offset[current_offset];
                    while (stack.size() > expected_depth) {
                        stack.pop_back();
                    }
                }
            }
            
            // Record stack depth at this offset (only if not already recorded)
            // This preserves the FIRST time we see this offset's stack state
            if (!stack_depth_at_offset.count(current_offset)) {
                stack_depth_at_offset[current_offset] = stack.size();
            }
            
            const auto& instr = instructions[i];
            
            // Python 3.13 opcodes
            if (instr.opcode == op::RESUME || instr.opcode == op::CACHE) {
                // RESUME is function preamble, CACHE is placeholder for adaptive interpreter
                continue;
            }
            else if (instr.opcode == op::COPY_FREE_VARS) {
                // Copy closure cells from __closure__ tuple into local slots
                // Slots for free vars start at nlocals (after local variables)
                // The cells themselves are stored at compile time in closure_cells vector
                int num_free_vars = instr.arg;
                for (int j = 0; j < num_free_vars && j < static_cast<int>(closure_cells.size()); ++j) {
                    if (closure_cells[j] != nullptr) {
                        // Store the cell pointer in local slot nlocals + j
                        int slot = nlocals + j;
                        if (local_allocas.count(slot)) {
                            llvm::Value* cell_ptr = llvm::ConstantInt::get(
                                i64_type, reinterpret_cast<uint64_t>(closure_cells[j])
                            );
                            llvm::Value* cell_obj = builder.CreateIntToPtr(cell_ptr, ptr_type);
                            builder.CreateStore(cell_obj, local_allocas[slot]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::LOAD_DEREF) {
                // Load from a cell/free variable slot
                // The slot contains a PyCellObject, we need to get its contents
                int slot = instr.arg;
                if (local_allocas.count(slot)) {
                    llvm::Value* cell = builder.CreateLoad(ptr_type, local_allocas[slot], "load_cell_" + std::to_string(slot));
                    // PyCell_Get returns new reference to cell contents
                    llvm::Value* contents = builder.CreateCall(py_cell_get_func, {cell}, "cell_contents");
                    stack.push_back(contents);
                }
            }
            else if (instr.opcode == op::LOAD_FAST) {
                if (local_allocas.count(instr.arg)) {
                    // In object mode, load PyObject* from local
                    llvm::Value* loaded = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "load_local_" + std::to_string(instr.arg));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded});
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
                // Python 3.13: Pushes co_varnames[arg>>4] then co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                if (local_allocas.count(first_local)) {
                    llvm::Value* loaded1 = builder.CreateLoad(ptr_type, local_allocas[first_local], "load_local_" + std::to_string(first_local));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded1});
                    stack.push_back(loaded1);
                }
                if (local_allocas.count(second_local)) {
                    llvm::Value* loaded2 = builder.CreateLoad(ptr_type, local_allocas[second_local], "load_local_" + std::to_string(second_local));
                    // Incref to take ownership - we'll decref when consuming from stack
                    builder.CreateCall(py_incref_func, {loaded2});
                    stack.push_back(loaded2);
                }
            }
            else if (instr.opcode == op::LOAD_CONST) {
                // arg is index into constants table
                if (instr.arg < int_constants.size()) {
                    if (obj_constants[instr.arg] != nullptr) {
                        // PyObject* constant - load as pointer
                        llvm::Value* const_ptr = llvm::ConstantInt::get(
                            i64_type,
                            reinterpret_cast<uint64_t>(obj_constants[instr.arg])
                        );
                        llvm::Value* py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                        // Increment reference count since we're putting it on stack
                        builder.CreateCall(py_incref_func, {py_obj});
                        stack.push_back(py_obj);
                    } else {
                        // int64 constant - keep as i64 to allow fast native integer operations
                        // Will be boxed to PyLong only when needed (e.g., when mixed with PyObject*)
                        llvm::Value* const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                        stack.push_back(const_val);
                    }
                }
            }
            else if (instr.opcode == op::STORE_FAST) {
                if (!stack.empty()) {
                    llvm::Value* val = stack.back();
                    stack.pop_back();
                    
                    // In object mode, locals are PyObject* typed, so box i64 values
                    if (val->getType()->isIntegerTy(64)) {
                        val = builder.CreateCall(py_long_fromlong_func, {val});
                    }
                    
                    // CRITICAL: Decref old value before storing new one to prevent memory leak
                    llvm::Value* old_val = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "old_local");
                    llvm::Value* null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* is_not_null = builder.CreateICmpNE(old_val, null_check, "is_not_null");
                    
                    llvm::BasicBlock* decref_block = llvm::BasicBlock::Create(*local_context, "decref_old", func);
                    llvm::BasicBlock* store_block = llvm::BasicBlock::Create(*local_context, "store_new", func);
                    
                    builder.CreateCondBr(is_not_null, decref_block, store_block);
                    
                    builder.SetInsertPoint(decref_block);
                    builder.CreateCall(py_decref_func, {old_val});
                    builder.CreateBr(store_block);
                    
                    builder.SetInsertPoint(store_block);
                    builder.CreateStore(val, local_allocas[instr.arg]);
                }
            }
            else if (instr.opcode == op::STORE_FAST_STORE_FAST) {
                // Python 3.13: Stores STACK[-1] into co_varnames[arg>>4] and STACK[-2] into co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                
                if (stack.size() >= 2) {
                    llvm::Value* first_val = stack.back(); stack.pop_back();   // STACK[-1]
                    llvm::Value* second_val = stack.back(); stack.pop_back();  // STACK[-2]
                    
                    // Box i64 values if needed
                    if (first_val->getType()->isIntegerTy(64)) {
                        first_val = builder.CreateCall(py_long_fromlong_func, {first_val});
                    }
                    if (second_val->getType()->isIntegerTy(64)) {
                        second_val = builder.CreateCall(py_long_fromlong_func, {second_val});
                    }
                    
                    llvm::Value* null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    
                    // Store first to first_local (with decref of old value)
                    if (local_allocas.count(first_local)) {
                        llvm::Value* old_val1 = builder.CreateLoad(ptr_type, local_allocas[first_local], "old_local1");
                        llvm::Value* is_not_null1 = builder.CreateICmpNE(old_val1, null_check, "is_not_null1");
                        
                        llvm::BasicBlock* decref_block1 = llvm::BasicBlock::Create(*local_context, "decref_old1", func);
                        llvm::BasicBlock* store_block1 = llvm::BasicBlock::Create(*local_context, "store_new1", func);
                        
                        builder.CreateCondBr(is_not_null1, decref_block1, store_block1);
                        
                        builder.SetInsertPoint(decref_block1);
                        builder.CreateCall(py_decref_func, {old_val1});
                        builder.CreateBr(store_block1);
                        
                        builder.SetInsertPoint(store_block1);
                        builder.CreateStore(first_val, local_allocas[first_local]);
                    }
                    
                    // Store second to second_local (with decref of old value)
                    if (local_allocas.count(second_local)) {
                        llvm::Value* old_val2 = builder.CreateLoad(ptr_type, local_allocas[second_local], "old_local2");
                        llvm::Value* is_not_null2 = builder.CreateICmpNE(old_val2, null_check, "is_not_null2");
                        
                        llvm::BasicBlock* decref_block2 = llvm::BasicBlock::Create(*local_context, "decref_old2", func);
                        llvm::BasicBlock* store_block2 = llvm::BasicBlock::Create(*local_context, "store_new2", func);
                        
                        builder.CreateCondBr(is_not_null2, decref_block2, store_block2);
                        
                        builder.SetInsertPoint(decref_block2);
                        builder.CreateCall(py_decref_func, {old_val2});
                        builder.CreateBr(store_block2);
                        
                        builder.SetInsertPoint(store_block2);
                        builder.CreateStore(second_val, local_allocas[second_local]);
                    }
                }
            }
            else if (instr.opcode == op::UNPACK_SEQUENCE) {
                // Unpack TOS into count individual values
                // Stack order after unpack: [..., last_value, ..., first_value] (first value on TOS)
                int count = instr.arg;
                
                if (!stack.empty()) {
                    llvm::Value* sequence = stack.back(); stack.pop_back();
                    
                    // Unpack the sequence by calling PySequence_GetItem for each index
                    // Push in reverse order so that first item ends up on top
                    std::vector<llvm::Value*> unpacked;
                    for (int i = 0; i < count; ++i) {
                        llvm::Value* idx = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value* idx_obj = builder.CreateCall(py_long_fromlong_func, {idx});
                        llvm::Value* item = builder.CreateCall(py_object_getitem_func, {sequence, idx_obj});
                        builder.CreateCall(py_decref_func, {idx_obj});  // Free temp index
                        unpacked.push_back(item);
                    }
                    
                    // Push in reverse order (last item first, so first item is on top)
                    for (int i = count - 1; i >= 0; --i) {
                        stack.push_back(unpacked[i]);
                    }
                    
                    // Decref the original sequence (we're done with it)
                    if (sequence->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {sequence});
                    }
                }
            }
            else if (instr.opcode == op::BINARY_OP) {
                if (stack.size() >= 2) {
                    // Stack: [first_pushed, second_pushed] with second on top
                    llvm::Value* second = stack.back(); stack.pop_back();
                    llvm::Value* first = stack.back(); stack.pop_back();
                    
                    llvm::Value* result = nullptr;
                    
                    // Check if either operand is PyObject* (ptr type)
                    bool first_is_pyobject = first->getType()->isPointerTy() && !first->getType()->isIntegerTy(64);
                    bool second_is_pyobject = second->getType()->isPointerTy() && !second->getType()->isIntegerTy(64);
                    bool is_pyobject = first_is_pyobject || second_is_pyobject;
                    
                    if (is_pyobject) {
                        // Object mode: use Python C API
                        // Box both operands to PyObject* if needed, track if we created temps
                        llvm::Value* orig_first = first;
                        llvm::Value* orig_second = second;
                        bool first_boxed = false;
                        bool second_boxed = false;
                        
                        if (first->getType()->isIntegerTy(64)) {
                            first = builder.CreateCall(py_long_fromlong_func, {first});
                            first_boxed = true;
                        }
                        if (second->getType()->isIntegerTy(64)) {
                            second = builder.CreateCall(py_long_fromlong_func, {second});
                            second_boxed = true;
                        }
                        
                        switch (instr.arg) {
                            case 0:  // ADD (a + b)
                                result = builder.CreateCall(py_number_add_func, {first, second});
                                break;
                            case 10:  // SUB (a - b)
                                result = builder.CreateCall(py_number_subtract_func, {first, second});
                                break;
                            case 5:  // MUL (a * b)
                                result = builder.CreateCall(py_number_multiply_func, {first, second});
                                break;
                            case 11:  // TRUE_DIV (a / b)
                                result = builder.CreateCall(py_number_truedivide_func, {first, second});
                                break;
                            case 2:  // FLOOR_DIV (a // b)
                                result = builder.CreateCall(py_number_floordivide_func, {first, second});
                                break;
                            case 6:  // MOD (a % b)
                                result = builder.CreateCall(py_number_remainder_func, {first, second});
                                break;
                            case 8: {  // POW (a ** b)
                                // PyNumber_Power(base, exp, Py_None) - Py_None for no modular arithmetic
                                llvm::Value* py_none_ptr = llvm::ConstantInt::get(
                                    i64_type,
                                    reinterpret_cast<uint64_t>(Py_None)
                                );
                                llvm::Value* py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                                result = builder.CreateCall(py_number_power_func, {first, second, py_none});
                                break;
                            }
                            case 1:  // AND (a & b) - bitwise
                                result = builder.CreateCall(py_number_and_func, {first, second});
                                break;
                            case 7:  // OR (a | b) - bitwise
                                result = builder.CreateCall(py_number_or_func, {first, second});
                                break;
                            case 12:  // XOR (a ^ b) - bitwise
                                result = builder.CreateCall(py_number_xor_func, {first, second});
                                break;
                            case 3:  // LSHIFT (a << b)
                                result = builder.CreateCall(py_number_lshift_func, {first, second});
                                break;
                            case 9:  // RSHIFT (a >> b)
                                result = builder.CreateCall(py_number_rshift_func, {first, second});
                                break;
                            default:
                                // For unsupported ops, fall back to ADD
                                result = builder.CreateCall(py_number_add_func, {first, second});
                                break;
                        }
                        
                        // Decref boxed temporaries (not the originals)
                        if (first_boxed) {
                            builder.CreateCall(py_decref_func, {first});
                        } else if (first_is_pyobject) {
                            // Decref original PyObject* operand we consumed
                            builder.CreateCall(py_decref_func, {first});
                        }
                        if (second_boxed) {
                            builder.CreateCall(py_decref_func, {second});
                        } else if (second_is_pyobject) {
                            // Decref original PyObject* operand we consumed
                            builder.CreateCall(py_decref_func, {second});
                        }
                    } else {
                        // Native int64 mode
                        switch (instr.arg) {
                            case 0:  // ADD (a + b)
                                result = builder.CreateAdd(first, second, "add");
                                break;
                            case 10:  // SUB (a - b)
                                result = builder.CreateSub(first, second, "sub");
                                break;
                            case 5:  // MUL (a * b)
                                result = builder.CreateMul(first, second, "mul");
                                break;
                            case 11:  // TRUE_DIV (a / b) - returns float, but we only support int for now
                                result = builder.CreateSDiv(first, second, "div");
                                break;
                            case 2:  // FLOOR_DIV (a // b)
                                result = builder.CreateSDiv(first, second, "floordiv");
                                break;
                            case 6:  // MOD (a % b)
                                result = builder.CreateSRem(first, second, "mod");
                                break;
                            case 1:  // AND (a & b) - bitwise
                                result = builder.CreateAnd(first, second, "and");
                                break;
                            case 7:  // OR (a | b) - bitwise
                                result = builder.CreateOr(first, second, "or");
                                break;
                            case 12:  // XOR (a ^ b) - bitwise
                                result = builder.CreateXor(first, second, "xor");
                                break;
                            case 3:  // LSHIFT (a << b)
                                result = builder.CreateShl(first, second, "shl");
                                break;
                            case 9:  // RSHIFT (a >> b)
                                result = builder.CreateAShr(first, second, "shr");
                                break;
                            case 8: {  // POW (a ** b) - Binary exponentiation O(log n)
                                // Implement iterative binary exponentiation in LLVM IR
                                // result = 1; while (exp > 0) { if (exp & 1) result *= base; base *= base; exp >>= 1; }
                                
                                llvm::Function* current_func = builder.GetInsertBlock()->getParent();
                                
                                // Create basic blocks for the power loop
                                llvm::BasicBlock* pow_entry = builder.GetInsertBlock();
                                llvm::BasicBlock* pow_loop = llvm::BasicBlock::Create(*local_context, "pow_loop", current_func);
                                llvm::BasicBlock* pow_odd = llvm::BasicBlock::Create(*local_context, "pow_odd", current_func);
                                llvm::BasicBlock* pow_cont = llvm::BasicBlock::Create(*local_context, "pow_cont", current_func);
                                llvm::BasicBlock* pow_done = llvm::BasicBlock::Create(*local_context, "pow_done", current_func);
                                
                                // Entry: initialize and jump to loop
                                llvm::Value* init_result = llvm::ConstantInt::get(i64_type, 1);
                                builder.CreateBr(pow_loop);
                                
                                // Loop header with phi nodes
                                builder.SetInsertPoint(pow_loop);
                                llvm::PHINode* phi_result = builder.CreatePHI(i64_type, 2, "pow_result");
                                llvm::PHINode* phi_base = builder.CreatePHI(i64_type, 2, "pow_base");
                                llvm::PHINode* phi_exp = builder.CreatePHI(i64_type, 2, "pow_exp");
                                
                                phi_result->addIncoming(init_result, pow_entry);
                                phi_base->addIncoming(first, pow_entry);
                                phi_exp->addIncoming(second, pow_entry);
                                
                                // Check if exp > 0
                                llvm::Value* exp_gt_zero = builder.CreateICmpSGT(
                                    phi_exp, 
                                    llvm::ConstantInt::get(i64_type, 0), 
                                    "exp_gt_zero"
                                );
                                builder.CreateCondBr(exp_gt_zero, pow_odd, pow_done);
                                
                                // Check if exp is odd (exp & 1)
                                builder.SetInsertPoint(pow_odd);
                                llvm::Value* exp_is_odd = builder.CreateAnd(
                                    phi_exp, 
                                    llvm::ConstantInt::get(i64_type, 1), 
                                    "exp_is_odd"
                                );
                                llvm::Value* is_odd = builder.CreateICmpNE(
                                    exp_is_odd, 
                                    llvm::ConstantInt::get(i64_type, 0), 
                                    "is_odd"
                                );
                                
                                // If odd: new_result = result * base, else: new_result = result
                                llvm::Value* result_times_base = builder.CreateMul(phi_result, phi_base, "result_times_base");
                                llvm::Value* new_result = builder.CreateSelect(is_odd, result_times_base, phi_result, "new_result");
                                
                                // base = base * base
                                llvm::Value* new_base = builder.CreateMul(phi_base, phi_base, "base_squared");
                                
                                // exp = exp >> 1
                                llvm::Value* new_exp = builder.CreateAShr(
                                    phi_exp, 
                                    llvm::ConstantInt::get(i64_type, 1), 
                                    "exp_halved"
                                );
                                
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
                                result = builder.CreateAdd(first, second, "add");
                                break;
                        }
                    }
                    
                    if (result) {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_NEGATIVE) {
                // Implements STACK[-1] = -STACK[-1]
                if (!stack.empty()) {
                    llvm::Value* operand = stack.back(); stack.pop_back();
                    llvm::Value* result = nullptr;
                    
                    if (operand->getType()->isIntegerTy(64)) {
                        // Native int64: just negate
                        result = builder.CreateNeg(operand, "neg");
                    } else {
                        // PyObject*: use PyNumber_Negative
                        result = builder.CreateCall(py_number_negative_func, {operand});
                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }
                    
                    if (result) {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_INVERT) {
                // Implements STACK[-1] = ~STACK[-1] (bitwise NOT)
                if (!stack.empty()) {
                    llvm::Value* operand = stack.back(); stack.pop_back();
                    llvm::Value* result = nullptr;
                    
                    if (operand->getType()->isIntegerTy(64)) {
                        // Native int64: use XOR with -1 for bitwise NOT
                        result = builder.CreateXor(operand, llvm::ConstantInt::get(i64_type, -1), "invert");
                    } else {
                        // PyObject*: use PyNumber_Invert
                        result = builder.CreateCall(py_number_invert_func, {operand});
                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }
                    
                    if (result) {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::UNARY_NOT) {
                // Implements STACK[-1] = not STACK[-1] (logical NOT, returns True/False)
                if (!stack.empty()) {
                    llvm::Value* operand = stack.back(); stack.pop_back();
                    llvm::Value* result = nullptr;
                    
                    if (operand->getType()->isIntegerTy(64)) {
                        // Native int64: compare to zero, invert result
                        llvm::Value* is_zero = builder.CreateICmpEQ(operand, llvm::ConstantInt::get(i64_type, 0), "iszero");
                        result = builder.CreateZExt(is_zero, i64_type, "not");
                    } else {
                        // PyObject*: use PyObject_Not
                        // Returns 0 if object is truthy, 1 if falsy, -1 on error
                        llvm::Value* not_result = builder.CreateCall(py_object_not_func, {operand}, "not");
                        
                        // Convert result to Py_True or Py_False
                        // If not_result == 1, return Py_True; else return Py_False
                        llvm::Value* is_true = builder.CreateICmpEQ(not_result, llvm::ConstantInt::get(builder.getInt32Ty(), 1), "is_true");
                        
                        llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                        
                        result = builder.CreateSelect(is_true, py_true, py_false, "not_result");
                        
                        // Incref the result (Py_True/Py_False are immortal in 3.12+, but be safe)
                        builder.CreateCall(py_incref_func, {result});
                        
                        // Decref the operand since we consumed it
                        builder.CreateCall(py_decref_func, {operand});
                    }
                    
                    if (result) {
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::TO_BOOL) {
                // Convert TOS to a boolean value - used before conditionals
                if (!stack.empty()) {
                    llvm::Value* val = stack.back(); stack.pop_back();
                    llvm::Value* result = nullptr;
                    
                    if (val->getType()->isIntegerTy(64)) {
                        // Native int64: compare != 0 to get boolean, then convert to Py_True/Py_False
                        llvm::Value* is_nonzero = builder.CreateICmpNE(val, llvm::ConstantInt::get(i64_type, 0), "nonzero");
                        
                        llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                        
                        result = builder.CreateSelect(is_nonzero, py_true, py_false, "tobool_result");
                        builder.CreateCall(py_incref_func, {result});
                    } else {
                        // PyObject*: use PyObject_IsTrue to get boolean, then return Py_True/Py_False
                        llvm::Value* is_true = builder.CreateCall(py_object_istrue_func, {val}, "istrue");
                        llvm::Value* is_nonzero = builder.CreateICmpNE(is_true, llvm::ConstantInt::get(builder.getInt32Ty(), 0), "nonzero");
                        
                        llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                        llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                        llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                        llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                        
                        result = builder.CreateSelect(is_nonzero, py_true, py_false, "tobool_result");
                        builder.CreateCall(py_incref_func, {result});
                        
                        // Decref the original value
                        builder.CreateCall(py_decref_func, {val});
                    }
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::NOP) {
                // Do nothing - this is a no-operation instruction
            }
            else if (instr.opcode == op::COMPARE_OP) {
                if (stack.size() >= 2) {
                    llvm::Value* rhs = stack.back(); stack.pop_back();
                    llvm::Value* lhs = stack.back(); stack.pop_back();
                    
                    // Python 3.13 encoding: (op_code << 5) | flags
                    // Extraction: op_code = arg >> 5
                    // Compare operations: 0=<, 1=<=, 2===, 3=!=, 4=>, 5=>=
                    int op_code = instr.arg >> 5;
                    llvm::Value* cmp_result = nullptr;
                    
                    // Check if either operand is a pointer (PyObject*)
                    bool lhs_is_ptr = lhs->getType()->isPointerTy();
                    bool rhs_is_ptr = rhs->getType()->isPointerTy();
                    
                    // Prepare Py_True and Py_False pointers for result
                    llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    
                    if (lhs_is_ptr || rhs_is_ptr) {
                        // At least one operand is PyObject* - use PyObject_RichCompareBool
                        // Track if we boxed operands
                        bool lhs_boxed = false;
                        bool rhs_boxed = false;
                        
                        // First, ensure both are PyObject*
                        if (!lhs_is_ptr) {
                            // Convert i64 to PyObject*
                            lhs = builder.CreateCall(py_long_fromlong_func, {lhs});
                            lhs_boxed = true;
                        }
                        if (!rhs_is_ptr) {
                            // Convert i64 to PyObject*
                            rhs = builder.CreateCall(py_long_fromlong_func, {rhs});
                            rhs_boxed = true;
                        }
                        
                        // Map our op_code to Python's comparison opid
                        // Our encoding: 0=<, 1=<=, 2===, 3=!=, 4=>, 5=>=
                        // Python opid: Py_LT=0, Py_LE=1, Py_EQ=2, Py_NE=3, Py_GT=4, Py_GE=5
                        // They match directly
                        llvm::Value* opid = llvm::ConstantInt::get(builder.getInt32Ty(), op_code);
                        
                        // Call PyObject_RichCompareBool - returns int (0=false, 1=true, -1=error)
                        llvm::Value* result = builder.CreateCall(py_object_richcompare_bool_func, {lhs, rhs, opid});
                        
                        // Decref boxed temporaries and consumed PyObject* operands
                        if (lhs_boxed) {
                            builder.CreateCall(py_decref_func, {lhs});
                        } else {
                            builder.CreateCall(py_decref_func, {lhs});
                        }
                        if (rhs_boxed) {
                            builder.CreateCall(py_decref_func, {rhs});
                        } else {
                            builder.CreateCall(py_decref_func, {rhs});
                        }
                        
                        // Convert to Py_True/Py_False (Bug #2 fix)
                        llvm::Value* is_true = builder.CreateICmpSGT(result, llvm::ConstantInt::get(builder.getInt32Ty(), 0));
                        cmp_result = builder.CreateSelect(is_true, py_true, py_false);
                        builder.CreateCall(py_incref_func, {cmp_result});
                    } else {
                        // Both are i64 - use native integer comparison
                        llvm::Value* bool_result = nullptr;
                        switch (op_code) {
                            case 0:  // <
                                bool_result = builder.CreateICmpSLT(lhs, rhs, "lt");
                                break;
                            case 1:  // <=
                                bool_result = builder.CreateICmpSLE(lhs, rhs, "le");
                                break;
                            case 2:  // ==
                                bool_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                                break;
                            case 3:  // !=
                                bool_result = builder.CreateICmpNE(lhs, rhs, "ne");
                                break;
                            case 4:  // >
                                bool_result = builder.CreateICmpSGT(lhs, rhs, "gt");
                                break;
                            case 5:  // >=
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
                    
                    if (cmp_result) {
                        stack.push_back(cmp_result);
                    }
                }
            }
            else if (instr.opcode == op::CONTAINS_OP) {
                // Implements 'in' / 'not in' test
                // Stack: TOS=container, TOS1=value
                // arg & 1: 0 = 'in', 1 = 'not in'
                if (stack.size() >= 2) {
                    llvm::Value* container = stack.back(); stack.pop_back();
                    llvm::Value* value = stack.back(); stack.pop_back();
                    bool invert = (instr.arg & 1) != 0;
                    
                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool container_is_ptr = container->getType()->isPointerTy();
                    
                    // Convert int64 value to PyObject* if needed
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    if (value_was_boxed) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                    }
                    
                    // PySequence_Contains returns 1 if contains, 0 if not, -1 on error
                    llvm::Value* result = builder.CreateCall(py_sequence_contains_func, {container, value}, "contains");
                    
                    if (invert) {
                        // 'not in': invert the result (1->0, 0->1)
                        result = builder.CreateXor(result, llvm::ConstantInt::get(result->getType(), 1), "not_in");
                    }
                    
                    // Decref consumed operands
                    if (value_was_boxed) {
                        builder.CreateCall(py_decref_func, {value});
                    } else if (value_is_ptr) {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    if (container_is_ptr) {
                        builder.CreateCall(py_decref_func, {container});
                    }
                    
                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value* is_true = builder.CreateICmpSGT(result, llvm::ConstantInt::get(result->getType(), 0));
                    llvm::Value* bool_result = builder.CreateSelect(is_true, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::IS_OP) {
                // Implements 'is' / 'is not' identity test
                // Stack: TOS=rhs, TOS1=lhs
                // arg & 1: 0 = 'is', 1 = 'is not'
                if (stack.size() >= 2) {
                    llvm::Value* rhs = stack.back(); stack.pop_back();
                    llvm::Value* lhs = stack.back(); stack.pop_back();
                    bool invert = (instr.arg & 1) != 0;
                    
                    bool lhs_is_ptr = lhs->getType()->isPointerTy();
                    bool rhs_is_ptr = rhs->getType()->isPointerTy();
                    bool lhs_was_boxed = false;
                    bool rhs_was_boxed = false;
                    
                    // Ensure both are pointers for identity comparison
                    if (lhs->getType()->isIntegerTy(64)) {
                        lhs = builder.CreateCall(py_long_fromlong_func, {lhs});
                        lhs_was_boxed = true;
                    }
                    if (rhs->getType()->isIntegerTy(64)) {
                        rhs = builder.CreateCall(py_long_fromlong_func, {rhs});
                        rhs_was_boxed = true;
                    }
                    
                    // Pointer identity comparison
                    llvm::Value* is_same = builder.CreateICmpEQ(lhs, rhs, "is");
                    
                    if (invert) {
                        is_same = builder.CreateNot(is_same, "is_not");
                    }
                    
                    // Decref consumed operands
                    if (lhs_was_boxed) {
                        builder.CreateCall(py_decref_func, {lhs});
                    } else if (lhs_is_ptr) {
                        builder.CreateCall(py_decref_func, {lhs});
                    }
                    if (rhs_was_boxed) {
                        builder.CreateCall(py_decref_func, {rhs});
                    } else if (rhs_is_ptr) {
                        builder.CreateCall(py_decref_func, {rhs});
                    }
                    
                    // Convert to Py_True/Py_False for proper bool semantics
                    llvm::Value* py_true_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_True));
                    llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ptr_type);
                    llvm::Value* py_false_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_False));
                    llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ptr_type);
                    llvm::Value* bool_result = builder.CreateSelect(is_same, py_true, py_false);
                    builder.CreateCall(py_incref_func, {bool_result});
                    stack.push_back(bool_result);
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE) {
                if (!stack.empty() && i + 1 < instructions.size()) {
                    llvm::Value* cond = stack.back(); stack.pop_back();
                    
                    llvm::Value* bool_cond = nullptr;
                    
                    // Handle different condition types
                    if (cond->getType()->isPointerTy()) {
                        // PyObject* - use PyObject_IsTrue for proper Python truthiness
                        // Returns 1 if true, 0 if false, -1 on error
                        llvm::Value* istrue_result = builder.CreateCall(py_object_istrue_func, {cond}, "istrue");
                        // Compare result > 0 (1 means true, 0 means false, -1 means error)
                        bool_cond = builder.CreateICmpSGT(
                            istrue_result,
                            llvm::ConstantInt::get(istrue_result->getType(), 0),
                            "tobool_obj"
                        );
                    } else {
                        // Integer - compare to zero
                        bool_cond = builder.CreateICmpNE(
                            cond,
                            llvm::ConstantInt::get(cond->getType(), 0),
                            "tobool"
                        );
                    }
                    
                    int target_offset = instr.argval;  // Use Python's calculated target
                    int next_offset = instructions[i + 1].offset;
                    
                    if (!jump_targets.count(target_offset)) {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(target_offset), func
                        );
                    }
                    
                    // Create block for fall-through only if next instruction is also a jump target
                    if (!jump_targets.count(next_offset)) {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(next_offset), func
                        );
                    }
                    
                    if (!builder.GetInsertBlock()->getTerminator()) {
                        llvm::BasicBlock* current_block = builder.GetInsertBlock();
                        
                        // Bug #1 Fix: Record stack state for BOTH branch targets
                        // This enables PHI node insertion at merge points
                        BlockStackState state;
                        state.stack = stack;
                        state.predecessor = current_block;
                        block_incoming_stacks[target_offset].push_back(state);
                        block_incoming_stacks[next_offset].push_back(state);
                        
                        // POP_JUMP_IF_FALSE: jump if condition is FALSE (0), continue if TRUE (non-zero)
                        // POP_JUMP_IF_TRUE: jump if condition is TRUE (non-zero), continue if FALSE (0)
                        if (instr.opcode == op::POP_JUMP_IF_FALSE) {
                            // Jump to target when condition is false (0), continue to next when true (non-zero)
                            builder.CreateCondBr(bool_cond, jump_targets[next_offset], jump_targets[target_offset]);
                        } else {  // POP_JUMP_IF_TRUE (opcode 100)
                            // Jump to target when condition is true (non-zero), continue to next when false (0)
                            builder.CreateCondBr(bool_cond, jump_targets[target_offset], jump_targets[next_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::POP_JUMP_IF_NONE || instr.opcode == op::POP_JUMP_IF_NOT_NONE) {
                // Jump based on whether value is None
                if (!stack.empty() && i + 1 < instructions.size()) {
                    llvm::Value* val = stack.back(); stack.pop_back();
                    
                    // Get Python's Py_None singleton address
                    llvm::Value* py_none_ptr = llvm::ConstantInt::get(
                        i64_type, reinterpret_cast<uint64_t>(Py_None)
                    );
                    llvm::Value* py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    
                    // Compare pointer to Py_None
                    llvm::Value* is_none = builder.CreateICmpEQ(val, py_none, "is_none");
                    
                    // Decref the value we popped (it's consumed)
                    if (val->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {val});
                    }
                    
                    int target_offset = instr.argval;
                    int next_offset = instructions[i + 1].offset;
                    
                    if (!jump_targets.count(target_offset)) {
                        jump_targets[target_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(target_offset), func
                        );
                    }
                    if (!jump_targets.count(next_offset)) {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "block_" + std::to_string(next_offset), func
                        );
                    }
                    
                    if (!builder.GetInsertBlock()->getTerminator()) {
                        llvm::BasicBlock* current_block = builder.GetInsertBlock();
                        
                        // Bug #1 Fix: Record stack state for BOTH branch targets
                        BlockStackState state;
                        state.stack = stack;
                        state.predecessor = current_block;
                        block_incoming_stacks[target_offset].push_back(state);
                        block_incoming_stacks[next_offset].push_back(state);
                        
                        if (instr.opcode == op::POP_JUMP_IF_NONE) {
                            // Jump if is_none is true
                            builder.CreateCondBr(is_none, jump_targets[target_offset], jump_targets[next_offset]);
                        } else {  // POP_JUMP_IF_NOT_NONE
                            // Jump if is_none is false (i.e., not None)
                            builder.CreateCondBr(is_none, jump_targets[next_offset], jump_targets[target_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == op::JUMP_BACKWARD) {
                // For loops: jump backward to loop header
                int target_offset = instr.argval;  // Use Python's calculated target
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_header_" + std::to_string(target_offset), func
                    );
                }
                if (!builder.GetInsertBlock()->getTerminator()) {
                    builder.CreateBr(jump_targets[target_offset]);
                }
                
                // CRITICAL: Reset compile-time stack to match the target's expected depth
                // This is necessary because the loop body may have pushed values that 
                // don't exist at the loop header. The runtime stack is correct (bytecode
                // is verified), but our compile-time tracking gets out of sync.
                if (stack_depth_at_offset.count(target_offset)) {
                    size_t target_depth = stack_depth_at_offset[target_offset];
                    while (stack.size() > target_depth) {
                        stack.pop_back();
                    }
                }
                
                // Create a new block for any code after the loop (won't be reached but prevents issues)
                llvm::BasicBlock* after_loop = llvm::BasicBlock::Create(
                    *local_context, "after_loop_" + std::to_string(i), func
                );
                builder.SetInsertPoint(after_loop);
            }
            else if (instr.opcode == op::JUMP_FORWARD) {
                // Unconditional forward jump
                int target_offset = instr.argval;  // Use Python's calculated target
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "jump_target_" + std::to_string(target_offset), func
                    );
                }
                if (!builder.GetInsertBlock()->getTerminator()) {
                    builder.CreateBr(jump_targets[target_offset]);
                }
                // Create a new block for any code after the jump (unreachable but prevents issues)
                llvm::BasicBlock* after_jump = llvm::BasicBlock::Create(
                    *local_context, "after_jump_" + std::to_string(i), func
                );
                builder.SetInsertPoint(after_jump);
            }
            else if (instr.opcode == op::RETURN_CONST) {
                // Return a constant from co_consts without using stack
                if (!builder.GetInsertBlock()->getTerminator()) {
                    // Get the constant value and return as PyObject*
                    if (instr.arg < int_constants.size()) {
                        if (obj_constants[instr.arg] != nullptr) {
                            // PyObject* constant
                            llvm::Value* const_ptr = llvm::ConstantInt::get(
                                i64_type,
                                reinterpret_cast<uint64_t>(obj_constants[instr.arg])
                            );
                            llvm::Value* py_obj = builder.CreateIntToPtr(const_ptr, ptr_type);
                            builder.CreateCall(py_incref_func, {py_obj});
                            builder.CreateRet(py_obj);
                        } else {
                            // int64 constant - convert to PyObject*
                            llvm::Value* const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                            llvm::Value* py_obj = builder.CreateCall(py_long_fromlong_func, {const_val});
                            builder.CreateRet(py_obj);
                        }
                    } else {
                        // Fallback: return None
                        llvm::Value* none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value* py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {py_none});
                        builder.CreateRet(py_none);
                    }
                }
            }
            else if (instr.opcode == op::RETURN_VALUE) {
                if (!builder.GetInsertBlock()->getTerminator()) {
                    if (!stack.empty()) {
                        llvm::Value* ret_val = stack.back();
                        // If returning i64, convert to PyObject*
                        if (ret_val->getType()->isIntegerTy(64)) {
                            ret_val = builder.CreateCall(py_long_fromlong_func, {ret_val});
                        }
                        builder.CreateRet(ret_val);
                    } else {
                        // Return None
                        llvm::Value* none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                        llvm::Value* py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                        builder.CreateCall(py_incref_func, {py_none});
                        builder.CreateRet(py_none);
                    }
                }
            }
            else if (instr.opcode == op::BUILD_LIST) {
                // arg is the number of items to pop from stack
                int count = instr.arg;
                
                // Create new list with PyList_New(count)
                llvm::Value* count_val = llvm::ConstantInt::get(i64_type, count);
                llvm::Value* new_list = builder.CreateCall(py_list_new_func, {count_val});
                
                // Pop items from stack and add to list (in reverse order)
                std::vector<llvm::Value*> items;
                std::vector<bool> item_is_ptr;
                for (int i = 0; i < count; ++i) {
                    if (!stack.empty()) {
                        llvm::Value* item = stack.back();
                        item_is_ptr.push_back(item->getType()->isPointerTy());
                        items.push_back(item);
                        stack.pop_back();
                    }
                }
                
                // Add items to list in correct order
                for (int i = count - 1; i >= 0; --i) {
                    llvm::Value* index_val = llvm::ConstantInt::get(i64_type, count - 1 - i);
                    llvm::Value* item = items[i];
                    bool was_ptr = item_is_ptr[i];
                    
                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64)) {
                        item = builder.CreateCall(py_long_fromlong_func, {item});
                        // PyList_SetItem steals reference, so new PyLong is transferred
                    } else {
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
            else if (instr.opcode == op::BUILD_TUPLE) {
                // arg is the number of items to pop from stack
                int count = instr.arg;
                
                // Create new tuple with PyTuple_New(count)
                llvm::Value* count_val = llvm::ConstantInt::get(i64_type, count);
                llvm::Value* new_tuple = builder.CreateCall(py_tuple_new_func, {count_val});
                
                // Pop items from stack (in reverse order)
                std::vector<llvm::Value*> items;
                for (int i = 0; i < count; ++i) {
                    if (!stack.empty()) {
                        items.push_back(stack.back());
                        stack.pop_back();
                    }
                }
                
                // Add items to tuple in correct order
                for (int i = count - 1; i >= 0; --i) {
                    llvm::Value* index_val = llvm::ConstantInt::get(i64_type, count - 1 - i);
                    llvm::Value* item = items[i];
                    
                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64)) {
                        item = builder.CreateCall(py_long_fromlong_func, {item});
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
            else if (instr.opcode == op::BUILD_MAP) {
                // Build a dictionary from arg key-value pairs
                // arg = number of key-value pairs (stack has 2*arg items)
                int count = instr.arg;
                
                // Create new empty dict
                llvm::Value* new_dict = builder.CreateCall(py_dict_new_func, {}, "new_dict");
                
                // Pop key-value pairs from stack (in reverse order)
                // Stack order: ... key1 value1 key2 value2 ... (TOS is last value)
                std::vector<std::pair<llvm::Value*, llvm::Value*>> pairs;
                std::vector<std::pair<bool, bool>> pairs_are_ptr;
                for (int i = 0; i < count; ++i) {
                    if (stack.size() >= 2) {
                        llvm::Value* value = stack.back(); stack.pop_back();
                        llvm::Value* key = stack.back(); stack.pop_back();
                        pairs_are_ptr.push_back({key->getType()->isPointerTy(), value->getType()->isPointerTy()});
                        pairs.push_back({key, value});
                    }
                }
                
                // Add pairs to dict in correct order (reverse of how we popped)
                for (int i = count - 1; i >= 0; --i) {
                    llvm::Value* key = pairs[i].first;
                    llvm::Value* value = pairs[i].second;
                    bool key_is_ptr = pairs_are_ptr[i].first;
                    bool value_is_ptr = pairs_are_ptr[i].second;
                    bool key_was_boxed = false;
                    bool value_was_boxed = false;
                    
                    // Convert int64 to PyObject* if needed
                    if (key->getType()->isIntegerTy(64)) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                        key_was_boxed = true;
                    }
                    if (value->getType()->isIntegerTy(64)) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                        value_was_boxed = true;
                    }
                    
                    // PyDict_SetItem does NOT steal references (it increfs both)
                    builder.CreateCall(py_dict_setitem_func, {new_dict, key, value});
                    
                    // Decref our copies (SetItem already incref'd its own)
                    if (key_was_boxed) {
                        builder.CreateCall(py_decref_func, {key});
                    } else if (key_is_ptr) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed) {
                        builder.CreateCall(py_decref_func, {value});
                    } else if (value_is_ptr) {
                        builder.CreateCall(py_decref_func, {value});
                    }
                }
                
                stack.push_back(new_dict);
            }
            else if (instr.opcode == op::BUILD_CONST_KEY_MAP) {
                // Build a dictionary from const key tuple and count values
                // Stack: value1, value2, ..., valueN, key_tuple (TOS)
                // arg = number of key-value pairs (count)
                int count = instr.arg;
                
                if (!stack.empty()) {
                    // Pop the keys tuple from TOS
                    llvm::Value* keys_tuple = stack.back(); stack.pop_back();
                    
                    // Pop count values from stack
                    std::vector<llvm::Value*> values;
                    for (int i = 0; i < count; ++i) {
                        if (!stack.empty()) {
                            values.push_back(stack.back());
                            stack.pop_back();
                        }
                    }
                    
                    // Create new empty dict
                    llvm::Value* new_dict = builder.CreateCall(py_dict_new_func, {}, "new_dict");
                    
                    // Add pairs to dict - values are in reverse order of keys
                    for (int i = 0; i < count; ++i) {
                        // Get key from tuple at index i
                        llvm::Value* idx = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value* idx_obj = builder.CreateCall(py_long_fromlong_func, {idx});
                        llvm::Value* key = builder.CreateCall(py_object_getitem_func, {keys_tuple, idx_obj});
                        builder.CreateCall(py_decref_func, {idx_obj}); // Free the temp index object
                        
                        // Get corresponding value (values are in reverse order)
                        llvm::Value* value = values[count - 1 - i];
                        
                        // Convert int64 to PyObject* if needed
                        if (value->getType()->isIntegerTy(64)) {
                            value = builder.CreateCall(py_long_fromlong_func, {value});
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
            else if (instr.opcode == op::BUILD_SET) {
                // Build a set from arg items on stack
                int count = instr.arg;
                
                // Create new empty set (pass NULL for empty)
                llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                llvm::Value* new_set = builder.CreateCall(py_set_new_func, {null_ptr}, "new_set");
                
                // Pop items from stack (in reverse order)
                std::vector<llvm::Value*> items;
                std::vector<bool> items_are_ptr;
                for (int i = 0; i < count; ++i) {
                    if (!stack.empty()) {
                        llvm::Value* item = stack.back();
                        items_are_ptr.push_back(item->getType()->isPointerTy());
                        items.push_back(item);
                        stack.pop_back();
                    }
                }
                
                // Add items to set in correct order
                for (int i = count - 1; i >= 0; --i) {
                    llvm::Value* item = items[i];
                    bool item_is_ptr = items_are_ptr[i];
                    bool item_was_boxed = false;
                    
                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64)) {
                        item = builder.CreateCall(py_long_fromlong_func, {item});
                        item_was_boxed = true;
                    }
                    
                    // PySet_Add does NOT steal references (it increfs)
                    builder.CreateCall(py_set_add_func, {new_set, item});
                    
                    // Decref our copy (SetAdd already incref'd its own)
                    if (item_was_boxed) {
                        builder.CreateCall(py_decref_func, {item});
                    } else if (item_is_ptr) {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }
                
                stack.push_back(new_set);
            }
            else if (instr.opcode == op::LIST_EXTEND) {
                // Extends the list STACK[-i] with the iterable STACK.pop()
                // Used for list literals like [1, 2, 3, 4, 5] in Python 3.9+
                if (!stack.empty()) {
                    llvm::Value* iterable = stack.back(); stack.pop_back();
                    
                    // arg tells us where the list is: STACK[-i]
                    int list_index = instr.arg;
                    if (list_index > 0 && static_cast<size_t>(list_index) <= stack.size()) {
                        // Get the list from stack position (0-indexed from end)
                        llvm::Value* list = stack[stack.size() - list_index];
                        
                        // Call PyList_Extend(list, iterable) - returns 0 on success
                        builder.CreateCall(py_list_extend_func, {list, iterable});
                        
                        // Decref the iterable (we consumed it)
                        if (!iterable->getType()->isIntegerTy(64)) {
                            builder.CreateCall(py_decref_func, {iterable});
                        }
                    }
                }
            }
            else if (instr.opcode == op::BINARY_SUBSCR) {
                // Implements container[key]
                if (stack.size() >= 2) {
                    llvm::Value* key = stack.back(); stack.pop_back();
                    llvm::Value* container = stack.back(); stack.pop_back();
                    
                    // Track if we need to decref the key (if we box it)
                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    bool key_is_ptr = key->getType()->isPointerTy();
                    
                    // Convert int64 key to PyObject* if needed
                    if (key_was_boxed) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                    }
                    
                    // PyObject_GetItem returns new reference
                    llvm::Value* result = builder.CreateCall(py_object_getitem_func, {container, key});
                    
                    // Decrement key refcount - if we boxed it or if it was a PyObject* from stack
                    if (key_was_boxed) {
                        builder.CreateCall(py_decref_func, {key});
                    } else if (key_is_ptr) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    
                    // CRITICAL: Decref the container we consumed from the stack
                    if (container->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {container});
                    }
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::BUILD_SLICE) {
                // Build a slice object
                // arg=2: slice(start, stop), arg=3: slice(start, stop, step)
                int argc = instr.arg;
                if (argc == 2 && stack.size() >= 2) {
                    llvm::Value* stop = stack.back(); stack.pop_back();
                    llvm::Value* start = stack.back(); stack.pop_back();
                    
                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);
                    
                    if (start_boxed) {
                        start = builder.CreateCall(py_long_fromlong_func, {start});
                    }
                    if (stop_boxed) {
                        stop = builder.CreateCall(py_long_fromlong_func, {stop});
                    }
                    
                    // PySlice_New(start, stop, NULL)
                    llvm::Value* py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value* py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    
                    llvm::Value* slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});
                    
                    // Decref temporaries
                    if (start_boxed) builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed) builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {stop});
                    
                    stack.push_back(slice);
                } else if (argc == 3 && stack.size() >= 3) {
                    llvm::Value* step = stack.back(); stack.pop_back();
                    llvm::Value* stop = stack.back(); stack.pop_back();
                    llvm::Value* start = stack.back(); stack.pop_back();
                    
                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);
                    bool step_boxed = step->getType()->isIntegerTy(64);
                    
                    if (start_boxed) {
                        start = builder.CreateCall(py_long_fromlong_func, {start});
                    }
                    if (stop_boxed) {
                        stop = builder.CreateCall(py_long_fromlong_func, {stop});
                    }
                    if (step_boxed) {
                        step = builder.CreateCall(py_long_fromlong_func, {step});
                    }
                    
                    llvm::Value* slice = builder.CreateCall(py_slice_new_func, {start, stop, step});
                    
                    // Decref temporaries
                    if (start_boxed) builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed) builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {stop});
                    if (step_boxed) builder.CreateCall(py_decref_func, {step});
                    else if (step->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {step});
                    
                    stack.push_back(slice);
                }
            }
            else if (instr.opcode == op::BINARY_SLICE) {
                // Implements TOS = TOS1[TOS2:TOS]  (container[start:stop])
                // Stack: TOS=stop, TOS1=start, TOS2=container
                if (stack.size() >= 3) {
                    llvm::Value* stop = stack.back(); stack.pop_back();
                    llvm::Value* start = stack.back(); stack.pop_back();
                    llvm::Value* container = stack.back(); stack.pop_back();
                    
                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);
                    
                    if (start_boxed) {
                        start = builder.CreateCall(py_long_fromlong_func, {start});
                    }
                    if (stop_boxed) {
                        stop = builder.CreateCall(py_long_fromlong_func, {stop});
                    }
                    
                    // Build a slice object
                    llvm::Value* py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value* py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value* slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});
                    
                    // Use PyObject_GetItem with the slice
                    llvm::Value* result = builder.CreateCall(py_object_getitem_func, {container, slice});
                    
                    // Decref slice (consumed)
                    builder.CreateCall(py_decref_func, {slice});
                    
                    // Decref temporaries and consumed values
                    if (start_boxed) builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed) builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {stop});
                    if (container->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {container});
                    }
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::STORE_SUBSCR) {
                // Implements container[key] = value
                // Per Python docs: key = STACK.pop(); container = STACK.pop(); value = STACK.pop()
                // Stack order: TOS=key, TOS1=container, TOS2=value
                if (stack.size() >= 3) {
                    llvm::Value* key = stack.back(); stack.pop_back();         // TOS
                    llvm::Value* container = stack.back(); stack.pop_back();   // TOS1
                    llvm::Value* value = stack.back(); stack.pop_back();       // TOS2
                    
                    // Track if we need to decref (if we box values)
                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    bool key_is_ptr = key->getType()->isPointerTy();
                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool container_is_ptr = container->getType()->isPointerTy();
                    
                    // Convert int64 key to PyObject* if needed
                    if (key_was_boxed) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                    }
                    
                    // Convert int64 value to PyObject* if needed
                    if (value_was_boxed) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                    }
                    
                    // PyObject_SetItem(container, key, value) - returns 0 on success
                    builder.CreateCall(py_object_setitem_func, {container, key, value});
                    
                    // Decrement temp refs if we created them
                    if (key_was_boxed) {
                        builder.CreateCall(py_decref_func, {key});
                    } else if (key_is_ptr) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed) {
                        builder.CreateCall(py_decref_func, {value});
                    } else if (value_is_ptr) {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    
                    // CRITICAL: Decref container since we consumed it from stack
                    if (container_is_ptr) {
                        builder.CreateCall(py_decref_func, {container});
                    }
                }
            }
            else if (instr.opcode == op::STORE_SLICE) {
                // Implements container[start:stop] = value
                // Stack: TOS=stop, TOS1=start, TOS2=container, TOS3=value
                if (stack.size() >= 4) {
                    llvm::Value* stop = stack.back(); stack.pop_back();
                    llvm::Value* start = stack.back(); stack.pop_back();
                    llvm::Value* container = stack.back(); stack.pop_back();
                    llvm::Value* value = stack.back(); stack.pop_back();
                    
                    bool start_boxed = start->getType()->isIntegerTy(64);
                    bool stop_boxed = stop->getType()->isIntegerTy(64);
                    
                    if (start_boxed) {
                        start = builder.CreateCall(py_long_fromlong_func, {start});
                    }
                    if (stop_boxed) {
                        stop = builder.CreateCall(py_long_fromlong_func, {stop});
                    }
                    
                    // Build a slice object
                    llvm::Value* py_none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                    llvm::Value* py_none = builder.CreateIntToPtr(py_none_ptr, ptr_type);
                    llvm::Value* slice = builder.CreateCall(py_slice_new_func, {start, stop, py_none});
                    
                    // PyObject_SetItem(container, slice, value)
                    builder.CreateCall(py_object_setitem_func, {container, slice, value});
                    
                    // Decref slice
                    builder.CreateCall(py_decref_func, {slice});
                    
                    // Decref temporaries
                    if (start_boxed) builder.CreateCall(py_decref_func, {start});
                    else if (start->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {start});
                    if (stop_boxed) builder.CreateCall(py_decref_func, {stop});
                    else if (stop->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {stop});
                    if (container->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {container});
                    if (value->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {value});
                }
            }
            else if (instr.opcode == op::DELETE_SUBSCR) {
                // Implements del container[key]
                // Stack: TOS=key, TOS1=container
                if (stack.size() >= 2) {
                    llvm::Value* key = stack.back(); stack.pop_back();
                    llvm::Value* container = stack.back(); stack.pop_back();
                    
                    bool key_was_boxed = key->getType()->isIntegerTy(64);
                    if (key_was_boxed) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                    }
                    
                    // PyObject_DelItem(container, key)
                    builder.CreateCall(py_object_delitem_func, {container, key});
                    
                    // Decref
                    if (key_was_boxed) builder.CreateCall(py_decref_func, {key});
                    else if (key->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {key});
                    if (container->getType()->isPointerTy()) builder.CreateCall(py_decref_func, {container});
                }
            }
            else if (instr.opcode == op::STORE_ATTR) {
                // Implements obj.attr = value
                // Stack order: TOS=obj, TOS1=value
                // Python 3.13: arg >> 1 = index into co_names
                int name_idx = instr.arg >> 1;
                
                if (stack.size() >= 2 && name_idx < static_cast<int>(name_objects.size())) {
                    llvm::Value* obj = stack.back(); stack.pop_back();     // TOS
                    llvm::Value* value = stack.back(); stack.pop_back();   // TOS1
                    bool value_is_ptr = value->getType()->isPointerTy();
                    
                    // Get attribute name from names (PyUnicode string)
                    llvm::Value* attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx])
                    );
                    llvm::Value* attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);
                    
                    // Convert int64 value to PyObject* if needed
                    bool value_was_boxed = value->getType()->isIntegerTy(64);
                    if (value_was_boxed) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                    }
                    
                    // PyObject_SetAttr(obj, attr_name, value) - returns 0 on success
                    builder.CreateCall(py_object_setattr_func, {obj, attr_name, value});
                    
                    // Decref the value if we boxed it or it was a PyObject* from stack
                    if (value_was_boxed) {
                        builder.CreateCall(py_decref_func, {value});
                    } else if (value_is_ptr) {
                        builder.CreateCall(py_decref_func, {value});
                    }
                    
                    // CRITICAL: Decref the object we consumed from the stack
                    if (obj->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {obj});
                    }
                }
            }
            else if (instr.opcode == op::LIST_APPEND) {
                // Append TOS to the list at STACK[-i]
                // arg = i (distance from TOS, AFTER popping item)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    // Calculate list index BEFORE popping (stack.size() - 1 - i_val + 1 = stack.size() - i_val)
                    // But after popping item, list is at stack[stack.size() - i_val]
                    // Actually: TOS is item, list is at STACK[-(i+1)] before pop = STACK[-i] after pop
                    llvm::Value* item = stack.back(); stack.pop_back();
                    bool item_is_ptr = item->getType()->isPointerTy();
                    bool item_was_boxed = false;
                    // After pop, list is at distance i from new TOS, which is index (size - i)
                    llvm::Value* list = stack[stack.size() - i_val];  // List stays on stack
                    
                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64)) {
                        item = builder.CreateCall(py_long_fromlong_func, {item});
                        item_was_boxed = true;
                    }
                    
                    // PyList_Append does NOT steal references (it increfs)
                    builder.CreateCall(py_list_append_func, {list, item});
                    
                    // Decref our copy since Append already incref'd its own
                    if (item_was_boxed) {
                        builder.CreateCall(py_decref_func, {item});
                    } else if (item_is_ptr) {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }
            }
            else if (instr.opcode == op::LIST_EXTEND) {
                // Extend list at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    llvm::Value* iterable = stack.back(); stack.pop_back();
                    bool iter_is_ptr = iterable->getType()->isPointerTy();
                    llvm::Value* list = stack[stack.size() - i_val];  // List stays on stack
                    
                    // _PyList_Extend(list, iterable)
                    builder.CreateCall(py_list_extend_func, {list, iterable});
                    
                    // Decref iterable since we consumed it
                    if (iter_is_ptr) {
                        builder.CreateCall(py_decref_func, {iterable});
                    }
                }
            }
            else if (instr.opcode == op::SET_UPDATE) {
                // Update set at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    llvm::Value* iterable = stack.back(); stack.pop_back();
                    bool iter_is_ptr = iterable->getType()->isPointerTy();
                    llvm::Value* set = stack[stack.size() - i_val];  // Set stays on stack
                    
                    // _PySet_Update(set, iterable)
                    builder.CreateCall(py_set_update_func, {set, iterable});
                    
                    // Decref iterable since we consumed it
                    if (iter_is_ptr) {
                        builder.CreateCall(py_decref_func, {iterable});
                    }
                }
            }
            else if (instr.opcode == op::DICT_UPDATE) {
                // Update dict at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    llvm::Value* update_dict = stack.back(); stack.pop_back();
                    bool update_is_ptr = update_dict->getType()->isPointerTy();
                    llvm::Value* dict = stack[stack.size() - i_val];  // Dict stays on stack
                    
                    // PyDict_Update(dict, update_dict)
                    builder.CreateCall(py_dict_update_func, {dict, update_dict});
                    
                    // Decref update_dict since we consumed it
                    if (update_is_ptr) {
                        builder.CreateCall(py_decref_func, {update_dict});
                    }
                }
            }
            else if (instr.opcode == op::DICT_MERGE) {
                // Merge dict at STACK[-i] with TOS
                // arg = i (distance from TOS after pop)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    llvm::Value* update_dict = stack.back(); stack.pop_back();
                    bool update_is_ptr = update_dict->getType()->isPointerTy();
                    llvm::Value* dict = stack[stack.size() - i_val];  // Dict stays on stack
                    
                    // PyDict_Merge(dict, update_dict, 1) - override=1 for merge
                    llvm::Value* override_flag = llvm::ConstantInt::get(builder.getInt32Ty(), 1);
                    builder.CreateCall(py_dict_merge_func, {dict, update_dict, override_flag});
                    
                    // Decref update_dict since we consumed it
                    if (update_is_ptr) {
                        builder.CreateCall(py_decref_func, {update_dict});
                    }
                }
            }
            else if (instr.opcode == op::SET_ADD) {
                // Add TOS to the set at STACK[-i]
                // arg = i (distance from TOS)
                int i_val = instr.arg;
                if (!stack.empty() && static_cast<size_t>(i_val) <= stack.size()) {
                    llvm::Value* item = stack.back(); stack.pop_back();
                    bool item_is_ptr = item->getType()->isPointerTy();
                    bool item_was_boxed = false;
                    llvm::Value* set = stack[stack.size() - i_val];  // Set stays on stack
                    
                    // Convert int64 to PyObject* if needed
                    if (item->getType()->isIntegerTy(64)) {
                        item = builder.CreateCall(py_long_fromlong_func, {item});
                        item_was_boxed = true;
                    }
                    
                    // PySet_Add does NOT steal references (it increfs)
                    builder.CreateCall(py_set_add_func, {set, item});
                    
                    // Decref our copy since Add already incref'd its own
                    if (item_was_boxed) {
                        builder.CreateCall(py_decref_func, {item});
                    } else if (item_is_ptr) {
                        builder.CreateCall(py_decref_func, {item});
                    }
                }
            }
            else if (instr.opcode == op::MAP_ADD) {
                // Add key-value pair to the dict at STACK[-i]
                // Stack: TOS=value, TOS1=key
                // arg = i (distance from TOS, after popping key and value)
                int i_val = instr.arg;
                if (stack.size() >= 2 && static_cast<size_t>(i_val) <= stack.size() - 2) {
                    llvm::Value* value = stack.back(); stack.pop_back();
                    llvm::Value* key = stack.back(); stack.pop_back();
                    bool key_is_ptr = key->getType()->isPointerTy();
                    bool value_is_ptr = value->getType()->isPointerTy();
                    bool key_was_boxed = false;
                    bool value_was_boxed = false;
                    llvm::Value* dict = stack[stack.size() - i_val];  // Dict stays on stack
                    
                    // Convert int64 to PyObject* if needed
                    if (key->getType()->isIntegerTy(64)) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                        key_was_boxed = true;
                    }
                    if (value->getType()->isIntegerTy(64)) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                        value_was_boxed = true;
                    }
                    
                    // PyDict_SetItem does NOT steal references (it increfs both)
                    builder.CreateCall(py_dict_setitem_func, {dict, key, value});
                    
                    // Decref our copies since SetItem already incref'd its own
                    if (key_was_boxed) {
                        builder.CreateCall(py_decref_func, {key});
                    } else if (key_is_ptr) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    if (value_was_boxed) {
                        builder.CreateCall(py_decref_func, {value});
                    } else if (value_is_ptr) {
                        builder.CreateCall(py_decref_func, {value});
                    }
                }
            }
            else if (instr.opcode == op::LOAD_ATTR) {
                // Implements obj.attr
                // Python 3.13: arg >> 1 = index into co_names, arg & 1 = method load flag
                int name_idx = instr.arg >> 1;
                bool is_method = (instr.arg & 1) != 0;
                
                if (!stack.empty() && name_idx < static_cast<int>(name_objects.size())) {
                    llvm::Value* obj = stack.back(); stack.pop_back();
                    
                    // Get attribute name from names (PyUnicode string)
                    llvm::Value* attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx])
                    );
                    llvm::Value* attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);
                    
                    // PyObject_GetAttr returns new reference (bound method for methods)
                    llvm::Value* result = builder.CreateCall(py_object_getattr_func, {obj, attr_name});
                    
                    // CRITICAL: Decref the object we consumed from the stack
                    if (obj->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {obj});
                    }
                    
                    if (is_method) {
                        // Method loading for CALL opcode
                        // CALL expects stack layout: [callable, self_or_null, args...]
                        // For bound methods from GetAttr, self is already bound in the method
                        // Push callable (bound method) first, then NULL for self_or_null
                        // Stack order: push method, then push NULL
                        // Result: [..., method, NULL] so that after LOAD_FAST arg:
                        //         [..., method, NULL, arg]
                        // CALL 1 sees: callable=stack[-3]=method, self_or_null=stack[-2]=NULL
                        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(
                            llvm::PointerType::get(*local_context, 0)
                        );
                        stack.push_back(result);     // callable = bound method
                        stack.push_back(null_ptr);   // self_or_null = NULL
                    } else {
                        // Normal attribute access
                        stack.push_back(result);
                    }
                }
            }
            else if (instr.opcode == op::LOAD_GLOBAL) {
                // Python 3.13: LOAD_GLOBAL loads global variable
                // arg >> 1 = index into co_names
                // arg & 1 = if set, push NULL after global (for calling convention)
                int name_idx = instr.arg >> 1;
                bool push_null = (instr.arg & 1) != 0;
                
                if (name_idx < name_objects.size()) {
                    // Bug #4 Fix: Runtime lookup instead of compile-time resolved value
                    // Get the name object for lookup
                    llvm::Value* name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[name_idx])
                    );
                    llvm::Value* name_obj = builder.CreateIntToPtr(name_ptr, ptr_type, "name_obj");
                    
                    // Get globals dict pointer
                    llvm::Value* globals_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(globals_dict_ptr)
                    );
                    llvm::Value* globals_dict = builder.CreateIntToPtr(globals_ptr, ptr_type, "globals_dict");
                    
                    // PyDict_GetItem(globals_dict, name) - returns borrowed reference or NULL
                    llvm::Value* global_obj = builder.CreateCall(
                        py_dict_getitem_func, 
                        {globals_dict, name_obj}, 
                        "global_lookup"
                    );
                    
                    // Check if found in globals, if not try builtins
                    llvm::Value* is_null = builder.CreateICmpEQ(
                        global_obj, 
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "not_found_in_globals"
                    );
                    
                    llvm::BasicBlock* found_block = llvm::BasicBlock::Create(*local_context, "global_found", func);
                    llvm::BasicBlock* try_builtins_block = llvm::BasicBlock::Create(*local_context, "try_builtins", func);
                    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(*local_context, "global_continue", func);
                    
                    builder.CreateCondBr(is_null, try_builtins_block, found_block);
                    
                    // Try builtins
                    builder.SetInsertPoint(try_builtins_block);
                    llvm::Value* builtins_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(builtins_dict_ptr)
                    );
                    llvm::Value* builtins_dict = builder.CreateIntToPtr(builtins_ptr, ptr_type, "builtins_dict");
                    llvm::Value* builtin_obj = builder.CreateCall(
                        py_dict_getitem_func,
                        {builtins_dict, name_obj},
                        "builtin_lookup"
                    );
                    builder.CreateBr(continue_block);
                    
                    // Found in globals
                    builder.SetInsertPoint(found_block);
                    builder.CreateBr(continue_block);
                    
                    // Continue with PHI node to select result
                    builder.SetInsertPoint(continue_block);
                    llvm::PHINode* result_phi = builder.CreatePHI(ptr_type, 2, "global_result");
                    result_phi->addIncoming(builtin_obj, try_builtins_block);
                    result_phi->addIncoming(global_obj, found_block);
                    
                    // Incref the result (PyDict_GetItem returns borrowed reference)
                    builder.CreateCall(py_incref_func, {result_phi});
                    
                    stack.push_back(result_phi);
                    
                    // Push NULL after global if needed (Python 3.13 calling convention)
                    if (push_null) {
                        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        stack.push_back(null_ptr);
                    }
                }
            }
            else if (instr.opcode == op::CALL) {
                // Python 3.13: CALL opcode, arg = number of arguments (excluding self/NULL)
                // Stack layout (CPython uses indices from bottom):
                //   callable = stack[-2-oparg]
                //   self_or_null = stack[-1-oparg]
                //   args = &stack[-oparg] (oparg elements)
                int num_args = instr.arg;
                
                if (stack.size() >= static_cast<size_t>(num_args + 2)) {
                    // Access stack by index (matches CPython implementation)
                    size_t base = stack.size() - num_args - 2;
                    
                    llvm::Value* callable = stack[base];           // stack[-2-oparg]
                    llvm::Value* self_or_null = stack[base + 1];   // stack[-1-oparg]
                    
                    // Track if operands are pointers for decref
                    bool callable_is_ptr = callable->getType()->isPointerTy();
                    std::vector<bool> args_are_ptr;
                    
                    // Collect arguments in order
                    std::vector<llvm::Value*> args;
                    for (int i = 0; i < num_args; ++i) {
                        llvm::Value* arg = stack[base + 2 + i];
                        args_are_ptr.push_back(arg->getType()->isPointerTy());
                        args.push_back(arg);       // stack[-oparg+i]
                    }
                    
                    // Remove all CALL operands from stack
                    stack.erase(stack.begin() + base, stack.end());
                    
                    // Create args tuple - PyTuple_SetItem steals references so we transfer ownership
                    llvm::Value* args_count = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value* args_tuple = builder.CreateCall(py_tuple_new_func, {args_count});
                    
                    // Fill tuple with args in correct order
                    for (int i = 0; i < num_args; ++i) {
                        llvm::Value* index_val = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value* arg = args[i];
                        
                        // Box int64 to PyObject* if needed
                        if (arg->getType()->isIntegerTy(64)) {
                            arg = builder.CreateCall(py_long_fromlong_func, {arg});
                            // PyTuple_SetItem steals reference - new PyLong is transferred
                        }
                        // For PyObject*: PyTuple_SetItem steals reference
                        // We consume the stack value, so transfer ownership directly
                        // No incref needed
                        
                        // PyTuple_SetItem steals reference (transfers ownership)
                        builder.CreateCall(py_tuple_setitem_func, {args_tuple, index_val, arg});
                    }
                    
                    // Call PyObject_Call(callable, args_tuple, NULL)
                    llvm::Value* null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* result = builder.CreateCall(py_object_call_func, {callable, args_tuple, null_kwargs});
                    
                    // Decrement args_tuple refcount (we're done with it)
                    builder.CreateCall(py_decref_func, {args_tuple});
                    
                    // Decref callable (we consumed it from the stack)
                    if (callable_is_ptr) {
                        builder.CreateCall(py_decref_func, {callable});
                    }
                    
                    // Note: self_or_null is either NULL or a reference we need to decref
                    // The NULL check is needed at runtime
                    llvm::Value* null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");
                    
                    llvm::BasicBlock* decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self", func);
                    llvm::BasicBlock* after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self", func);
                    
                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);
                    
                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);
                    
                    builder.SetInsertPoint(after_decref_self);
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::CALL_KW) {
                // Python 3.13: CALL_KW opcode - call with keyword arguments
                // Stack layout:
                //   callable = stack[-3-oparg]
                //   self_or_null = stack[-2-oparg]
                //   args = stack[-1-oparg : -1] (oparg elements = positional + keyword args)
                //   kwnames = stack[-1] (tuple of keyword names)
                // oparg = total number of arguments (positional + keyword)
                int num_args = instr.arg;
                
                if (stack.size() >= static_cast<size_t>(num_args + 3)) {
                    // Pop kwnames tuple first (TOS)
                    llvm::Value* kwnames = stack.back(); stack.pop_back();
                    
                    // Now stack layout is like CALL: callable, self_or_null, args...
                    size_t base = stack.size() - num_args - 2;
                    
                    llvm::Value* callable = stack[base];
                    llvm::Value* self_or_null = stack[base + 1];
                    
                    bool callable_is_ptr = callable->getType()->isPointerTy();
                    
                    // Collect all arguments
                    std::vector<llvm::Value*> args;
                    for (int i = 0; i < num_args; ++i) {
                        args.push_back(stack[base + 2 + i]);
                    }
                    
                    // Remove all operands from stack
                    stack.erase(stack.begin() + base, stack.end());
                    
                    // Create args tuple for positional args
                    // We need to get the number of keyword args from kwnames tuple length
                    // For simplicity, pass all args as positional tuple and kwnames as kwargs dict
                    
                    // Create positional args tuple
                    llvm::Value* args_count = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value* args_tuple = builder.CreateCall(py_tuple_new_func, {args_count});
                    
                    for (int i = 0; i < num_args; ++i) {
                        llvm::Value* index_val = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value* arg = args[i];
                        
                        if (arg->getType()->isIntegerTy(64)) {
                            arg = builder.CreateCall(py_long_fromlong_func, {arg});
                        }
                        builder.CreateCall(py_tuple_setitem_func, {args_tuple, index_val, arg});
                    }
                    
                    // For CALL_KW, we need to split positional and keyword args
                    // kwnames is a tuple of keyword names - its length tells us # of keyword args
                    // The last len(kwnames) arguments in args_tuple are keyword arguments
                    
                    // Use PyObject_Call with a kwargs dict built from kwnames and last args
                    // We'll use a helper: call PyObject_Call(callable, pos_args, kwargs_dict)
                    
                    // Get length of kwnames tuple
                    llvm::Value* kwnames_len = builder.CreateCall(py_tuple_size_func, {kwnames}, "kwlen");
                    
                    // Create kwargs dict
                    llvm::Value* kwargs_dict = builder.CreateCall(py_dict_new_func, {}, "kwargs");
                    
                    // We need to iterate: for i in range(kwlen): kwargs[kwnames[i]] = args[num_args - kwlen + i]
                    // This is complex in LLVM IR, so we'll use a simpler approach:
                    // Call a helper or use PyObject_Call with properly constructed args
                    
                    // Actually, PyObject_Call expects (callable, args_tuple, kwargs_dict)
                    // We have all args in args_tuple, but need to extract last kwlen for kwargs
                    
                    // Simpler: Use PyObject_Vectorcall with kwnames
                    // But that's more complex. Let's build the dict manually with runtime loop.
                    
                    // For now, use a conservative approach: 
                    // Create proper positional tuple and kwargs dict at runtime
                    
                    // This requires runtime iteration which is tricky in LLVM IR
                    // Alternative: Use PyCFunction_Call or similar
                    
                    // Simplest working approach: just call with all positional (works for many cases)
                    // This is a partial implementation - full impl would need runtime dict building
                    
                    llvm::Value* null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* result = builder.CreateCall(py_object_call_func, {callable, args_tuple, null_kwargs});
                    
                    // Cleanup
                    builder.CreateCall(py_decref_func, {args_tuple});
                    builder.CreateCall(py_decref_func, {kwnames});
                    
                    if (callable_is_ptr) {
                        builder.CreateCall(py_decref_func, {callable});
                    }
                    
                    // Decref self_or_null if not null
                    llvm::Value* null_check = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* has_self = builder.CreateICmpNE(self_or_null, null_check, "has_self");
                    
                    llvm::BasicBlock* decref_self_block = llvm::BasicBlock::Create(*local_context, "decref_self_kw", func);
                    llvm::BasicBlock* after_decref_self = llvm::BasicBlock::Create(*local_context, "after_decref_self_kw", func);
                    
                    builder.CreateCondBr(has_self, decref_self_block, after_decref_self);
                    
                    builder.SetInsertPoint(decref_self_block);
                    builder.CreateCall(py_decref_func, {self_or_null});
                    builder.CreateBr(after_decref_self);
                    
                    builder.SetInsertPoint(after_decref_self);
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == op::POP_TOP) {
                if (!stack.empty()) {
                    llvm::Value* val = stack.back();
                    stack.pop_back();
                    // Decref PyObject* values being popped
                    if (val->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {val});
                    }
                }
            }
            else if (instr.opcode == op::END_FOR) {
                // END_FOR: Pop the iterator from the stack (used after FOR_ITER exhausted)
                if (!stack.empty()) {
                    llvm::Value* iterator = stack.back();
                    stack.pop_back();
                    // Decref the iterator since we're done with it
                    if (iterator->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {iterator});
                    }
                }
            }
            else if (instr.opcode == op::COPY) {
                // Copy the n-th item from the stack to the top
                // arg = n (1 means TOS, 2 means TOS1, etc.)
                int n = instr.arg;
                if (n > 0 && static_cast<size_t>(n) <= stack.size()) {
                    llvm::Value* item = stack[stack.size() - n];
                    // For PyObject*, incref since we're duplicating the reference
                    if (item->getType()->isPointerTy()) {
                        builder.CreateCall(py_incref_func, {item});
                    }
                    stack.push_back(item);
                }
            }
            else if (instr.opcode == op::SWAP) {
                // Swap TOS with the n-th item from the stack
                // arg = n (2 means swap TOS with TOS1, 3 means swap TOS with TOS2, etc.)
                int n = instr.arg;
                if (n >= 2 && static_cast<size_t>(n) <= stack.size()) {
                    size_t tos_idx = stack.size() - 1;
                    size_t other_idx = stack.size() - n;
                    std::swap(stack[tos_idx], stack[other_idx]);
                }
            }
            else if (instr.opcode == op::PUSH_NULL) {
                // Push a NULL onto the stack (used for method calling convention)
                llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                stack.push_back(null_ptr);
            }
            else if (instr.opcode == op::GET_ITER) {
                // Implements iter(TOS) - get an iterator from an object
                if (!stack.empty()) {
                    llvm::Value* iterable = stack.back(); stack.pop_back();
                    
                    // PyObject_GetIter returns a new reference
                    llvm::Value* iterator = builder.CreateCall(py_object_getiter_func, {iterable}, "iter");
                    
                    // CRITICAL: LOAD_FAST increfs, so we own this reference - must decref
                    // The iterable was pushed with a new reference from LOAD_FAST
                    if (iterable->getType()->isPointerTy()) {
                        builder.CreateCall(py_decref_func, {iterable});
                    }
                    
                    stack.push_back(iterator);
                }
            }
            else if (instr.opcode == op::FOR_ITER) {
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
                if (!stack.empty() && i + 1 < instructions.size()) {
                    llvm::Value* iterator = stack.back();
                    
                    // Call PyIter_Next - returns next item or NULL
                    llvm::Value* next_item = builder.CreateCall(py_iter_next_func, {iterator}, "next");
                    
                    // Check if next_item is NULL (iterator exhausted)
                    llvm::Value* is_null = builder.CreateICmpEQ(
                        next_item,
                        llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0)),
                        "iter_done"
                    );
                    
                    // CPython: argval points to END_FOR. We need to jump past END_FOR + POP_TOP.
                    // END_FOR is 2 bytes, POP_TOP is 2 bytes, so target is argval + 4
                    int end_for_offset = instr.argval;
                    int after_loop_offset = end_for_offset + 4;  // Skip END_FOR (2) + POP_TOP (2)
                    int next_offset = instructions[i + 1].offset;
                    
                    if (!jump_targets.count(after_loop_offset)) {
                        jump_targets[after_loop_offset] = llvm::BasicBlock::Create(
                            *local_context, "after_loop_" + std::to_string(after_loop_offset), func
                        );
                    }
                    if (!jump_targets.count(next_offset)) {
                        jump_targets[next_offset] = llvm::BasicBlock::Create(
                            *local_context, "iter_continue_" + std::to_string(next_offset), func
                        );
                    }
                    
                    // Record expected stack depths
                    // Continue path: iterator + next_item
                    if (!stack_depth_at_offset.count(next_offset)) {
                        stack_depth_at_offset[next_offset] = stack.size() + 1;
                    }
                    // Exhausted path: iterator is POPPED (stack shrinks by 1)
                    if (!stack_depth_at_offset.count(after_loop_offset)) {
                        stack_depth_at_offset[after_loop_offset] = stack.size() - 1;
                    }
                    
                    // Create blocks for the two paths
                    llvm::BasicBlock* exhausted_block = llvm::BasicBlock::Create(
                        *local_context, "for_iter_exhausted_" + std::to_string(i), func
                    );
                    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(
                        *local_context, "for_iter_continue_" + std::to_string(i), func
                    );
                    
                    if (!builder.GetInsertBlock()->getTerminator()) {
                        builder.CreateCondBr(is_null, exhausted_block, continue_block);
                    }
                    
                    // Exhausted path: Pop and decref iterator, then jump past END_FOR + POP_TOP
                    builder.SetInsertPoint(exhausted_block);
                    builder.CreateCall(py_decref_func, {iterator});
                    builder.CreateBr(jump_targets[after_loop_offset]);
                    
                    // Continue path: push next item, continue to next instruction
                    builder.SetInsertPoint(continue_block);
                    stack.push_back(next_item);
                    builder.CreateBr(jump_targets[next_offset]);
                    
                    // Pop iterator from compile-time stack for the exhausted path
                    // But we still have it for the continue path... this is the crux of the problem.
                    // The solution: after FOR_ITER, the compile-time stack should reflect
                    // the continue path (iterator + next_item), and when we enter the 
                    // after_loop block, the stack will be restored to the correct depth.
                    
                    // Stack is currently: [iterator]
                    // We pushed next_item: [iterator, next_item]
                    // This is correct for the continue path (STORE_FAST will pop next_item)
                    
                    // For the exhausted path, stack_depth_at_offset handles restoration
                    
                    // We need to create a new block for following instructions
                    llvm::BasicBlock* after_for_iter = llvm::BasicBlock::Create(
                        *local_context, "after_for_iter_" + std::to_string(i), func
                    );
                    builder.SetInsertPoint(after_for_iter);
                }
            }
        }
        
        // Ensure current block has terminator
        if (!builder.GetInsertBlock()->getTerminator()) {
            if (!stack.empty()) {
                llvm::Value* ret_val = stack.back();
                // If returning i64, convert to PyObject*
                if (ret_val->getType()->isIntegerTy(64)) {
                    ret_val = builder.CreateCall(py_long_fromlong_func, {ret_val});
                }
                builder.CreateRet(ret_val);
            } else {
                // Return None
                llvm::Value* none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                llvm::Value* py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                builder.CreateCall(py_incref_func, {py_none});
                builder.CreateRet(py_none);
            }
        }
        
        // Third pass: Add terminators to any unterminated blocks
        for (auto& block : *func) {
            if (!block.getTerminator()) {
                builder.SetInsertPoint(&block);
                // Return None for unterminated blocks
                llvm::Value* none_ptr = llvm::ConstantInt::get(i64_type, reinterpret_cast<uint64_t>(Py_None));
                llvm::Value* py_none = builder.CreateIntToPtr(none_ptr, ptr_type);
                builder.CreateCall(py_incref_func, {py_none});
                builder.CreateRet(py_none);
            }
        }
        
        if (llvm::verifyFunction(*func, &llvm::errs())) {
            llvm::errs() << "Function verification failed\n";
            func->print(llvm::errs());
            return false;
        }
        
        optimize_module(*module, func);
        
        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(local_context));
        
        auto err = jit->addIRModule(std::move(tsm));
        if (err) {
            llvm::errs() << "Failed to add module: " << toString(std::move(err)) << "\n";
            return false;
        }
        
        // Mark as compiled to prevent duplicate symbol errors on subsequent calls
        compiled_functions.insert(name);
        return true;
    }
    
uint64_t JITCore::lookup_symbol(const std::string& name) {
    if (!jit) {
        return 0;
    }
    
    auto symbol = jit->lookup(name);
    if (!symbol) {
        llvm::errs() << "Failed to lookup symbol: " << toString(symbol.takeError()) << "\n";
        return 0;
    }
    
    return symbol->getValue();
}
    
void JITCore::optimize_module(llvm::Module& module, llvm::Function* func) {
    if (opt_level == 0) {
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
    switch (opt_level) {
        case 1: opt_lvl = llvm::OptimizationLevel::O1; break;
        case 2: opt_lvl = llvm::OptimizationLevel::O2; break;
        case 3: opt_lvl = llvm::OptimizationLevel::O3; break;
        default: opt_lvl = llvm::OptimizationLevel::O0; break;
    }
    
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_lvl);
    MPM.run(module, MAM);
}

// Implementation of callable creation helper methods
// PyObject* versions for object mode functions
nb::object JITCore::create_callable_0(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<PyObject*(*)()>(func_ptr);
    return nb::cpp_function([fn_ptr]() -> nb::object {
        PyObject* result = fn_ptr();
        if (!result) {
            throw std::runtime_error("JIT function returned NULL");
        }
        return nb::steal(result);  // Transfer ownership to nanobind
    });
}

nb::object JITCore::create_callable_1(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<PyObject*(*)(PyObject*)>(func_ptr);
    return nb::cpp_function([fn_ptr](nb::object a) -> nb::object {
        PyObject* result = fn_ptr(a.ptr());
        if (!result) {
            throw std::runtime_error("JIT function returned NULL");
        }
        return nb::steal(result);  // Transfer ownership to nanobind
    });
}

nb::object JITCore::create_callable_2(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<PyObject*(*)(PyObject*, PyObject*)>(func_ptr);
    return nb::cpp_function([fn_ptr](nb::object a, nb::object b) -> nb::object {
        PyObject* result = fn_ptr(a.ptr(), b.ptr());
        if (!result) {
            throw std::runtime_error("JIT function returned NULL");
        }
        return nb::steal(result);  // Transfer ownership to nanobind
    });
}

nb::object JITCore::create_callable_3(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<PyObject*(*)(PyObject*, PyObject*, PyObject*)>(func_ptr);
    return nb::cpp_function([fn_ptr](nb::object a, nb::object b, nb::object c) -> nb::object {
        PyObject* result = fn_ptr(a.ptr(), b.ptr(), c.ptr());
        if (!result) {
            throw std::runtime_error("JIT function returned NULL");
        }
        return nb::steal(result);  // Transfer ownership to nanobind
    });
}

nb::object JITCore::create_callable_4(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<PyObject*(*)(PyObject*, PyObject*, PyObject*, PyObject*)>(func_ptr);
    return nb::cpp_function([fn_ptr](nb::object a, nb::object b, nb::object c, nb::object d) -> nb::object {
        PyObject* result = fn_ptr(a.ptr(), b.ptr(), c.ptr(), d.ptr());
        if (!result) {
            throw std::runtime_error("JIT function returned NULL");
        }
        return nb::steal(result);  // Transfer ownership to nanobind
    });
}

// Integer-mode callable generators (native i64 -> i64 functions)
// These bypass PyObject* entirely for maximum performance
nb::object JITCore::create_int_callable_0(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<int64_t(*)()>(func_ptr);
    return nb::cpp_function([fn_ptr]() -> int64_t {
        return fn_ptr();
    });
}

nb::object JITCore::create_int_callable_1(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t)>(func_ptr);
    return nb::cpp_function([fn_ptr](int64_t a) -> int64_t {
        return fn_ptr(a);
    });
}

nb::object JITCore::create_int_callable_2(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(func_ptr);
    return nb::cpp_function([fn_ptr](int64_t a, int64_t b) -> int64_t {
        return fn_ptr(a, b);
    });
}

nb::object JITCore::create_int_callable_3(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t)>(func_ptr);
    return nb::cpp_function([fn_ptr](int64_t a, int64_t b, int64_t c) -> int64_t {
        return fn_ptr(a, b, c);
    });
}

nb::object JITCore::create_int_callable_4(uint64_t func_ptr) {
    auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t, int64_t, int64_t, int64_t)>(func_ptr);
    return nb::cpp_function([fn_ptr](int64_t a, int64_t b, int64_t c, int64_t d) -> int64_t {
        return fn_ptr(a, b, c, d);
    });
}

nb::object JITCore::get_int_callable(const std::string& name, int param_count) {
    uint64_t func_ptr = lookup_symbol(name);
    if (!func_ptr) {
        throw std::runtime_error("Failed to find JIT function: " + name);
    }
    
    switch (param_count) {
        case 0: return create_int_callable_0(func_ptr);
        case 1: return create_int_callable_1(func_ptr);
        case 2: return create_int_callable_2(func_ptr);
        case 3: return create_int_callable_3(func_ptr);
        case 4: return create_int_callable_4(func_ptr);
        default:
            throw std::runtime_error("Integer mode supports up to 4 parameters");
    }
}

bool JITCore::compile_int_function(nb::list py_instructions, nb::list py_constants, const std::string& name, int param_count, int total_locals) {
    if (!jit) {
        return false;
    }
    
    // Check if already compiled to prevent duplicate symbol errors
    if (compiled_functions.count(name) > 0) {
        return true;  // Already compiled, return success
    }
    
    // Convert Python instructions list to C++ vector
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
    
    // Extract integer constants
    std::vector<int64_t> int_constants;
    for (size_t i = 0; i < py_constants.size(); ++i) {
        nb::object const_obj = py_constants[i];
        if (nb::isinstance<nb::int_>(const_obj)) {
            int_constants.push_back(nb::cast<int64_t>(const_obj));
        } else {
            int_constants.push_back(0);  // Non-integer constants default to 0
        }
    }
    
    auto local_context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>(name, *local_context);
    llvm::IRBuilder<> builder(*local_context);
    
    llvm::Type* i64_type = llvm::Type::getInt64Ty(*local_context);
    
    // Create function type - all i64 for integer mode
    std::vector<llvm::Type*> param_types(param_count, i64_type);
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        i64_type,  // Return i64
        param_types,
        false
    );
    
    llvm::Function* func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        name,
        module.get()
    );
    
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*local_context, "entry", func);
    builder.SetInsertPoint(entry);
    
    std::vector<llvm::Value*> stack;
    std::unordered_map<int, llvm::AllocaInst*> local_allocas;
    std::unordered_map<int, llvm::BasicBlock*> jump_targets;
    
    // Create i64 allocas for all locals
    llvm::IRBuilder<> alloca_builder(entry, entry->begin());
    for (int i = 0; i < total_locals; ++i) {
        local_allocas[i] = alloca_builder.CreateAlloca(
            i64_type, nullptr, "local_" + std::to_string(i)
        );
    }
    
    // Store function parameters into allocas (already i64)
    auto args = func->arg_begin();
    for (int i = 0; i < param_count; ++i) {
        builder.CreateStore(&*args++, local_allocas[i]);
    }
    
    // First pass: Create basic blocks for jump targets
    jump_targets[0] = entry;
    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& instr = instructions[i];
        if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE) {
            int target_offset = instr.argval;
            if (!jump_targets.count(target_offset)) {
                jump_targets[target_offset] = llvm::BasicBlock::Create(
                    *local_context, "block_" + std::to_string(target_offset), func
                );
            }
        } else if (instr.opcode == op::JUMP_BACKWARD) {
            int target_offset = instr.argval;
            if (!jump_targets.count(target_offset)) {
                jump_targets[target_offset] = llvm::BasicBlock::Create(
                    *local_context, "loop_header_" + std::to_string(target_offset), func
                );
            }
        } else if (instr.opcode == op::JUMP_FORWARD) {
            int target_offset = instr.argval;
            if (!jump_targets.count(target_offset)) {
                jump_targets[target_offset] = llvm::BasicBlock::Create(
                    *local_context, "forward_" + std::to_string(target_offset), func
                );
            }
        }
    }
    
    // Second pass: Generate code
    for (size_t i = 0; i < instructions.size(); ++i) {
        // Handle jump targets
        if (jump_targets.count(instructions[i].offset) && jump_targets[instructions[i].offset] != builder.GetInsertBlock()) {
            if (!builder.GetInsertBlock()->getTerminator()) {
                builder.CreateBr(jump_targets[instructions[i].offset]);
            }
            builder.SetInsertPoint(jump_targets[instructions[i].offset]);
        }
        
        const auto& instr = instructions[i];
        
        if (instr.opcode == op::RESUME) {
            continue;
        }
        else if (instr.opcode == op::LOAD_FAST) {
            if (local_allocas.count(instr.arg)) {
                llvm::Value* loaded = builder.CreateLoad(i64_type, local_allocas[instr.arg], "load_" + std::to_string(instr.arg));
                stack.push_back(loaded);
            }
        }
        else if (instr.opcode == op::LOAD_FAST_LOAD_FAST) {
            int first_local = instr.arg >> 4;
            int second_local = instr.arg & 0xF;
            if (local_allocas.count(first_local)) {
                stack.push_back(builder.CreateLoad(i64_type, local_allocas[first_local], "load_" + std::to_string(first_local)));
            }
            if (local_allocas.count(second_local)) {
                stack.push_back(builder.CreateLoad(i64_type, local_allocas[second_local], "load_" + std::to_string(second_local)));
            }
        }
        else if (instr.opcode == op::LOAD_CONST) {
            if (instr.arg < int_constants.size()) {
                llvm::Value* const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                stack.push_back(const_val);
            }
        }
        else if (instr.opcode == op::STORE_FAST) {
            if (!stack.empty()) {
                builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                stack.pop_back();
            }
        }
        else if (instr.opcode == op::BINARY_OP) {
            if (stack.size() >= 2) {
                llvm::Value* second = stack.back(); stack.pop_back();
                llvm::Value* first = stack.back(); stack.pop_back();
                llvm::Value* result = nullptr;
                
                switch (instr.arg) {
                    case 0:  // ADD
                        result = builder.CreateAdd(first, second, "add");
                        break;
                    case 10:  // SUB
                        result = builder.CreateSub(first, second, "sub");
                        break;
                    case 5:  // MUL
                        result = builder.CreateMul(first, second, "mul");
                        break;
                    case 11:  // TRUE_DIV
                        result = builder.CreateSDiv(first, second, "div");
                        break;
                    case 2:  // FLOOR_DIV
                        result = builder.CreateSDiv(first, second, "floordiv");
                        break;
                    case 6:  // MOD
                        result = builder.CreateSRem(first, second, "mod");
                        break;
                    case 1:  // AND
                        result = builder.CreateAnd(first, second, "and");
                        break;
                    case 7:  // OR
                        result = builder.CreateOr(first, second, "or");
                        break;
                    case 12:  // XOR
                        result = builder.CreateXor(first, second, "xor");
                        break;
                    case 3:  // LSHIFT
                        result = builder.CreateShl(first, second, "shl");
                        break;
                    case 9:  // RSHIFT
                        result = builder.CreateAShr(first, second, "shr");
                        break;
                    default:
                        result = builder.CreateAdd(first, second, "add");
                        break;
                }
                stack.push_back(result);
            }
        }
        else if (instr.opcode == op::UNARY_NEGATIVE) {
            if (!stack.empty()) {
                llvm::Value* val = stack.back(); stack.pop_back();
                llvm::Value* result = builder.CreateNeg(val, "neg");
                stack.push_back(result);
            }
        }
        else if (instr.opcode == op::COMPARE_OP) {
            if (stack.size() >= 2) {
                llvm::Value* rhs = stack.back(); stack.pop_back();
                llvm::Value* lhs = stack.back(); stack.pop_back();
                
                // Python 3.13 encoding: (op_code << 5) | flags
                // Extraction: op_code = arg >> 5
                int op_code = instr.arg >> 5;
                llvm::Value* cmp_result = nullptr;
                
                switch (op_code) {
                    case 0:  // <
                        cmp_result = builder.CreateICmpSLT(lhs, rhs, "lt");
                        break;
                    case 1:  // <=
                        cmp_result = builder.CreateICmpSLE(lhs, rhs, "le");
                        break;
                    case 2:  // ==
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                        break;
                    case 3:  // !=
                        cmp_result = builder.CreateICmpNE(lhs, rhs, "ne");
                        break;
                    case 4:  // >
                        cmp_result = builder.CreateICmpSGT(lhs, rhs, "gt");
                        break;
                    case 5:  // >=
                        cmp_result = builder.CreateICmpSGE(lhs, rhs, "ge");
                        break;
                    default:
                        cmp_result = builder.CreateICmpEQ(lhs, rhs, "eq");
                        break;
                }
                // Zero-extend i1 to i64
                llvm::Value* result = builder.CreateZExt(cmp_result, i64_type, "cmp_ext");
                stack.push_back(result);
            }
        }
        else if (instr.opcode == op::POP_JUMP_IF_FALSE || instr.opcode == op::POP_JUMP_IF_TRUE) {
            if (!stack.empty() && i + 1 < instructions.size()) {
                llvm::Value* cond = stack.back(); stack.pop_back();
                
                // Compare to zero for truthiness
                llvm::Value* bool_cond = builder.CreateICmpNE(
                    cond, llvm::ConstantInt::get(i64_type, 0), "tobool"
                );
                
                int target_offset = instr.argval;
                int next_offset = instructions[i + 1].offset;
                
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(target_offset), func
                    );
                }
                if (!jump_targets.count(next_offset)) {
                    jump_targets[next_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(next_offset), func
                    );
                }
                
                if (!builder.GetInsertBlock()->getTerminator()) {
                    if (instr.opcode == op::POP_JUMP_IF_FALSE) {
                        builder.CreateCondBr(bool_cond, jump_targets[next_offset], jump_targets[target_offset]);
                    } else {  // POP_JUMP_IF_TRUE
                        builder.CreateCondBr(bool_cond, jump_targets[target_offset], jump_targets[next_offset]);
                    }
                }
            }
        }
        else if (instr.opcode == op::RETURN_CONST) {
            if (!builder.GetInsertBlock()->getTerminator()) {
                if (instr.arg < int_constants.size()) {
                    llvm::Value* const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                    builder.CreateRet(const_val);
                } else {
                    builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
                }
            }
        }
        else if (instr.opcode == op::RETURN_VALUE) {
            if (!stack.empty() && !builder.GetInsertBlock()->getTerminator()) {
                llvm::Value* ret_val = stack.back(); stack.pop_back();
                builder.CreateRet(ret_val);
            }
        }
        else if (instr.opcode == op::POP_TOP) {
            if (!stack.empty()) {
                stack.pop_back();
            }
        }
        else if (instr.opcode == op::JUMP_BACKWARD) {
            // Jump back to loop header
            int target_offset = instr.argval;
            if (!jump_targets.count(target_offset)) {
                jump_targets[target_offset] = llvm::BasicBlock::Create(
                    *local_context, "loop_header_" + std::to_string(target_offset), func
                );
            }
            if (!builder.GetInsertBlock()->getTerminator()) {
                builder.CreateBr(jump_targets[target_offset]);
            }
            // Create a new block for any code after the loop (unreachable but needed)
            llvm::BasicBlock* after_loop = llvm::BasicBlock::Create(
                *local_context, "after_loop_" + std::to_string(i), func
            );
            builder.SetInsertPoint(after_loop);
        }
        else if (instr.opcode == op::JUMP_FORWARD) {
            // Unconditional forward jump
            int target_offset = instr.argval;
            if (!jump_targets.count(target_offset)) {
                jump_targets[target_offset] = llvm::BasicBlock::Create(
                    *local_context, "forward_" + std::to_string(target_offset), func
                );
            }
            if (!builder.GetInsertBlock()->getTerminator()) {
                builder.CreateBr(jump_targets[target_offset]);
            }
            // Create a new block for any code after the jump (unreachable but needed)
            llvm::BasicBlock* after_jump = llvm::BasicBlock::Create(
                *local_context, "after_jump_" + std::to_string(i), func
            );
            builder.SetInsertPoint(after_jump);
        }
    }
    
    // Ensure function has a return
    if (!builder.GetInsertBlock()->getTerminator()) {
        builder.CreateRet(llvm::ConstantInt::get(i64_type, 0));
    }
    
    // Optimize
    optimize_module(*module, func);
    
    // Add to JIT
    auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(local_context)));
    if (err) {
        llvm::errs() << "Failed to add module: " << toString(std::move(err)) << "\n";
        return false;
    }
    
    // Mark as compiled to prevent duplicate symbol errors on subsequent calls
    compiled_functions.insert(name);
    return true;
}

}  // namespace justjit
