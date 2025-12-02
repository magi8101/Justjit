#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>
#include <string>
#include <unordered_set>

namespace nb = nanobind;

namespace justjit {

struct Instruction {
    uint16_t opcode;
    uint16_t arg;
    int32_t argval;  // Actual target offset for jump instructions (can be negative for constants)
    uint16_t offset;
};

class JITCore {
public:
    JITCore();
    ~JITCore();  // Clean up stored Python references
    
    void set_opt_level(int level);
    int get_opt_level() const;
    nb::object get_callable(const std::string& name, int param_count);
    nb::object get_int_callable(const std::string& name, int param_count);  // For integer-mode functions
    bool compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::object py_globals_dict, nb::object py_builtins_dict, nb::list py_closure_cells, const std::string& name, int param_count = 2, int total_locals = 3, int nlocals = 3);
    bool compile_int_function(nb::list py_instructions, nb::list py_constants, const std::string& name, int param_count = 2, int total_locals = 3);  // Integer-only mode
    uint64_t lookup_symbol(const std::string& name);
    
    // Helper to declare Python C API functions in LLVM module
    void declare_python_api_functions(llvm::Module* module, llvm::IRBuilder<>* builder);
    
    // =========================================================================
    // Python C API function declarations (public for modular opcode handlers)
    // =========================================================================
    llvm::Function* py_list_new_func = nullptr;
    llvm::Function* py_list_setitem_func = nullptr;
    llvm::Function* py_object_getitem_func = nullptr;
    llvm::Function* py_incref_func = nullptr;
    llvm::Function* py_decref_func = nullptr;
    llvm::Function* py_long_fromlong_func = nullptr;
    llvm::Function* py_tuple_new_func = nullptr;
    llvm::Function* py_tuple_setitem_func = nullptr;
    llvm::Function* py_number_add_func = nullptr;
    llvm::Function* py_number_subtract_func = nullptr;
    llvm::Function* py_number_multiply_func = nullptr;
    llvm::Function* py_number_truedivide_func = nullptr;
    llvm::Function* py_number_floordivide_func = nullptr;
    llvm::Function* py_number_remainder_func = nullptr;
    llvm::Function* py_number_power_func = nullptr;
    llvm::Function* py_number_negative_func = nullptr;
    llvm::Function* py_object_str_func = nullptr;
    llvm::Function* py_unicode_concat_func = nullptr;
    llvm::Function* py_object_getattr_func = nullptr;
    llvm::Function* py_object_setattr_func = nullptr;
    llvm::Function* py_object_setitem_func = nullptr;
    llvm::Function* py_object_call_func = nullptr;
    llvm::Function* py_long_aslong_func = nullptr;
    llvm::Function* py_object_richcompare_bool_func = nullptr;
    llvm::Function* py_object_istrue_func = nullptr;
    
    // Additional Python C API functions for more opcodes
    llvm::Function* py_number_invert_func = nullptr;
    llvm::Function* py_object_not_func = nullptr;
    llvm::Function* py_object_getiter_func = nullptr;
    llvm::Function* py_iter_next_func = nullptr;
    llvm::Function* py_dict_new_func = nullptr;
    llvm::Function* py_dict_setitem_func = nullptr;
    llvm::Function* py_set_new_func = nullptr;
    llvm::Function* py_set_add_func = nullptr;
    llvm::Function* py_list_append_func = nullptr;
    llvm::Function* py_list_extend_func = nullptr;
    llvm::Function* py_sequence_contains_func = nullptr;
    llvm::Function* py_number_lshift_func = nullptr;
    llvm::Function* py_number_rshift_func = nullptr;
    llvm::Function* py_number_and_func = nullptr;
    llvm::Function* py_number_or_func = nullptr;
    llvm::Function* py_number_xor_func = nullptr;
    llvm::Function* py_cell_get_func = nullptr;
    llvm::Function* py_tuple_getitem_func = nullptr;
    llvm::Function* py_tuple_size_func = nullptr;
    llvm::Function* py_slice_new_func = nullptr;
    llvm::Function* py_sequence_getslice_func = nullptr;
    llvm::Function* py_sequence_setslice_func = nullptr;
    llvm::Function* py_object_delitem_func = nullptr;
    llvm::Function* py_set_update_func = nullptr;
    llvm::Function* py_dict_update_func = nullptr;
    llvm::Function* py_dict_merge_func = nullptr;
    llvm::Function* py_dict_getitem_func = nullptr;  // For runtime global lookup (Bug #4 fix)
    
    // Exception handling API functions (Bug #3 fix)
    llvm::Function* py_err_occurred_func = nullptr;
    llvm::Function* py_err_fetch_func = nullptr;
    llvm::Function* py_err_restore_func = nullptr;
    llvm::Function* py_err_set_object_func = nullptr;
    llvm::Function* py_err_set_string_func = nullptr;
    llvm::Function* py_err_clear_func = nullptr;
    llvm::Function* py_exception_matches_func = nullptr;
    llvm::Function* py_object_type_func = nullptr;
    llvm::Function* py_exception_set_cause_func = nullptr;
    
private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<llvm::LLVMContext> context;
    int opt_level = 3;
    
    // Store references to Python objects we've incref'd (for cleanup)
    std::vector<PyObject*> stored_constants;
    std::vector<PyObject*> stored_names;
    
    // Runtime globals/builtins dicts for LOAD_GLOBAL (Bug #4 fix)
    PyObject* globals_dict_ptr = nullptr;
    PyObject* builtins_dict_ptr = nullptr;
    
    // Cache of already-compiled function names to prevent duplicate symbol errors
    std::unordered_set<std::string> compiled_functions;
    
    // Closure cells storage (for COPY_FREE_VARS / LOAD_DEREF)
    std::vector<PyObject*> stored_closure_cells;
    
    nb::object create_callable_0(uint64_t func_ptr);
    nb::object create_callable_1(uint64_t func_ptr);
    nb::object create_callable_2(uint64_t func_ptr);
    nb::object create_callable_3(uint64_t func_ptr);
    nb::object create_callable_4(uint64_t func_ptr);
    
    // Integer-mode callable generators (native i64 -> i64 functions)
    nb::object create_int_callable_0(uint64_t func_ptr);
    nb::object create_int_callable_1(uint64_t func_ptr);
    nb::object create_int_callable_2(uint64_t func_ptr);
    nb::object create_int_callable_3(uint64_t func_ptr);
    nb::object create_int_callable_4(uint64_t func_ptr);
    
    void optimize_module(llvm::Module& module, llvm::Function* func);
};

}
