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

namespace nb = nanobind;

namespace justjit {

struct Instruction {
    uint16_t opcode;
    uint16_t arg;
    uint16_t argval;  // Actual target offset for jump instructions
    uint16_t offset;
};

class JITCore {
public:
    JITCore();
    ~JITCore() = default;
    
    void set_opt_level(int level);
    int get_opt_level() const;
    nb::object get_callable(const std::string& name, int param_count);
    bool compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::list py_globals, const std::string& name, int param_count = 2, int total_locals = 3);
    uint64_t lookup_symbol(const std::string& name);
    
    // Helper to declare Python C API functions in LLVM module
    void declare_python_api_functions(llvm::Module* module, llvm::IRBuilder<>* builder);
    
private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<llvm::LLVMContext> context;
    int opt_level = 3;
    
    // Python C API function declarations (cached per module)
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
    llvm::Function* py_object_str_func = nullptr;
    llvm::Function* py_unicode_concat_func = nullptr;
    llvm::Function* py_object_getattr_func = nullptr;
    llvm::Function* py_object_setattr_func = nullptr;
    llvm::Function* py_object_setitem_func = nullptr;
    llvm::Function* py_object_call_func = nullptr;
    llvm::Function* py_long_aslong_func = nullptr;
    
    nb::object create_callable_0(uint64_t func_ptr);
    nb::object create_callable_1(uint64_t func_ptr);
    nb::object create_callable_2(uint64_t func_ptr);
    nb::object create_callable_3(uint64_t func_ptr);
    nb::object create_callable_4(uint64_t func_ptr);
    
    void optimize_module(llvm::Module& module, llvm::Function* func);
};

}
