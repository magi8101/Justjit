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
#include <vector>
#include <unordered_set>

namespace nb = nanobind;

namespace justjit
{
    // =========================================================================
    // JIT Generator State Machine
    // =========================================================================
    // Generator implementation using a state machine approach.
    // Each generator function is compiled into a "step" function that:
    //   - Takes (state*, locals_array*, sent_value) as input
    //   - Returns the yielded value (or NULL if done/error)
    //   - Updates state to indicate next resume point
    //
    // State values:
    //   0 = initial (not started)
    //   1..N = resume points after yield
    //   -1 = completed (returned)
    //   -2 = error
    // =========================================================================

    // Forward declaration of the JIT generator object
    struct JITGeneratorObject;

    // Type definition for generator step function
    // Signature: PyObject* step_func(int32_t* state, PyObject** locals, PyObject* sent_value)
    typedef PyObject* (*GeneratorStepFunc)(int32_t* state, PyObject** locals, PyObject* sent_value);

    // JIT Generator object - a Python object that wraps a compiled generator
    struct JITGeneratorObject {
        PyObject_HEAD
        int32_t state;              // Current state (0=initial, >0=suspended at yield N, -1=done)
        PyObject** locals;          // Array of local variables (preserved across yields)
        Py_ssize_t num_locals;      // Number of local variable slots
        GeneratorStepFunc step_func; // Pointer to the compiled step function
        PyObject* name;             // Generator name (for repr)
        PyObject* qualname;         // Qualified name
    };

    // Python type object for JIT generators (defined in jit_core.cpp)
    extern PyTypeObject JITGenerator_Type;

    // Helper functions for JIT generator
    PyObject* JITGenerator_New(GeneratorStepFunc step_func, Py_ssize_t num_locals,
                               PyObject* name, PyObject* qualname);
    PyObject* JITGenerator_Send(JITGeneratorObject* gen, PyObject* value);

    // =========================================================================
    // JIT Coroutine Object
    // =========================================================================
    // Coroutines are similar to generators but implement the coroutine protocol:
    // - __await__() returns self (the awaitable iterator)
    // - send(), throw(), close() work the same as generators
    // - SEND opcode handles yield-from/await delegation
    // =========================================================================

    // Forward declaration of the JIT coroutine object
    struct JITCoroutineObject;

    // JIT Coroutine object - wraps a compiled async function
    struct JITCoroutineObject {
        PyObject_HEAD
        int32_t state;              // Current state (0=initial, >0=suspended at await, -1=done)
        PyObject** locals;          // Array of local variables (preserved across awaits)
        Py_ssize_t num_locals;      // Number of local variable slots
        GeneratorStepFunc step_func; // Pointer to the compiled step function (same signature)
        PyObject* name;             // Coroutine name (for repr)
        PyObject* qualname;         // Qualified name
        PyObject* awaiting;         // Currently awaited object (for SEND delegation)
    };

    // Python type object for JIT coroutines (defined in jit_core.cpp)
    extern PyTypeObject JITCoroutine_Type;

    // Helper functions for JIT coroutine
    PyObject* JITCoroutine_New(GeneratorStepFunc step_func, Py_ssize_t num_locals,
                               PyObject* name, PyObject* qualname);
    PyObject* JITCoroutine_Send(JITCoroutineObject* coro, PyObject* value);

    struct Instruction
    {
        uint16_t opcode;
        uint16_t arg;
        int32_t argval; // Actual target offset for jump instructions (can be negative for constants)
        uint16_t offset;
    };

    // Exception table entry for try/except handling
    struct ExceptionTableEntry
    {
        int32_t start;  // Start offset of protected range
        int32_t end;    // End offset of protected range
        int32_t target; // Handler offset (PUSH_EXC_INFO location)
        int32_t depth;  // Stack depth to unwind to
        bool lasti;     // Whether to push last instruction offset
    };

    // =========================================================================
    // Control Flow Graph (CFG) Data Structures for PHI Node Support
    // =========================================================================
    // These structures enable proper SSA form generation for complex control flow
    // patterns like pattern matching (match/case), loops, and exception handling.
    // =========================================================================

    // Represents a basic block in the CFG
    struct BasicBlockInfo
    {
        int start_offset;                      // Bytecode offset where block starts
        int end_offset;                        // Bytecode offset where block ends (exclusive)
        std::vector<int> predecessors;         // Offsets of predecessor blocks
        std::vector<int> successors;           // Offsets of successor blocks
        int stack_depth_at_entry;              // Expected stack depth when entering this block
        bool is_exception_handler;             // True if this is an exception handler block
        bool needs_phi_nodes;                  // True if multiple predecessors with different stacks
        llvm::BasicBlock* llvm_block;          // The LLVM BasicBlock for this CFG block
    };

    // Stack state at a specific point in control flow
    struct CFGStackState
    {
        std::vector<llvm::Value*> stack;       // Values on the stack
        llvm::BasicBlock* from_block;          // Which LLVM block this state came from
        int from_offset;                       // Bytecode offset this state is from
    };

    class JITCore
    {
    public:
        JITCore();
        ~JITCore(); // Clean up stored Python references

        void set_opt_level(int level);
        int get_opt_level() const;
        void set_dump_ir(bool dump);
        bool get_dump_ir() const;
        std::string get_last_ir() const;
        nb::object get_callable(const std::string &name, int param_count);
        nb::object get_int_callable(const std::string &name, int param_count); // For integer-mode functions
        bool compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::object py_globals_dict, nb::object py_builtins_dict, nb::list py_closure_cells, nb::list py_exception_table, const std::string &name, int param_count = 2, int total_locals = 3, int nlocals = 3);
        bool compile_int_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Integer-only mode
        bool compile_float_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Float-only mode
        nb::object get_float_callable(const std::string &name, int param_count); // For float-mode functions
        bool compile_bool_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Bool-only mode
        nb::object get_bool_callable(const std::string &name, int param_count); // For bool-mode functions
        bool compile_int32_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Int32 mode (C interop)
        nb::object get_int32_callable(const std::string &name, int param_count); // For int32-mode functions
        bool compile_float32_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Float32 mode (SIMD/ML)
        nb::object get_float32_callable(const std::string &name, int param_count); // For float32-mode functions
        bool compile_complex128_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Complex128 mode (scientific)
        nb::object get_complex128_callable(const std::string &name, int param_count); // For complex128-mode functions
        bool compile_ptr_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Ptr mode (array access)
        nb::object get_ptr_callable(const std::string &name, int param_count); // For ptr-mode functions
        bool compile_vec4f_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Vec4f mode (SSE SIMD)
        nb::object get_vec4f_callable(const std::string &name, int param_count); // For vec4f-mode functions
        bool compile_vec8i_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Vec8i mode (AVX SIMD)
        nb::object get_vec8i_callable(const std::string &name, int param_count); // For vec8i-mode functions
        bool compile_complex64_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Complex64 mode (single-precision)
        nb::object get_complex64_callable(const std::string &name, int param_count); // For complex64-mode functions
        bool compile_optional_f64_function(nb::list py_instructions, nb::list py_constants, const std::string &name, int param_count = 2, int total_locals = 3); // Optional<f64> mode (nullable)
        nb::object get_optional_f64_callable(const std::string &name, int param_count); // For optional_f64-mode functions
        
        // Generator compilation - transforms generator function to state machine step function
        bool compile_generator(nb::list py_instructions, nb::list py_constants, nb::list py_names, 
                              nb::object py_globals_dict, nb::object py_builtins_dict, 
                              nb::list py_closure_cells, nb::list py_exception_table,
                              const std::string &name, int param_count, int total_locals, int nlocals);
        
        // Get a generator factory callable (returns a new generator on each call)
        nb::object get_generator_callable(const std::string &name, int param_count, int total_locals,
                                          nb::object func_name, nb::object func_qualname);
        
        uint64_t lookup_symbol(const std::string &name);

        // Helper to declare Python C API functions in LLVM module
        void declare_python_api_functions(llvm::Module *module, llvm::IRBuilder<> *builder);

        // =========================================================================
        // Python C API function declarations (public for modular opcode handlers)
        // =========================================================================
        llvm::Function *py_list_new_func = nullptr;
        llvm::Function *py_list_setitem_func = nullptr;
        llvm::Function *py_object_getitem_func = nullptr;
        llvm::Function *py_incref_func = nullptr;
        llvm::Function *py_xincref_func = nullptr; // NULL-safe Py_XINCREF
        llvm::Function *py_decref_func = nullptr;
        llvm::Function *py_xdecref_func = nullptr; // NULL-safe Py_XDECREF
        llvm::Function *py_long_fromlong_func = nullptr;
        llvm::Function *py_long_fromlonglong_func = nullptr; // For 64-bit int conversion (Windows fix)
        llvm::Function *py_tuple_new_func = nullptr;
        llvm::Function *py_tuple_setitem_func = nullptr;
        llvm::Function *py_number_add_func = nullptr;
        llvm::Function *py_number_subtract_func = nullptr;
        llvm::Function *py_number_multiply_func = nullptr;
        llvm::Function *py_number_matrixmultiply_func = nullptr;
        llvm::Function *py_number_truedivide_func = nullptr;
        llvm::Function *py_number_floordivide_func = nullptr;
        llvm::Function *py_number_remainder_func = nullptr;
        llvm::Function *py_number_power_func = nullptr;
        llvm::Function *py_number_negative_func = nullptr;
        llvm::Function *py_number_positive_func = nullptr; // For unary + operator (INTRINSIC_UNARY_POSITIVE)
        llvm::Function *py_object_str_func = nullptr;
        llvm::Function *py_unicode_concat_func = nullptr;
        llvm::Function *py_object_getattr_func = nullptr;
        llvm::Function *py_object_setattr_func = nullptr;
        llvm::Function *py_object_setitem_func = nullptr;
        llvm::Function *py_object_call_func = nullptr;
        llvm::Function *py_long_aslong_func = nullptr;
        llvm::Function *py_object_richcompare_bool_func = nullptr;
        llvm::Function *py_object_istrue_func = nullptr;
        llvm::Function *py_object_isinstance_func = nullptr;

        // Additional Python C API functions for more opcodes
        llvm::Function *py_number_invert_func = nullptr;
        llvm::Function *py_object_not_func = nullptr;
        llvm::Function *py_object_getiter_func = nullptr;
        llvm::Function *py_iter_next_func = nullptr;
        llvm::Function *py_dict_new_func = nullptr;
        llvm::Function *py_dict_setitem_func = nullptr;
        llvm::Function *py_set_new_func = nullptr;
        llvm::Function *py_set_add_func = nullptr;
        llvm::Function *py_list_append_func = nullptr;
        llvm::Function *py_list_extend_func = nullptr;
        llvm::Function *py_sequence_contains_func = nullptr;
        llvm::Function *py_number_lshift_func = nullptr;
        llvm::Function *py_number_rshift_func = nullptr;
        llvm::Function *py_number_and_func = nullptr;
        llvm::Function *py_number_or_func = nullptr;
        llvm::Function *py_number_xor_func = nullptr;
        llvm::Function *py_cell_get_func = nullptr;
        llvm::Function *py_tuple_getitem_func = nullptr;
        llvm::Function *py_tuple_size_func = nullptr;
        llvm::Function *py_slice_new_func = nullptr;
        llvm::Function *py_sequence_getslice_func = nullptr;
        llvm::Function *py_sequence_setslice_func = nullptr;
        llvm::Function *py_sequence_size_func = nullptr;    // Py_ssize_t PySequence_Size(PyObject*)
        llvm::Function *py_sequence_tuple_func = nullptr;   // PyObject* PySequence_Tuple(PyObject*)
        llvm::Function *py_sequence_getitem_func = nullptr; // PyObject* PySequence_GetItem(PyObject*, Py_ssize_t)
        llvm::Function *py_object_delitem_func = nullptr;
        llvm::Function *py_set_update_func = nullptr;
        llvm::Function *py_dict_update_func = nullptr;
        llvm::Function *py_dict_merge_func = nullptr;
        llvm::Function *py_dict_getitem_func = nullptr; // For runtime global lookup (Bug #4 fix)

        // Exception handling API functions (Bug #3 fix)
        llvm::Function *py_err_occurred_func = nullptr;
        llvm::Function *py_err_fetch_func = nullptr;
        llvm::Function *py_err_restore_func = nullptr;
        llvm::Function *py_err_set_object_func = nullptr;
        llvm::Function *py_err_set_string_func = nullptr;
        llvm::Function *py_err_clear_func = nullptr;
        llvm::Function *py_exception_matches_func = nullptr;
        llvm::Function *py_object_type_func = nullptr;
        llvm::Function *py_exception_set_cause_func = nullptr;

        // Attribute/name deletion API functions
        llvm::Function *py_object_delattr_func = nullptr; // int PyObject_DelAttr(PyObject*, PyObject*)
        llvm::Function *py_dict_delitem_func = nullptr;   // int PyDict_DelItem(PyObject*, PyObject*)
        llvm::Function *py_cell_set_func = nullptr;       // int PyCell_Set(PyObject*, PyObject*)

        // Format/string API functions (f-string support)
        llvm::Function *py_object_format_func = nullptr; // PyObject* PyObject_Format(PyObject*, PyObject*)
        llvm::Function *py_object_repr_func = nullptr;   // PyObject* PyObject_Repr(PyObject*)
        llvm::Function *py_object_ascii_func = nullptr;  // PyObject* PyObject_ASCII(PyObject*)

        // Import API functions
        llvm::Function *py_import_importmodule_func = nullptr; // PyObject* PyImport_ImportModuleLevelObject(...)

        // Function creation API (MAKE_FUNCTION / SET_FUNCTION_ATTRIBUTE opcodes)
        llvm::Function *py_function_new_func = nullptr;             // PyObject* PyFunction_New(PyObject* code, PyObject* globals)
        llvm::Function *py_function_set_defaults_func = nullptr;    // int PyFunction_SetDefaults(PyObject* func, PyObject* defaults)
        llvm::Function *py_function_set_kwdefaults_func = nullptr;  // int PyFunction_SetKwDefaults(PyObject* func, PyObject* kwdefaults)
        llvm::Function *py_function_set_annotations_func = nullptr; // int PyFunction_SetAnnotations(PyObject* func, PyObject* annotations)
        llvm::Function *py_function_set_closure_func = nullptr;     // int PyFunction_SetClosure(PyObject* func, PyObject* closure)

        // Box/Unbox API functions (Phase 1 Type System)
        llvm::Function *py_long_aslonglong_func = nullptr;   // long long PyLong_AsLongLong(PyObject*)
        llvm::Function *py_float_asdouble_func = nullptr;    // double PyFloat_AsDouble(PyObject*)
        llvm::Function *py_float_fromdouble_func = nullptr;  // PyObject* PyFloat_FromDouble(double)
        llvm::Function *py_bool_fromlong_func = nullptr;     // PyObject* PyBool_FromLong(long)

        // JIT helper functions
        llvm::Function *jit_call_with_kwargs_func = nullptr; // PyObject* jit_call_with_kwargs(...) for CALL_KW
        llvm::Function *jit_debug_trace_func = nullptr;      // void jit_debug_trace(...) for debugging
        llvm::Function *jit_debug_stack_func = nullptr;      // void jit_debug_stack(...) for debugging

        // Async generator support functions
        llvm::Function *jit_get_aiter_func = nullptr;        // PyObject* JITGetAIter(PyObject*) for GET_AITER
        llvm::Function *jit_get_anext_func = nullptr;        // PyObject* JITGetANext(PyObject*) for GET_ANEXT
        llvm::Function *jit_end_async_for_func = nullptr;    // int JITEndAsyncFor(PyObject*) for END_ASYNC_FOR
        llvm::Function *jit_async_gen_wrap_func = nullptr;   // PyObject* JITAsyncGenWrap(PyObject*) for ASYNC_GEN_WRAP
        llvm::Function *jit_async_gen_unwrap_func = nullptr; // PyObject* JITAsyncGenUnwrap(PyObject*) for unwrapping

    private:
        std::unique_ptr<llvm::orc::LLJIT> jit;
        std::unique_ptr<llvm::LLVMContext> context;
        int opt_level = 3;
        bool dump_ir = false;
        std::string last_ir;

        // Store references to Python objects we've incref'd (for cleanup)
        std::vector<PyObject *> stored_constants;
        std::vector<PyObject *> stored_names;

        // Runtime globals/builtins dicts for LOAD_GLOBAL (Bug #4 fix)
        PyObject *globals_dict_ptr = nullptr;
        PyObject *builtins_dict_ptr = nullptr;

        // Cache of already-compiled function names to prevent duplicate symbol errors
        std::unordered_set<std::string> compiled_functions;

        // Cache of generator metadata (actual total_locals after simulation)
        std::unordered_map<std::string, int> generator_total_locals;

        // Closure cells storage (for COPY_FREE_VARS / LOAD_DEREF)
        std::vector<PyObject *> stored_closure_cells;

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

        // Float-mode callable generators (native f64 -> f64 functions)
        nb::object create_float_callable_0(uint64_t func_ptr);
        nb::object create_float_callable_1(uint64_t func_ptr);
        nb::object create_float_callable_2(uint64_t func_ptr);
        nb::object create_float_callable_3(uint64_t func_ptr);
        nb::object create_float_callable_4(uint64_t func_ptr);

        // Bool-mode callable generators (native i64 -> bool functions)
        nb::object create_bool_callable_0(uint64_t func_ptr);
        nb::object create_bool_callable_1(uint64_t func_ptr);
        nb::object create_bool_callable_2(uint64_t func_ptr);
        nb::object create_bool_callable_3(uint64_t func_ptr);
        nb::object create_bool_callable_4(uint64_t func_ptr);

        // Int32-mode callable generators (native i32 functions)
        nb::object create_int32_callable_0(uint64_t func_ptr);
        nb::object create_int32_callable_1(uint64_t func_ptr);
        nb::object create_int32_callable_2(uint64_t func_ptr);
        nb::object create_int32_callable_3(uint64_t func_ptr);
        nb::object create_int32_callable_4(uint64_t func_ptr);

        // Float32-mode callable generators (native f32 functions)
        nb::object create_float32_callable_0(uint64_t func_ptr);
        nb::object create_float32_callable_1(uint64_t func_ptr);
        nb::object create_float32_callable_2(uint64_t func_ptr);
        nb::object create_float32_callable_3(uint64_t func_ptr);
        nb::object create_float32_callable_4(uint64_t func_ptr);

        // Complex128-mode callable generators (native {double,double} functions)
        nb::object create_complex128_callable_0(uint64_t func_ptr);
        nb::object create_complex128_callable_1(uint64_t func_ptr);
        nb::object create_complex128_callable_2(uint64_t func_ptr);

        // Ptr-mode callable generators (ptr + index operations)
        nb::object create_ptr_callable_2(uint64_t func_ptr);
        nb::object create_ptr_callable_3(uint64_t func_ptr);

        // Vec4f-mode callable generators (<4 x float> SIMD)
        nb::object create_vec4f_callable_2(uint64_t func_ptr);

        // Vec8i-mode callable generators (<8 x i32> SIMD)
        nb::object create_vec8i_callable_2(uint64_t func_ptr);

        // Complex64-mode callable generators ({float, float})
        nb::object create_complex64_callable_0(uint64_t func_ptr);
        nb::object create_complex64_callable_1(uint64_t func_ptr);
        nb::object create_complex64_callable_2(uint64_t func_ptr);

        // OptionalF64-mode callable generators ({bool, double})
        nb::object create_optional_f64_callable_0(uint64_t func_ptr);
        nb::object create_optional_f64_callable_1(uint64_t func_ptr);
        nb::object create_optional_f64_callable_2(uint64_t func_ptr);

        void optimize_module(llvm::Module &module, llvm::Function *func);
    };

}
