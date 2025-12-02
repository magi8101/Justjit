// opcode_handlers.h - Modular opcode handler declarations
#pragma once

#include "jit_core.h"
#include "opcodes.h"
#include <llvm/IR/IRBuilder.h>
#include <vector>

namespace justjit {

// Forward declarations for opcode handler context
struct OpcodeContext {
    llvm::IRBuilder<>& builder;
    llvm::LLVMContext& context;
    llvm::Function* func;
    std::vector<llvm::Value*>& stack;
    std::vector<llvm::Value*>& locals;
    
    // Types
    llvm::Type* ptr_type;
    llvm::Type* i64_type;
    
    // Python API functions (from JITCore)
    JITCore* jit_core;
    
    // Constants and names
    const std::vector<int64_t>& int_constants;
    const std::vector<PyObject*>& obj_constants;
    const std::vector<PyObject*>& name_objects;
    const std::vector<PyObject*>& global_objects;
    
    // Jump targets for control flow
    std::unordered_map<int, llvm::BasicBlock*>& jump_targets;
    
    // Current instruction info
    size_t instr_index;
    const std::vector<Instruction>& instructions;
};

// ============================================================================
// Exception Handling Opcodes (exception_ops.cpp)
// ============================================================================
bool handle_PUSH_EXC_INFO(OpcodeContext& ctx, const Instruction& instr);
bool handle_POP_EXCEPT(OpcodeContext& ctx, const Instruction& instr);
bool handle_CHECK_EXC_MATCH(OpcodeContext& ctx, const Instruction& instr);
bool handle_RAISE_VARARGS(OpcodeContext& ctx, const Instruction& instr);
bool handle_RERAISE(OpcodeContext& ctx, const Instruction& instr);
bool handle_CLEANUP_THROW(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Advanced Call Opcodes (call_ops.cpp)
// ============================================================================
bool handle_CALL_KW(OpcodeContext& ctx, const Instruction& instr);
bool handle_CALL_FUNCTION_EX(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Function/Class Creation Opcodes (function_ops.cpp)
// ============================================================================
bool handle_MAKE_FUNCTION(OpcodeContext& ctx, const Instruction& instr);
bool handle_LOAD_BUILD_CLASS(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Import Opcodes (import_ops.cpp)
// ============================================================================
bool handle_IMPORT_NAME(OpcodeContext& ctx, const Instruction& instr);
bool handle_IMPORT_FROM(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Generator/Async Opcodes (generator_ops.cpp)
// ============================================================================
bool handle_YIELD_VALUE(OpcodeContext& ctx, const Instruction& instr);
bool handle_RETURN_GENERATOR(OpcodeContext& ctx, const Instruction& instr);
bool handle_SEND(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Pattern Matching Opcodes (match_ops.cpp)
// ============================================================================
bool handle_MATCH_MAPPING(OpcodeContext& ctx, const Instruction& instr);
bool handle_MATCH_SEQUENCE(OpcodeContext& ctx, const Instruction& instr);
bool handle_MATCH_KEYS(OpcodeContext& ctx, const Instruction& instr);
bool handle_MATCH_CLASS(OpcodeContext& ctx, const Instruction& instr);

// ============================================================================
// Format Opcodes (format_ops.cpp)
// ============================================================================
bool handle_FORMAT_SIMPLE(OpcodeContext& ctx, const Instruction& instr);
bool handle_FORMAT_WITH_SPEC(OpcodeContext& ctx, const Instruction& instr);
bool handle_CONVERT_VALUE(OpcodeContext& ctx, const Instruction& instr);

}  // namespace justjit
