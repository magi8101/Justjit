// exception_ops.cpp - Exception handling opcode handlers
#include "opcode_handlers.h"
#include <llvm/IR/IRBuilder.h>

namespace justjit {

bool handle_PUSH_EXC_INFO(OpcodeContext& ctx, const Instruction& instr) {
    // PUSH_EXC_INFO: Push exception info onto stack
    // This is used at the start of an except block
    // Stack before: exc
    // Stack after: prev_exc, exc (pushes previous exception state)
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    if (stack.empty()) {
        return false;
    }
    
    // For now, implement as identity - we need proper exception state tracking
    // The exc is already on stack, we need to push prev_exc below it
    
    llvm::Value* exc = stack.back();
    
    // Get previous exception - PyErr_GetRaisedException()
    // For now, push None as placeholder
    llvm::Value* py_none_ptr = llvm::ConstantInt::get(ctx.i64_type, reinterpret_cast<uint64_t>(Py_None));
    llvm::Value* prev_exc = builder.CreateIntToPtr(py_none_ptr, ctx.ptr_type, "prev_exc");
    builder.CreateCall(jit->py_incref_func, {prev_exc});
    
    // Stack order: prev_exc, exc
    stack.pop_back();
    stack.push_back(prev_exc);
    stack.push_back(exc);
    
    return true;
}

bool handle_POP_EXCEPT(OpcodeContext& ctx, const Instruction& instr) {
    // POP_EXCEPT: Pop exception from stack at end of except block
    // Pops exception and restores previous exception state
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    if (stack.empty()) {
        return false;
    }
    
    llvm::Value* exc = stack.back(); stack.pop_back();
    
    // Decref the exception
    if (exc->getType()->isPointerTy()) {
        builder.CreateCall(jit->py_decref_func, {exc});
    }
    
    return true;
}

bool handle_CHECK_EXC_MATCH(OpcodeContext& ctx, const Instruction& instr) {
    // CHECK_EXC_MATCH: Check if TOS is an exception matching TOS1
    // Stack: exc_type, exc -> exc_type, bool_result
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    if (stack.size() < 2) {
        return false;
    }
    
    llvm::Value* exc = stack.back(); stack.pop_back();
    llvm::Value* exc_type = stack.back(); stack.pop_back();
    
    // PyErr_GivenExceptionMatches(exc, exc_type) - returns int
    // Declare the function inline
    llvm::FunctionType* match_type = llvm::FunctionType::get(
        ctx.i64_type, {ctx.ptr_type, ctx.ptr_type}, false);
    llvm::FunctionCallee match_func = jit->py_incref_func->getParent()->getOrInsertFunction(
        "PyErr_GivenExceptionMatches", match_type);
    
    llvm::Value* match = builder.CreateCall(match_func, {exc, exc_type}, "exc_match");
    
    // Convert int result to Python bool
    llvm::Value* is_match = builder.CreateICmpNE(match, llvm::ConstantInt::get(ctx.i64_type, 0));
    
    llvm::Value* py_true_ptr = llvm::ConstantInt::get(ctx.i64_type, reinterpret_cast<uint64_t>(Py_True));
    llvm::Value* py_true = builder.CreateIntToPtr(py_true_ptr, ctx.ptr_type);
    llvm::Value* py_false_ptr = llvm::ConstantInt::get(ctx.i64_type, reinterpret_cast<uint64_t>(Py_False));
    llvm::Value* py_false = builder.CreateIntToPtr(py_false_ptr, ctx.ptr_type);
    
    llvm::Value* result = builder.CreateSelect(is_match, py_true, py_false, "match_result");
    builder.CreateCall(jit->py_incref_func, {result});
    
    // Put exc_type back, then result
    stack.push_back(exc_type);
    stack.push_back(result);
    
    // Decref exc (we consumed it)
    if (exc->getType()->isPointerTy()) {
        builder.CreateCall(jit->py_decref_func, {exc});
    }
    
    return true;
}

bool handle_RAISE_VARARGS(OpcodeContext& ctx, const Instruction& instr) {
    // RAISE_VARARGS: Raise an exception
    // arg=0: re-raise current exception
    // arg=1: raise TOS
    // arg=2: raise TOS1 from TOS (chained exception)
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    int argc = instr.arg;
    
    if (argc == 0) {
        // Re-raise - need PyErr_SetRaisedException or similar
        // For now, just return error
        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx.context, 0));
        stack.push_back(null_ptr);
        return true;
    }
    
    if (stack.size() < static_cast<size_t>(argc)) {
        return false;
    }
    
    if (argc >= 1) {
        llvm::Value* exc = stack.back(); stack.pop_back();
        
        if (argc == 2) {
            llvm::Value* cause = stack.back(); stack.pop_back();
            // Set __cause__ on exception
            // PyException_SetCause(exc, cause)
            builder.CreateCall(jit->py_decref_func, {cause});
        }
        
        // Raise the exception: PyErr_SetObject(type, value)
        // Declare PyErr_SetObject inline
        llvm::FunctionType* set_err_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context), {ctx.ptr_type, ctx.ptr_type}, false);
        llvm::FunctionCallee set_err_func = jit->py_incref_func->getParent()->getOrInsertFunction(
            "PyErr_SetObject", set_err_type);
        
        // Get the type of the exception
        llvm::FunctionType* get_type_type = llvm::FunctionType::get(ctx.ptr_type, {ctx.ptr_type}, false);
        llvm::FunctionCallee get_type_func = jit->py_incref_func->getParent()->getOrInsertFunction(
            "PyObject_Type", get_type_type);
        
        llvm::Value* exc_type = builder.CreateCall(get_type_func, {exc}, "exc_type");
        builder.CreateCall(set_err_func, {exc_type, exc});
        builder.CreateCall(jit->py_decref_func, {exc_type});
    }
    
    // Return NULL to signal exception
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx.context, 0));
    stack.push_back(null_ptr);
    
    return true;
}

bool handle_CLEANUP_THROW(OpcodeContext& ctx, const Instruction& instr) {
    // CLEANUP_THROW: Clean up after throw in generator
    // This is complex - placeholder for now
    return false;
}

bool handle_RERAISE(OpcodeContext& ctx, const Instruction& instr) {
    // RERAISE: Re-raise the exception
    // arg specifies depth of exception handlers
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    // Get current exception and re-raise it
    // For now, return NULL to indicate exception
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx.context, 0));
    stack.push_back(null_ptr);
    
    return true;
}

}  // namespace justjit
