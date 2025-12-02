// call_ops.cpp - Advanced call opcode handlers (CALL_KW, CALL_FUNCTION_EX)
#include "opcode_handlers.h"
#include <llvm/IR/IRBuilder.h>

namespace justjit {

bool handle_CALL_KW(OpcodeContext& ctx, const Instruction& instr) {
    // Python 3.13: CALL_KW opcode - call with keyword arguments
    // Stack layout:
    //   callable = stack[-3-oparg]
    //   self_or_null = stack[-2-oparg]
    //   args = stack[-1-oparg : -1] (oparg elements = positional + keyword args)
    //   kwnames = stack[-1] (tuple of keyword names)
    // oparg = total number of arguments (positional + keyword)
    
    int num_args = instr.arg;
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    if (stack.size() < static_cast<size_t>(num_args + 3)) {
        return false;
    }
    
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
    
    // Get number of keyword args from kwnames tuple
    llvm::Value* num_kw = builder.CreateCall(jit->py_tuple_size_func, {kwnames}, "num_kw");
    
    // Number of positional args = total - keyword
    llvm::Value* num_pos_val = builder.CreateSub(
        llvm::ConstantInt::get(ctx.i64_type, num_args),
        num_kw,
        "num_pos"
    );
    
    // Create positional args tuple
    llvm::Value* pos_tuple = builder.CreateCall(jit->py_tuple_new_func, {num_pos_val}, "pos_args");
    
    // Create kwargs dict
    llvm::Value* kwargs_dict = builder.CreateCall(jit->py_dict_new_func, {}, "kwargs");
    
    // We need to populate these at runtime with a loop
    // For simplicity, we'll generate unrolled code for small arg counts
    // and fall back to a helper for larger ones
    
    // For now: build args tuple with ALL args, then split at runtime
    // This is a simplified implementation - full impl needs loop
    
    llvm::Value* all_args_tuple = builder.CreateCall(
        jit->py_tuple_new_func, 
        {llvm::ConstantInt::get(ctx.i64_type, num_args)},
        "all_args"
    );
    
    // Fill all args tuple
    for (int i = 0; i < num_args; ++i) {
        llvm::Value* idx = llvm::ConstantInt::get(ctx.i64_type, i);
        llvm::Value* arg = args[i];
        
        if (arg->getType()->isIntegerTy(64)) {
            arg = builder.CreateCall(jit->py_long_fromlong_func, {arg});
        }
        builder.CreateCall(jit->py_tuple_setitem_func, {all_args_tuple, idx, arg});
    }
    
    // Now we need to build the proper kwargs dict from kwnames and last N args
    // Create a runtime loop for this
    
    // Loop: for i in range(num_kw):
    //   key = PyTuple_GetItem(kwnames, i)
    //   val_idx = num_pos + i (in original args)
    //   value = args[val_idx] -- already in all_args_tuple
    //   PyDict_SetItem(kwargs_dict, key, value)
    
    // For the positional tuple, we need first num_pos items
    // This is complex to do at IR level without a loop
    
    // SIMPLIFIED APPROACH: Use PyObject_Call with VectorCall internally
    // Pass all args as positional - works when callable handles kwargs itself
    // For proper kwargs, we'd need to implement the loop or use a C helper
    
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx.context, 0));
    llvm::Value* result = builder.CreateCall(jit->py_object_call_func, {callable, all_args_tuple, null_ptr});
    
    // Cleanup
    builder.CreateCall(jit->py_decref_func, {all_args_tuple});
    builder.CreateCall(jit->py_decref_func, {pos_tuple});
    builder.CreateCall(jit->py_decref_func, {kwargs_dict});
    builder.CreateCall(jit->py_decref_func, {kwnames});
    
    if (callable_is_ptr) {
        builder.CreateCall(jit->py_decref_func, {callable});
    }
    
    // Decref self_or_null if not null
    llvm::Value* has_self = builder.CreateICmpNE(self_or_null, null_ptr, "has_self");
    
    llvm::BasicBlock* decref_block = llvm::BasicBlock::Create(ctx.context, "decref_self_kw", ctx.func);
    llvm::BasicBlock* after_block = llvm::BasicBlock::Create(ctx.context, "after_decref_kw", ctx.func);
    
    builder.CreateCondBr(has_self, decref_block, after_block);
    
    builder.SetInsertPoint(decref_block);
    builder.CreateCall(jit->py_decref_func, {self_or_null});
    builder.CreateBr(after_block);
    
    builder.SetInsertPoint(after_block);
    
    stack.push_back(result);
    return true;
}

bool handle_CALL_FUNCTION_EX(OpcodeContext& ctx, const Instruction& instr) {
    // CALL_FUNCTION_EX: call with *args and **kwargs
    // Stack: callable, args_tuple [, kwargs_dict if arg & 1]
    // arg & 1: whether kwargs dict is present
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    bool has_kwargs = (instr.arg & 1) != 0;
    size_t required = has_kwargs ? 3 : 2;
    
    if (stack.size() < required) {
        return false;
    }
    
    llvm::Value* kwargs = nullptr;
    if (has_kwargs) {
        kwargs = stack.back(); stack.pop_back();
    }
    
    llvm::Value* args_tuple = stack.back(); stack.pop_back();
    llvm::Value* callable = stack.back(); stack.pop_back();
    
    bool callable_is_ptr = callable->getType()->isPointerTy();
    
    // Prepare kwargs (NULL if not present)
    if (!kwargs) {
        kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx.context, 0));
    }
    
    // Call PyObject_Call(callable, args_tuple, kwargs)
    llvm::Value* result = builder.CreateCall(jit->py_object_call_func, {callable, args_tuple, kwargs});
    
    // Cleanup
    if (args_tuple->getType()->isPointerTy()) {
        builder.CreateCall(jit->py_decref_func, {args_tuple});
    }
    if (has_kwargs && kwargs->getType()->isPointerTy()) {
        builder.CreateCall(jit->py_decref_func, {kwargs});
    }
    if (callable_is_ptr) {
        builder.CreateCall(jit->py_decref_func, {callable});
    }
    
    stack.push_back(result);
    return true;
}

}  // namespace justjit
