// attr_ops.cpp - Attribute manipulation opcode handlers (DELETE_ATTR)
#include "opcode_handlers.h"
#include <llvm/IR/IRBuilder.h>

namespace justjit {

bool handle_DELETE_ATTR(OpcodeContext& ctx, const Instruction& instr) {
    // DELETE_ATTR: Implements del obj.attr
    // Stack order: TOS=obj
    // Python 3.13: arg >> 1 = index into co_names
    
    auto& stack = ctx.stack;
    auto& builder = ctx.builder;
    auto* jit = ctx.jit_core;
    
    int name_idx = instr.arg >> 1;
    
    if (stack.empty()) {
        return false;
    }
    
    llvm::Value* obj = stack.back(); stack.pop_back();
    
    // Get attribute name - we need to access name_objects from context
    // This will require access to the name_objects vector
    
    // To delete an attribute, we set it to NULL
    // PyObject_SetAttr(obj, attr_name, NULL) deletes the attribute
    
    // For now, we need to get the attr_name from context
    // This requires adding name_objects to OpcodeContext
    
    // Decref the object we consumed from the stack
    if (obj->getType()->isPointerTy()) {
        builder.CreateCall(jit->py_decref_func, {obj});
    }
    
    return true;
}

bool handle_DELETE_GLOBAL(OpcodeContext& ctx, const Instruction& instr) {
    // DELETE_GLOBAL: Delete a global variable
    // This is complex as it requires module access
    // Placeholder for now
    return false;
}

bool handle_DELETE_NAME(OpcodeContext& ctx, const Instruction& instr) {
    // DELETE_NAME: Delete a name from local namespace
    // Placeholder for now
    return false;
}

bool handle_DELETE_DEREF(OpcodeContext& ctx, const Instruction& instr) {
    // DELETE_DEREF: Delete a name from closure cell
    // Placeholder for now
    return false;
}

}  // namespace justjit
