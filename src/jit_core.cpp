#include "jit_core.h"
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
}

bool JITCore::compile_function(nb::list py_instructions, nb::list py_constants, nb::list py_names, nb::list py_globals, const std::string& name, int param_count, int total_locals) {
    if (!jit) {
        return false;
    }
    
    // Convert Python instructions list to C++ vector
    std::vector<Instruction> instructions;
    for (size_t i = 0; i < py_instructions.size(); ++i) {
        nb::dict instr_dict = nb::cast<nb::dict>(py_instructions[i]);
        Instruction instr;
        instr.opcode = nb::cast<uint8_t>(instr_dict["opcode"]);
        instr.arg = nb::cast<uint16_t>(instr_dict["arg"]);
        instr.argval = nb::cast<uint16_t>(instr_dict["argval"]);  // Get actual jump target from Python
        instr.offset = nb::cast<uint16_t>(instr_dict["offset"]);
        instructions.push_back(instr);
    }
    
    // Convert Python constants list - support both int64 and PyObject*
    std::vector<int64_t> int_constants;
    std::vector<PyObject*> obj_constants;
    for (size_t i = 0; i < py_constants.size(); ++i) {
        nb::object const_obj = py_constants[i];
        
        // Try to convert to int64 first
        try {
            int64_t int_val = nb::cast<int64_t>(const_obj);
            int_constants.push_back(int_val);
            obj_constants.push_back(nullptr);  // Mark as int constant
        } catch (...) {
            // If not an int, store as PyObject*
            int_constants.push_back(0);
            PyObject* py_obj = const_obj.ptr();
            Py_INCREF(py_obj);  // Keep reference alive
            obj_constants.push_back(py_obj);
        }
    }
    
    // Extract names (used by LOAD_ATTR, LOAD_GLOBAL, etc)
    std::vector<PyObject*> name_objects;
    for (size_t i = 0; i < py_names.size(); ++i) {
        nb::object name_obj = py_names[i];
        PyObject* py_name = name_obj.ptr();
        Py_INCREF(py_name);  // Keep reference alive
        name_objects.push_back(py_name);
    }
    
    // Extract globals (used by LOAD_GLOBAL)
    std::vector<PyObject*> global_objects;
    for (size_t i = 0; i < py_globals.size(); ++i) {
        nb::object global_obj = py_globals[i];
        if (global_obj.is_none()) {
            global_objects.push_back(nullptr);
        } else {
            PyObject* py_global = global_obj.ptr();
            Py_INCREF(py_global);  // Keep reference alive
            global_objects.push_back(py_global);
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
        
        // Create allocas only for actual locals needed (not 256!)
        // In object mode, all locals are PyObject* (ptr type)
        llvm::IRBuilder<> alloca_builder(entry, entry->begin());
        for (int i = 0; i < total_locals; ++i) {
            local_allocas[i] = alloca_builder.CreateAlloca(
                ptr_type, nullptr, "local_" + std::to_string(i)
            );
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
            
            if (instr.opcode == 97 || instr.opcode == 100) {  // POP_JUMP_IF_FALSE / POP_JUMP_IF_TRUE
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "block_" + std::to_string(target_offset), func
                    );
                }
            } else if (instr.opcode == 77) {  // JUMP_BACKWARD
                // Use argval which Python's dis module already calculated for us
                int target_offset = instr.argval;
                if (!jump_targets.count(target_offset)) {
                    jump_targets[target_offset] = llvm::BasicBlock::Create(
                        *local_context, "loop_" + std::to_string(target_offset), func
                    );
                }
            }
        }
        
        // Second pass: Generate code
        for (size_t i = 0; i < instructions.size(); ++i) {
            // If this offset is a jump target, switch to that block
            if (jump_targets.count(instructions[i].offset) && jump_targets[instructions[i].offset] != builder.GetInsertBlock()) {
                if (!builder.GetInsertBlock()->getTerminator()) {
                    builder.CreateBr(jump_targets[instructions[i].offset]);
                }
                builder.SetInsertPoint(jump_targets[instructions[i].offset]);
            }
            
            const auto& instr = instructions[i];
            
            // Python 3.13 opcodes (from dis module)
            if (instr.opcode == 149) {  // RESUME
                continue;
            }
            else if (instr.opcode == 85) {  // LOAD_FAST
                if (local_allocas.count(instr.arg)) {
                    // In object mode, load PyObject* from local
                    llvm::Value* loaded = builder.CreateLoad(ptr_type, local_allocas[instr.arg], "load_local_" + std::to_string(instr.arg));
                    stack.push_back(loaded);
                }
            }
            else if (instr.opcode == 88) {  // LOAD_FAST_LOAD_FAST
                // Python 3.13: Pushes co_varnames[arg>>4] then co_varnames[arg&15]
                int first_local = instr.arg >> 4;
                int second_local = instr.arg & 0xF;
                if (local_allocas.count(first_local)) {
                    llvm::Value* loaded1 = builder.CreateLoad(ptr_type, local_allocas[first_local], "load_local_" + std::to_string(first_local));
                    stack.push_back(loaded1);
                }
                if (local_allocas.count(second_local)) {
                    llvm::Value* loaded2 = builder.CreateLoad(ptr_type, local_allocas[second_local], "load_local_" + std::to_string(second_local));
                    stack.push_back(loaded2);
                }
            }
            else if (instr.opcode == 83) {  // LOAD_CONST
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
                        // int64 constant
                        llvm::Value* const_val = llvm::ConstantInt::get(i64_type, int_constants[instr.arg]);
                        stack.push_back(const_val);
                    }
                }
            }
            else if (instr.opcode == 110) {  // STORE_FAST
                if (!stack.empty()) {
                    builder.CreateStore(stack.back(), local_allocas[instr.arg]);
                    stack.pop_back();
                }
            }
            else if (instr.opcode == 45) {  // BINARY_OP
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
                        // Box both operands to PyObject* if needed
                        if (first->getType()->isIntegerTy(64)) {
                            first = builder.CreateCall(py_long_fromlong_func, {first});
                        }
                        if (second->getType()->isIntegerTy(64)) {
                            second = builder.CreateCall(py_long_fromlong_func, {second});
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
                            default:
                                // For unsupported ops, fall back to ADD
                                result = builder.CreateCall(py_number_add_func, {first, second});
                                break;
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
                            case 8:  // POW (a ** b)
                                // For now, treat as multiplication (proper implementation needs pow function)
                                result = builder.CreateMul(first, second, "pow_stub");
                                break;
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
            else if (instr.opcode == 58) {  // COMPARE_OP
                if (stack.size() >= 2) {
                    llvm::Value* rhs = stack.back(); stack.pop_back();
                    llvm::Value* lhs = stack.back(); stack.pop_back();
                    
                    // Python 3.13 encoding: (op_code << 5) | flags
                    // Extraction: op_code = arg >> 5
                    // Compare operations: 0=<, 1=<=, 2===, 3=!=, 4=>, 5=>=
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
                    
                    if (cmp_result) {
                        stack.push_back(builder.CreateZExt(cmp_result, llvm::Type::getInt64Ty(*local_context)));
                    }
                }
            }
            else if (instr.opcode == 97 || instr.opcode == 100) {  // POP_JUMP_IF_FALSE / POP_JUMP_IF_TRUE
                if (!stack.empty() && i + 1 < instructions.size()) {
                    llvm::Value* cond = stack.back(); stack.pop_back();
                    
                    llvm::Value* bool_cond = builder.CreateICmpNE(
                        cond,
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0),
                        "tobool"
                    );
                    
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
                        // POP_JUMP_IF_FALSE: jump if condition is FALSE (0), continue if TRUE (non-zero)
                        // POP_JUMP_IF_TRUE: jump if condition is TRUE (non-zero), continue if FALSE (0)
                        if (instr.opcode == 97) {  // POP_JUMP_IF_FALSE
                            // Jump to target when condition is false (0), continue to next when true (non-zero)
                            builder.CreateCondBr(bool_cond, jump_targets[next_offset], jump_targets[target_offset]);
                        } else {  // POP_JUMP_IF_TRUE (opcode 100)
                            // Jump to target when condition is true (non-zero), continue to next when false (0)
                            builder.CreateCondBr(bool_cond, jump_targets[target_offset], jump_targets[next_offset]);
                        }
                    }
                }
            }
            else if (instr.opcode == 77) {  // JUMP_BACKWARD
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
                // Create a new block for any code after the loop (won't be reached but prevents issues)
                llvm::BasicBlock* after_loop = llvm::BasicBlock::Create(
                    *local_context, "after_loop_" + std::to_string(i), func
                );
                builder.SetInsertPoint(after_loop);
            }
            else if (instr.opcode == 103) {  // RETURN_CONST
                // Return a constant from co_consts without using stack
                if (!builder.GetInsertBlock()->getTerminator()) {
                    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), instr.arg));
                }
            }
            else if (instr.opcode == 36) {  // RETURN_VALUE
                if (!builder.GetInsertBlock()->getTerminator()) {
                    if (!stack.empty()) {
                        builder.CreateRet(stack.back());
                    } else {
                        builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0));
                    }
                }
            }
            else if (instr.opcode == 47) {  // BUILD_LIST
                // arg is the number of items to pop from stack
                int count = instr.arg;
                
                // Create new list with PyList_New(count)
                llvm::Value* count_val = llvm::ConstantInt::get(i64_type, count);
                llvm::Value* new_list = builder.CreateCall(py_list_new_func, {count_val});
                
                // Pop items from stack and add to list (in reverse order)
                std::vector<llvm::Value*> items;
                for (int i = 0; i < count; ++i) {
                    if (!stack.empty()) {
                        items.push_back(stack.back());
                        stack.pop_back();
                    }
                }
                
                // Add items to list in correct order
                for (int i = count - 1; i >= 0; --i) {
                    llvm::Value* index_val = llvm::ConstantInt::get(i64_type, count - 1 - i);
                    llvm::Value* item = items[i];
                    
                    // Check if item is int64, convert to PyLong if needed
                    // For now assume items are already PyObject* or convert ints
                    if (item->getType()->isIntegerTy(64)) {
                        // Convert int64 to PyObject*
                        item = builder.CreateCall(py_long_fromlong_func, {item});
                    }
                    
                    // PyList_SetItem steals reference, but we need to keep one for the item
                    builder.CreateCall(py_incref_func, {item});
                    builder.CreateCall(py_list_setitem_func, {new_list, index_val, item});
                }
                
                stack.push_back(new_list);
            }
            else if (instr.opcode == 52) {  // BUILD_TUPLE
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
                    }
                    
                    // PyTuple_SetItem steals reference
                    builder.CreateCall(py_incref_func, {item});
                    builder.CreateCall(py_tuple_setitem_func, {new_tuple, index_val, item});
                }
                
                stack.push_back(new_tuple);
            }
            else if (instr.opcode == 5) {  // BINARY_SUBSCR
                // Implements container[key]
                if (stack.size() >= 2) {
                    llvm::Value* key = stack.back(); stack.pop_back();
                    llvm::Value* container = stack.back(); stack.pop_back();
                    
                    // Convert int64 key to PyObject* if needed
                    if (key->getType()->isIntegerTy(64)) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                    }
                    
                    // PyObject_GetItem returns new reference
                    llvm::Value* result = builder.CreateCall(py_object_getitem_func, {container, key});
                    
                    // Decrement key refcount if we created it
                    if (!key->getType()->isIntegerTy(64)) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == 39) {  // STORE_SUBSCR (was 25!)
                // Implements container[key] = value
                // Stack: TOS=value, TOS1=container, TOS2=key
                if (stack.size() >= 3) {
                    llvm::Value* value = stack.back(); stack.pop_back();       // TOS
                    llvm::Value* container = stack.back(); stack.pop_back();   // TOS1
                    llvm::Value* key = stack.back(); stack.pop_back();         // TOS2
                    
                    // Convert int64 key to PyObject* if needed
                    if (key->getType()->isIntegerTy(64)) {
                        key = builder.CreateCall(py_long_fromlong_func, {key});
                    }
                    
                    // Convert int64 value to PyObject* if needed
                    if (value->getType()->isIntegerTy(64)) {
                        value = builder.CreateCall(py_long_fromlong_func, {value});
                    }
                    
                    // PyObject_SetItem(container, key, value) - returns 0 on success
                    builder.CreateCall(py_object_setitem_func, {container, key, value});
                    
                    // Decrement temp refs if we created them
                    if (!key->getType()->isIntegerTy(64)) {
                        builder.CreateCall(py_decref_func, {key});
                    }
                }
            }
            else if (instr.opcode == 82) {  // LOAD_ATTR
                // Implements obj.attr - arg is index into co_names for attribute name
                if (!stack.empty() && instr.arg < name_objects.size()) {
                    llvm::Value* obj = stack.back(); stack.pop_back();
                    
                    // Get attribute name from names (PyUnicode string)
                    llvm::Value* attr_name_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(name_objects[instr.arg])
                    );
                    llvm::Value* attr_name = builder.CreateIntToPtr(attr_name_ptr, ptr_type);
                    
                    // PyObject_GetAttr returns new reference
                    llvm::Value* result = builder.CreateCall(py_object_getattr_func, {obj, attr_name});
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == 91) {  // LOAD_GLOBAL
                // Python 3.13: LOAD_GLOBAL loads global variable
                // arg >> 1 = index into co_names
                // arg & 1 = if set, push NULL after global (for calling convention)
                int name_idx = instr.arg >> 1;
                bool push_null = (instr.arg & 1) != 0;
                
                if (name_idx < global_objects.size() && global_objects[name_idx] != nullptr) {
                    // Get global object
                    llvm::Value* global_ptr = llvm::ConstantInt::get(
                        i64_type,
                        reinterpret_cast<uint64_t>(global_objects[name_idx])
                    );
                    llvm::Value* global_obj = builder.CreateIntToPtr(global_ptr, ptr_type);
                    
                    // Increment refcount since we're putting it on the stack
                    builder.CreateCall(py_incref_func, {global_obj});
                    
                    stack.push_back(global_obj);
                    
                    // Push NULL after global if needed (Python 3.13 calling convention)
                    // This creates stack layout: [global, NULL] for CALL to use as [callable, self_or_null]
                    if (push_null) {
                        llvm::Value* null_ptr = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                        stack.push_back(null_ptr);
                    }
                }
            }
            else if (instr.opcode == 53) {  // CALL
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
                    
                    // Collect arguments in order
                    std::vector<llvm::Value*> args;
                    for (int i = 0; i < num_args; ++i) {
                        args.push_back(stack[base + 2 + i]);       // stack[-oparg+i]
                    }
                    
                    // Remove all CALL operands from stack
                    stack.erase(stack.begin() + base, stack.end());
                    
                    // Create args tuple
                    llvm::Value* args_count = llvm::ConstantInt::get(i64_type, num_args);
                    llvm::Value* args_tuple = builder.CreateCall(py_tuple_new_func, {args_count});
                    
                    // Fill tuple with args in correct order (already in correct order now!)
                    for (int i = 0; i < num_args; ++i) {
                        llvm::Value* index_val = llvm::ConstantInt::get(i64_type, i);
                        llvm::Value* arg = args[i];
                        
                        // Box int64 to PyObject* if needed
                        if (arg->getType()->isIntegerTy(64)) {
                            arg = builder.CreateCall(py_long_fromlong_func, {arg});
                        }
                        
                        // PyTuple_SetItem steals reference
                        builder.CreateCall(py_incref_func, {arg});
                        builder.CreateCall(py_tuple_setitem_func, {args_tuple, index_val, arg});
                    }
                    
                    // Call PyObject_Call(callable, args_tuple, NULL)
                    llvm::Value* null_kwargs = llvm::ConstantPointerNull::get(llvm::PointerType::get(*local_context, 0));
                    llvm::Value* result = builder.CreateCall(py_object_call_func, {callable, args_tuple, null_kwargs});
                    
                    // Decrement args_tuple refcount
                    builder.CreateCall(py_decref_func, {args_tuple});
                    
                    stack.push_back(result);
                }
            }
            else if (instr.opcode == 32) {  // POP_TOP
                if (!stack.empty()) {
                    stack.pop_back();
                }
            }
        }
        
        // Ensure current block has terminator
        if (!builder.GetInsertBlock()->getTerminator()) {
            if (!stack.empty()) {
                builder.CreateRet(stack.back());
            } else {
                builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0));
            }
        }
        
        // Third pass: Add terminators to any unterminated blocks
        for (auto& block : *func) {
            if (!block.getTerminator()) {
                builder.SetInsertPoint(&block);
                builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*local_context), 0));
            }
        }
        
        if (llvm::verifyFunction(*func, &llvm::errs())) {
            llvm::errs() << "Function verification failed\n";
            return false;
        }
        if (llvm::verifyFunction(*func, &llvm::errs())) {
            llvm::errs() << "Function verification failed\n";
            return false;
        }
        
        optimize_module(*module, func);
        
        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(local_context));
        
        auto err = jit->addIRModule(std::move(tsm));
        if (err) {
            llvm::errs() << "Failed to add module: " << toString(std::move(err)) << "\n";
            return false;
        }
        
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

}  // namespace justjit
