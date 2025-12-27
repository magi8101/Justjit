Internals
=========

This page describes how JustJIT works internally.

Architecture Overview
---------------------

JustJIT transforms Python functions through this pipeline:

.. code-block:: text

   Python Function
        |
        v
   Bytecode Extraction (dis module)
        |
        v
   LLVM IR Generation (JITCore)
        |
        v
   LLVM Optimization Passes
        |
        v
   Native Machine Code (ORC JIT)
        |
        v
   Callable Wrapper

The Bytecode-to-IR Pipeline
---------------------------

When you decorate a function with ``@jit``:

1. **Bytecode Extraction**

   JustJIT uses Python's ``dis`` module to extract bytecode instructions:

   .. code-block:: python

      for instr in dis.get_instructions(func):
          # instr.opcode, instr.arg, instr.offset

   Each instruction becomes a dict with ``opcode``, ``arg``, ``argval``, and ``offset``.

2. **Constants and Names**

   Constants (numbers, strings) and names (variable names, attributes) are extracted from the code object:

   .. code-block:: python

      constants = list(func.__code__.co_consts)
      names = list(func.__code__.co_names)

3. **LLVM IR Generation**

   The ``JITCore`` class processes each bytecode instruction and generates corresponding LLVM IR.

   For example, ``BINARY_OP`` with ``arg=0`` (addition) becomes:

   .. code-block:: cpp

      llvm::Value* lhs = stack.back(); stack.pop_back();
      llvm::Value* rhs = stack.back(); stack.pop_back();
      llvm::Value* result = builder.CreateAdd(lhs, rhs);
      stack.push_back(result);

4. **LLVM Optimization**

   The generated IR passes through LLVM's optimization pipeline:

   - Dead code elimination
   - Constant propagation
   - Loop optimizations
   - Vectorization (when applicable)

5. **Native Code Generation**

   LLVM's ORC JIT compiles the IR to native machine code and loads it into memory.

6. **Callable Wrapper**

   A Python callable is created that converts Python arguments to native types, calls the native function, and converts the result back to Python.

Stack-Based Compilation
-----------------------

Python bytecode is stack-based. JustJIT simulates this stack during IR generation:

.. code-block:: cpp

   std::vector<llvm::Value*> stack;

   // LOAD_CONST: Push constant onto stack
   stack.push_back(llvm::ConstantInt::get(i64_type, constant));

   // BINARY_OP: Pop two values, push result
   auto rhs = stack.back(); stack.pop_back();
   auto lhs = stack.back(); stack.pop_back();
   stack.push_back(builder.CreateAdd(lhs, rhs));

   // RETURN_VALUE: Return stack top
   builder.CreateRet(stack.back());

Type System
-----------

JustJIT uses a type system defined in ``type_system.h``:

.. code-block:: cpp

   enum class JITType : uint8_t {
       OBJECT = 0,    // PyObject*
       INT64 = 1,     // i64
       FLOAT64 = 2,   // f64
       BOOL = 3,      // i1
       INT32 = 5,     // i32
       FLOAT32 = 6,   // f32
       COMPLEX128 = 7,// {f64, f64}
       PTR_F64 = 8,   // ptr
       VEC4F = 9,     // <4 x f32>
       VEC8I = 10,    // <8 x i32>
       COMPLEX64 = 11,// {f32, f32}
       OPTIONAL_F64 = 12, // {i64, f64}
   };

Each mode sets the type for all values in the function.

Control Flow Graph
------------------

For complex control flow (loops, conditionals, pattern matching), JustJIT builds a CFG:

.. code-block:: cpp

   struct BasicBlockInfo {
       int start_offset;
       std::vector<int> predecessors;
       std::vector<int> successors;
       llvm::BasicBlock* llvm_block;
   };

Jump instructions create new basic blocks. The CFG is used for:

- Correct branch generation
- PHI node insertion at merge points
- Stack state tracking across branches

Exception Handling
------------------

JustJIT parses Python 3.11+ exception tables:

.. code-block:: python

   entries = _parse_exception_table(func)
   # Each entry: {start, end, target, depth, lasti}

Exception ranges map to LLVM try/catch patterns. The ``target`` offset becomes the exception handler basic block.

Generator Compilation
---------------------

Generators are compiled as state machines:

.. code-block:: cpp

   // State: 0=initial, 1..N=suspended at yield N, -1=done
   typedef PyObject* (*GeneratorStepFunc)(
       int32_t* state,
       PyObject** locals,
       PyObject* sent_value
   );

Each ``yield`` becomes a state transition:

1. Save local variables to the ``locals`` array
2. Set state to the yield number
3. Return the yielded value

On resume, the step function dispatches to the correct state.

Callable Wrappers
-----------------

The C++ code creates Python callables that bridge Python and native code:

.. code-block:: cpp

   nb::object create_int_callable_2(uint64_t func_ptr) {
       auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(func_ptr);
       return nb::cpp_function([fn_ptr](nb::object a, nb::object b) {
           int64_t arg0 = nb::cast<int64_t>(a);
           int64_t arg1 = nb::cast<int64_t>(b);
           int64_t result = fn_ptr(arg0, arg1);
           return result;
       });
   }

For struct types (complex, optional), pointer-based calling conventions are used to handle Windows x64 ABI requirements.

Source Files
------------

- ``src/jit_core.cpp`` - Main JIT implementation (15K+ lines)
- ``src/jit_core.h`` - JITCore class and data structures
- ``src/type_system.h`` - Type definitions
- ``src/bindings.cpp`` - Python bindings via nanobind
- ``src/justjit/__init__.py`` - Python wrapper and ``@jit`` decorator

Dependencies
------------

- **LLVM** (18+): Compiler infrastructure for IR and code generation
- **nanobind**: Python binding library
- **Python C API**: For object manipulation and type conversion
