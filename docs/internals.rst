Internals
=========

This page provides a deep dive into JustJIT's architecture and implementation. The core is implemented in C++ using LLVM as the compilation backend.

Source Code Overview
--------------------

JustJIT consists of approximately 16,000 lines of C++ code:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - File
     - Lines
     - Purpose
   * - ``src/jit_core.cpp``
     - 15,513
     - Main JIT compiler implementation
   * - ``src/jit_core.h``
     - 397
     - Class declarations and data structures
   * - ``src/type_system.h``
     - 87
     - Native type definitions (JITType enum)
   * - ``src/bindings.cpp``
     - 89
     - Python bindings via nanobind
   * - ``src/opcodes.h``
     - ~300
     - Python 3.13 opcode definitions
   * - ``src/justjit/__init__.py``
     - 972
     - Python wrapper and @jit decorator

Architecture Overview
---------------------

.. code-block:: text

                    Python Function
                          |
                          v
    ┌─────────────────────────────────────────────┐
    │              Python Layer                    │
    │  __init__.py: @jit decorator, bytecode      │
    │  extraction via dis module                  │
    └─────────────────────────────────────────────┘
                          |
                          v
    ┌─────────────────────────────────────────────┐
    │              C++ Layer (JITCore)             │
    │                                              │
    │  compile_function()     - Object mode        │
    │  compile_int_function() - Native i64         │
    │  compile_float_function() - Native f64       │
    │  compile_generator()    - State machine      │
    │  ... 11 compile methods total                │
    └─────────────────────────────────────────────┘
                          |
                          v
    ┌─────────────────────────────────────────────┐
    │              LLVM Layer                      │
    │                                              │
    │  IRBuilder   - Generate LLVM IR              │
    │  PassBuilder - Optimization passes           │
    │  ORC JIT     - Just-in-time compilation      │
    └─────────────────────────────────────────────┘
                          |
                          v
                   Native Machine Code

Python Layer (__init__.py)
--------------------------

The Python layer (972 lines) handles the user-facing API and bridges to C++:

The @jit Decorator
^^^^^^^^^^^^^^^^^^

The ``jit()`` function (line 96) is the main entry point:

.. code-block:: python

   def jit(
       func=None, *, opt_level=3, vectorize=True,
       inline=True, parallel=False, lazy=False, mode="auto"
   ):
       """JIT compile a Python function."""
       if func is None:
           def decorator(f):
               return _create_jit_wrapper(f, ...)
           return decorator
       return _create_jit_wrapper(func, ...)

**Supported Modes:**

- ``"auto"`` / ``"object"``: Full Python semantics (PyObject*)
- ``"int"``: Native 64-bit integers (i64)
- ``"float"``: Native 64-bit floats (f64)
- ``"bool"``: Native booleans
- ``"int32"``: 32-bit integers for C interop
- ``"float32"``: 32-bit floats for ML/SIMD
- ``"complex128"``: Double-precision complex {f64, f64}
- ``"complex64"``: Single-precision complex {f32, f32}
- ``"ptr"``: Pointer arithmetic for arrays
- ``"vec4f"``: SSE SIMD <4 x float>
- ``"vec8i"``: AVX SIMD <8 x i32>
- ``"optional_f64"``: Nullable float {has_value, value}

Bytecode Extraction
^^^^^^^^^^^^^^^^^^^

The ``_extract_bytecode()`` function (line 139) parses Python's ``dis`` module output:

.. code-block:: python

   def _extract_bytecode(func):
       JUMP_OPCODES = {
           "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
           "JUMP_FORWARD", "JUMP_BACKWARD", ...
       }
       instructions = []
       for instr in dis.get_instructions(func):
           if instr.opname == "CACHE":
               continue  # Skip adaptive interpreter placeholders
           # Only pass argval (jump target) for jump opcodes
           argval = instr.argval if instr.opname in JUMP_OPCODES else 0
           instructions.append({
               "opcode": instr.opcode,
               "arg": instr.arg or 0,
               "argval": argval,
               "offset": instr.offset,
           })
       return instructions

Exception Table Parsing
^^^^^^^^^^^^^^^^^^^^^^^

Python 3.11+ uses an exception table for try/except. The ``_parse_exception_table()`` function (line 220) decodes it:

.. code-block:: python

   def _parse_exception_table(func):
       table = func.__code__.co_exceptiontable
       def read_varint(data, pos):
           """Read big-endian varint: bit 6 = continuation, bits 0-5 = value."""
           b = data[pos]
           val = b & 0x3F
           while b & 0x40:  # Continuation bit
               val <<= 6
               b = data[pos := pos + 1]
               val |= b & 0x3F
           return val, pos + 1

       entries = []
       i = 0
       while i < len(table):
           start, i = read_varint(table, i)
           length, i = read_varint(table, i)
           target, i = read_varint(table, i)
           depth_lasti, i = read_varint(table, i)
           entries.append({
               "start": start * 2,    # Convert to byte offset
               "end": (start + length) * 2,
               "target": target * 2,
               "depth": depth_lasti >> 1,
               "lasti": bool(depth_lasti & 1),
           })
       return entries

Generator/Coroutine Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions are checked for generator/coroutine flags (line 77):

.. code-block:: python

   _CO_GENERATOR = 0x20
   _CO_COROUTINE = 0x80
   _CO_ASYNC_GENERATOR = 0x200

   def _is_generator_or_coroutine(func):
       flags = func.__code__.co_flags
       return bool(flags & (_CO_GENERATOR | _CO_COROUTINE | _CO_ASYNC_GENERATOR))

When detected, ``_create_generator_wrapper()`` or ``_create_coroutine_wrapper()`` is used instead of the regular wrapper.

The dump_ir Function
^^^^^^^^^^^^^^^^^^^^

The ``dump_ir()`` function (line 863) retrieves LLVM IR for inspection:

.. code-block:: python

   def dump_ir(func):
       """Dump the LLVM IR for a JIT-compiled function."""
       jit_instance = func._jit_instance
       jit_instance.set_dump_ir(True)
       # Recompile with unique name to capture IR
       ir = jit_instance.get_last_ir()
       jit_instance.set_dump_ir(False)
       return ir

JITCore Class
-------------

The ``JITCore`` class (defined in ``jit_core.h``) is the central component:

.. code-block:: cpp

   class JITCore {
   public:
       JITCore();                              // Initialize LLVM ORC JIT
       ~JITCore();                             // Release Python references

       // Compilation methods (one per mode)
       bool compile_function(...);             // Object mode (5,700 lines)
       bool compile_int_function(...);         // Int mode (938 lines)
       bool compile_float_function(...);       // Float mode (723 lines)
       bool compile_bool_function(...);        // Bool mode (390 lines)
       bool compile_int32_function(...);
       bool compile_float32_function(...);
       bool compile_complex128_function(...);
       bool compile_complex64_function(...);
       bool compile_ptr_function(...);
       bool compile_vec4f_function(...);
       bool compile_vec8i_function(...);
       bool compile_optional_f64_function(...);
       bool compile_generator(...);            // Generator state machine (3,380 lines)

       // Callable creation
       nb::object get_callable(...);
       nb::object get_int_callable(...);
       // ... one for each mode

   private:
       std::unique_ptr<llvm::orc::LLJIT> jit;  // LLVM ORC JIT engine
       std::unique_ptr<llvm::LLVMContext> context;
       std::vector<PyObject*> stored_constants; // Python refs for cleanup
       // ...
   };

Constructor Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``JITCore`` is constructed:

1. Initialize LLVM native target (x86, ARM, etc.)
2. Create LLJIT instance via ``LLJITBuilder``
3. Register C helper functions as absolute symbols:

   - ``jit_call_with_kwargs`` - Handles keyword arguments
   - ``jit_xincref`` / ``jit_xdecref`` - NULL-safe reference counting
   - ``JITGetAwaitable`` - Async/await support
   - ``JITMatchKeys`` / ``JITMatchClass`` - Pattern matching support
   - ``jit_unbox_int`` / ``jit_box_int`` - Type conversions

Compilation Pipeline
--------------------

Object Mode (compile_function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main ``compile_function`` method handles full Python semantics:

**Step 1: Parse Instructions**

.. code-block:: cpp

   std::vector<Instruction> instructions;
   for (auto item : py_instructions) {
       Instruction instr;
       instr.opcode = nb::cast<uint16_t>(item["opcode"]);
       instr.arg = nb::cast<uint16_t>(item["arg"]);
       instr.argval = nb::cast<int32_t>(item["argval"]);
       instr.offset = nb::cast<uint16_t>(item["offset"]);
       instructions.push_back(instr);
   }

**Step 2: Build Control Flow Graph**

Three helper functions analyze bytecode:

- ``find_block_starts()`` - Identify basic block boundaries
- ``build_cfg()`` - Construct predecessor/successor relationships
- ``compute_stack_depths()`` - Dataflow analysis for stack depths

.. code-block:: cpp

   std::set<int> block_starts = find_block_starts(instructions, exception_table);
   std::map<int, BasicBlockInfo> cfg = build_cfg(instructions, exception_table, block_starts);
   compute_stack_depths(cfg, instructions);

**Step 3: Create LLVM IR**

For each bytecode instruction, generate corresponding LLVM IR:

.. code-block:: cpp

   switch (opcode) {
       case LOAD_CONST: {
           PyObject* const_obj = constants[arg];
           llvm::Value* ptr = builder.CreateIntToPtr(
               llvm::ConstantInt::get(i64_type, (uint64_t)const_obj),
               pyobj_ptr_type
           );
           stack.push_back(ptr);
           break;
       }
       case BINARY_OP: {
           llvm::Value* rhs = stack.back(); stack.pop_back();
           llvm::Value* lhs = stack.back(); stack.pop_back();
           // Call PyNumber_Add, PyNumber_Subtract, etc.
           llvm::Value* result = builder.CreateCall(py_number_add_func, {lhs, rhs});
           stack.push_back(result);
           break;
       }
       // ... 100+ opcode handlers
   }

**Step 4: Optimize and Compile**

.. code-block:: cpp

   // Run LLVM optimization passes
   optimize_module(module, func);

   // Add module to JIT
   auto tsm = llvm::orc::ThreadSafeModule(std::move(module), std::move(context));
   jit->addIRModule(std::move(tsm));

Native Mode Compilation
^^^^^^^^^^^^^^^^^^^^^^^

Native modes generate simpler IR without Python API calls. Each mode has a dedicated ``compile_*_function()`` method:

**Int Mode** (``compile_int_function``, lines 8045-8983):

Generates native ``i64`` arithmetic. Features:

- Native range loop detection and optimization
- Division-by-zero checking with branch to fallback
- Power operation via ``llvm.powi.i64``

.. code-block:: cpp

   // Range loop optimization in compile_int_function
   struct RangeLoop {
       llvm::Value* current;   // Loop variable (i64)
       llvm::Value* stop;      // End value
       llvm::Value* step;      // Step (default 1)
       llvm::BasicBlock* body_block;
       llvm::BasicBlock* exit_block;
   };
   std::map<int, RangeLoop> range_loops;

**Float Mode** (``compile_float_function``, lines 8985-9756):

Generates native ``double`` operations:

.. code-block:: cpp

   case BINARY_OP: {
       llvm::Value* rhs = stack.back(); stack.pop_back();
       llvm::Value* lhs = stack.back(); stack.pop_back();
       switch (op_arg) {
           case 0: result = builder.CreateFAdd(lhs, rhs); break;  // +
           case 1: result = builder.CreateFSub(lhs, rhs); break;  // -
           case 5: result = builder.CreateFMul(lhs, rhs); break;  // *
           case 11: result = builder.CreateFDiv(lhs, rhs); break; // /
           case 2: {  // ** power
               // Floor division: call llvm.floor after division
               llvm::Value* div = builder.CreateFDiv(lhs, rhs);
               result = builder.CreateCall(floor_func, {div});
               break;
           }
       }
   }

**SIMD Modes** (vec4f, vec8i):

SIMD modes use pointer-based ABI for Windows x64 compatibility:

.. code-block:: cpp

   // compile_vec4f_function (lines 11158-11289)
   // Signature: void fn(float* out, float* a, float* b)
   llvm::Type* vec4f_type = llvm::FixedVectorType::get(
       builder.getFloatTy(), 4  // <4 x float> for SSE
   );

   // Load with 16-byte alignment for SSE
   llvm::Value* vec_a = builder.CreateAlignedLoad(
       vec4f_type, a_ptr, llvm::Align(16)
   );

   // Arithmetic becomes vector operations
   llvm::Value* result = builder.CreateFAdd(vec_a, vec_b);

   // Store result with alignment
   builder.CreateAlignedStore(result, out_ptr, llvm::Align(16));

For AVX (vec8i mode, lines 11291-11418):

.. code-block:: cpp

   // <8 x i32> for AVX with 32-byte alignment
   llvm::Type* vec8i_type = llvm::FixedVectorType::get(
       builder.getInt32Ty(), 8
   );
   llvm::Value* vec = builder.CreateAlignedLoad(
       vec8i_type, ptr, llvm::Align(32)
   );

**Complex Modes** (complex128, complex64):

Complex numbers use struct types:

.. code-block:: cpp

   // compile_complex128_function (lines 10352-10600)
   // {double, double} struct for complex numbers
   llvm::StructType* complex_type = llvm::StructType::get(
       context, {builder.getDoubleTy(), builder.getDoubleTy()}
   );

   // Helper lambdas for component access
   auto extract_real = [&](llvm::Value* c) {
       return builder.CreateExtractValue(c, {0}, "real");
   };
   auto extract_imag = [&](llvm::Value* c) {
       return builder.CreateExtractValue(c, {1}, "imag");
   };
   auto make_complex = [&](llvm::Value* r, llvm::Value* i) {
       llvm::Value* c = llvm::UndefValue::get(complex_type);
       c = builder.CreateInsertValue(c, r, {0});
       c = builder.CreateInsertValue(c, i, {1});
       return c;
   };

   // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
   llvm::Value* a = extract_real(lhs), *b = extract_imag(lhs);
   llvm::Value* c = extract_real(rhs), *d = extract_imag(rhs);
   llvm::Value* ac = builder.CreateFMul(a, c);
   llvm::Value* bd = builder.CreateFMul(b, d);
   llvm::Value* ad = builder.CreateFMul(a, d);
   llvm::Value* bc = builder.CreateFMul(b, c);
   result = make_complex(
       builder.CreateFSub(ac, bd),
       builder.CreateFAdd(ad, bc)
   );

**Optional Float Mode** (optional_f64):

For nullable floats, uses a {has_value, value} struct:

.. code-block:: cpp

   // compile_optional_f64_function (lines 10760-10900)
   llvm::StructType* opt_type = llvm::StructType::get(
       context, {builder.getInt64Ty(), builder.getDoubleTy()}
   );

   // Check has_value before operations
   llvm::Value* has_val = builder.CreateExtractValue(arg, {0});
   llvm::Value* is_some = builder.CreateICmpNE(
       has_val, llvm::ConstantInt::get(i64_type, 0)
   );
   builder.CreateCondBr(is_some, compute_block, none_block);

Type System
-----------

JustJIT's type system is defined in ``type_system.h``:

.. code-block:: cpp

   enum class JITType : uint8_t {
       OBJECT = 0,      // PyObject* (full Python)
       INT64 = 1,       // i64
       FLOAT64 = 2,     // f64 (double)
       BOOL = 3,        // i1
       UINT64 = 4,      // u64
       INT32 = 5,       // i32
       FLOAT32 = 6,     // f32 (float)
       COMPLEX128 = 7,  // {f64, f64}
       PTR_F64 = 8,     // ptr to double array
       VEC4F = 9,       // <4 x f32> (SSE)
       VEC8I = 10,      // <8 x i32> (AVX)
       COMPLEX64 = 11,  // {f32, f32}
       OPTIONAL_F64 = 12, // {i64, f64} nullable
   };

LLVM Type Mapping
^^^^^^^^^^^^^^^^^

Each JITType maps to an LLVM type:

.. list-table::
   :header-rows: 1

   * - JITType
     - LLVM Type
     - C++ Type
   * - INT64
     - ``i64``
     - ``int64_t``
   * - FLOAT64
     - ``double``
     - ``double``
   * - COMPLEX128
     - ``{double, double}``
     - ``struct { double real; double imag; }``
   * - VEC4F
     - ``<4 x float>``
     - ``float[4]``
   * - OPTIONAL_F64
     - ``{i64, f64}``
     - ``struct { int64_t has_value; double value; }``

Generator Compilation
---------------------

Generators are compiled as state machines (``compile_generator``, 3,380 lines):

**State Machine Model**

.. code-block:: text

   State values:
     0     = Initial (not started)
     1..N  = Suspended at yield N
     -1    = Completed (returned)
     -2    = Error

**Step Function Signature**

.. code-block:: cpp

   typedef PyObject* (*GeneratorStepFunc)(
       int32_t* state,        // Current state, modified by function
       PyObject** locals,     // Array of local variables
       PyObject* sent_value   // Value from send()
   );

**YIELD_VALUE Handling**

Each ``YIELD_VALUE`` becomes a state transition:

1. Save all local variables to ``locals`` array
2. Set ``*state`` to the yield number
3. Return the yielded value

**Resume Handling**

On resume, the step function:

1. Read ``*state`` to determine resume point
2. Restore locals from the ``locals`` array
3. Jump to the appropriate point in the code

JIT Generator Object
^^^^^^^^^^^^^^^^^^^^

The ``JITGeneratorObject`` is a Python type:

.. code-block:: cpp

   struct JITGeneratorObject {
       PyObject_HEAD
       int32_t state;              // Current state
       PyObject** locals;          // Preserved local variables
       Py_ssize_t num_locals;      // Size of locals array
       GeneratorStepFunc step_func; // Compiled step function
       PyObject* name;             // For repr()
       PyObject* qualname;
   };

It implements the iterator protocol (``__iter__``, ``__next__``) and generator methods (``send``, ``throw``, ``close``).

Python C API Integration
------------------------

JustJIT calls Python C API functions for object mode operations. These are declared in ``declare_python_api_functions()`` (484 lines):

**Arithmetic Operations**

- ``PyNumber_Add``, ``PyNumber_Subtract``, ``PyNumber_Multiply``
- ``PyNumber_TrueDivide``, ``PyNumber_FloorDivide``, ``PyNumber_Remainder``
- ``PyNumber_Power``, ``PyNumber_Negative``, ``PyNumber_Positive``

**Container Operations**

- ``PyList_New``, ``PyList_SetItem``, ``PyList_Append``
- ``PyDict_New``, ``PyDict_SetItem``, ``PyDict_GetItem``
- ``PyTuple_New``, ``PyTuple_SetItem``, ``PyTuple_GetItem``
- ``PySet_New``, ``PySet_Add``

**Object Operations**

- ``PyObject_GetAttr``, ``PyObject_SetAttr``
- ``PyObject_GetItem``, ``PyObject_SetItem``
- ``PyObject_Call``, ``PyObject_GetIter``
- ``PyObject_RichCompareBool``, ``PyObject_IsTrue``

**Exception Handling**

- ``PyErr_Occurred``, ``PyErr_Fetch``, ``PyErr_Restore``
- ``PyExc_StopIteration``, ``PyErr_SetObject``

ABI Considerations
------------------

Windows x64 ABI
^^^^^^^^^^^^^^^

Windows x64 has specific requirements for struct returns. JustJIT uses pointer-based ABI for complex types:

.. code-block:: cpp

   // Instead of returning struct by value:
   // Complex128 fn(Complex128 a, Complex128 b);

   // Use pointer-based ABI:
   void fn(Complex128* out, Complex128* a, Complex128* b);

This applies to:

- ``complex128``, ``complex64`` - Complex numbers
- ``optional_f64`` - Nullable floats
- ``vec4f``, ``vec8i`` - SIMD vectors

Callable Wrappers
^^^^^^^^^^^^^^^^^

Each mode has callable wrappers that convert Python objects to native types:

.. code-block:: cpp

   // Int mode callable for 2 arguments
   nb::object create_int_callable_2(uint64_t func_ptr) {
       auto fn_ptr = reinterpret_cast<int64_t(*)(int64_t, int64_t)>(func_ptr);
       return nb::cpp_function([fn_ptr](nb::object a, nb::object b) {
           int64_t arg0 = nb::cast<int64_t>(a);
           int64_t arg1 = nb::cast<int64_t>(b);
           return fn_ptr(arg0, arg1);
       });
   }

LLVM Optimization
-----------------

JustJIT uses LLVM's new pass manager for optimization:

.. code-block:: cpp

   void JITCore::optimize_module(llvm::Module &module, llvm::Function *func) {
       llvm::PassBuilder pb;
       llvm::LoopAnalysisManager lam;
       llvm::FunctionAnalysisManager fam;
       llvm::CGSCCAnalysisManager cgam;
       llvm::ModuleAnalysisManager mam;

       pb.registerModuleAnalyses(mam);
       pb.registerCGSCCAnalyses(cgam);
       pb.registerFunctionAnalyses(fam);
       pb.registerLoopAnalyses(lam);
       pb.crossRegisterProxies(lam, fam, cgam, mam);

       // Select optimization level
       llvm::OptimizationLevel level;
       switch (opt_level) {
           case 0: level = llvm::OptimizationLevel::O0; break;
           case 1: level = llvm::OptimizationLevel::O1; break;
           case 2: level = llvm::OptimizationLevel::O2; break;
           default: level = llvm::OptimizationLevel::O3; break;
       }

       auto mpm = pb.buildPerModuleDefaultPipeline(level);
       mpm.run(module, mam);
   }

Supported Opcodes
-----------------

JustJIT supports nearly all Python 3.13 opcodes. Based on ``_is_simple_generator()`` and ``_is_simple_coroutine()`` in ``__init__.py``:

**Stack Operations**

- ``LOAD_CONST``, ``LOAD_FAST``, ``LOAD_FAST_CHECK``, ``STORE_FAST``, ``POP_TOP``, ``SWAP``, ``COPY``
- ``LOAD_FAST_LOAD_FAST``, ``STORE_FAST_STORE_FAST``, ``STORE_FAST_LOAD_FAST`` (Python 3.13 combined opcodes)
- ``LOAD_FAST_AND_CLEAR``, ``PUSH_NULL``, ``NOP``, ``CACHE``, ``RESUME``

**Binary/Unary Operations**

- ``BINARY_OP`` (all 15 operators: +, -, *, /, //, %, **, @, <<, >>, &, |, ^, and in-place variants)
- ``UNARY_NEGATIVE``, ``UNARY_NOT``, ``UNARY_INVERT``
- ``TO_BOOL``

**Comparison and Containment**

- ``COMPARE_OP`` (all 6 comparison operators)
- ``IS_OP``, ``CONTAINS_OP``

**Control Flow**

- ``JUMP_FORWARD``, ``JUMP_BACKWARD``, ``JUMP_BACKWARD_NO_INTERRUPT``
- ``POP_JUMP_IF_TRUE``, ``POP_JUMP_IF_FALSE``
- ``POP_JUMP_IF_NONE``, ``POP_JUMP_IF_NOT_NONE``

**Loops and Iteration**

- ``FOR_ITER``, ``GET_ITER``, ``END_FOR``

**Functions and Calls**

- ``CALL``, ``CALL_KW``, ``CALL_FUNCTION_EX``
- ``RETURN_VALUE``, ``RETURN_CONST``
- ``MAKE_FUNCTION``, ``SET_FUNCTION_ATTRIBUTE``
- ``CALL_INTRINSIC_1``, ``CALL_INTRINSIC_2`` (all intrinsics including TypeVar, ParamSpec, TypeVarTuple, TypeAlias)

**Containers**

- ``BUILD_LIST``, ``BUILD_TUPLE``, ``BUILD_MAP``, ``BUILD_SET``, ``BUILD_STRING``, ``BUILD_CONST_KEY_MAP``
- ``LIST_APPEND``, ``LIST_EXTEND``, ``SET_ADD``, ``SET_UPDATE``, ``MAP_ADD``, ``DICT_UPDATE``, ``DICT_MERGE``
- ``UNPACK_SEQUENCE``, ``UNPACK_EX``

**Subscript and Slicing**

- ``BINARY_SUBSCR``, ``STORE_SUBSCR``, ``DELETE_SUBSCR``
- ``BUILD_SLICE``, ``BINARY_SLICE``, ``STORE_SLICE``

**Attributes and Globals**

- ``LOAD_ATTR``, ``STORE_ATTR``, ``LOAD_GLOBAL``, ``STORE_GLOBAL``, ``LOAD_NAME``
- ``LOAD_LOCALS``, ``LOAD_FROM_DICT_OR_DEREF``, ``LOAD_FROM_DICT_OR_GLOBALS`` (Python 3.13 annotation scopes)

**Closures**

- ``LOAD_DEREF``, ``STORE_DEREF``, ``LOAD_CLOSURE``, ``MAKE_CELL``, ``COPY_FREE_VARS``

**Pattern Matching** (Python 3.10+)

- ``MATCH_SEQUENCE``, ``MATCH_MAPPING``, ``MATCH_CLASS``, ``MATCH_KEYS``

**Exception Handling**

- ``PUSH_EXC_INFO``, ``POP_EXCEPT``, ``CHECK_EXC_MATCH``, ``CHECK_EG_MATCH``
- ``RAISE_VARARGS``, ``RERAISE``
- ``CLEANUP_THROW`` (exception handling during throw()/close())

**With Statements and Context Managers**

- ``BEFORE_WITH``, ``BEFORE_ASYNC_WITH``, ``WITH_EXCEPT_START``

**Class Support**

- ``SETUP_ANNOTATIONS``, ``EXIT_INIT_CHECK``

**Imports**

- ``IMPORT_NAME``, ``IMPORT_FROM``

**Generator/Coroutine**

- ``YIELD_VALUE``, ``RETURN_GENERATOR``
- ``GET_YIELD_FROM_ITER``, ``SEND``, ``END_SEND``
- ``GET_AWAITABLE`` (async/await support)

**Not Yet Supported**

- Async generators (``async def`` with ``yield``) - requires ``GET_AITER``, ``GET_ANEXT``, ``END_ASYNC_FOR``

Building from Source
--------------------

Requirements:

- LLVM 18+ development files
- CMake 3.20+
- Python 3.13 development headers
- C++17 compatible compiler

.. code-block:: bash

   # Clone repository
   git clone https://github.com/magi8101/JustJIT.git
   cd JustJIT

   # Set LLVM_DIR
   export LLVM_DIR=/path/to/llvm/lib/cmake/llvm

   # Build
   pip install .

The build uses scikit-build-core and CMake to compile the C++ extension.
