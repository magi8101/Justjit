Compilation Modes
=================

JustJIT supports 12 native compilation modes. Each mode compiles Python functions to work with a specific LLVM type, eliminating Python object overhead.

Mode Summary
------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Mode
     - LLVM Type
     - Description
   * - ``auto`` / ``object``
     - PyObject*
     - Full Python semantics. Default mode, most compatible.
   * - ``int``
     - i64
     - 64-bit signed integer. Best for integer math and loops.
   * - ``float``
     - f64
     - 64-bit floating point. Best for floating-point arithmetic.
   * - ``bool``
     - i1
     - Boolean values. Best for boolean logic functions.
   * - ``int32``
     - i32
     - 32-bit signed integer. C interop and memory efficiency.
   * - ``float32``
     - f32
     - 32-bit floating point. SIMD preparation and ML.
   * - ``complex128``
     - {f64, f64}
     - Double-precision complex number (real, imag).
   * - ``complex64``
     - {f32, f32}
     - Single-precision complex number.
   * - ``ptr``
     - ptr
     - Pointer to f64 array. Direct array access.
   * - ``vec4f``
     - <4 x f32>
     - SSE SIMD vector (4 floats).
   * - ``vec8i``
     - <8 x i32>
     - AVX SIMD vector (8 integers).
   * - ``optional_f64``
     - {i64, f64}
     - Nullable float64 with None handling.

Object Mode (auto)
------------------

The ``auto`` mode (also called ``object`` mode) is the default. It compiles functions while preserving full Python semantics:

.. code-block:: python

   @justjit.jit  # mode='auto' is the default
   def flexible_add(a, b):
       return a + b

   flexible_add(1, 2)          # Works with integers
   flexible_add(1.5, 2.5)      # Works with floats
   flexible_add("hello", " ")  # Works with strings

This mode generates LLVM IR that calls Python C API functions (``PyNumber_Add``, ``PyObject_GetAttr``, etc.), so it maintains full compatibility with any Python type.

Supported operations:

- All Python operations via C API calls
- Exception handling (try/except/finally)
- Pattern matching (match/case)
- Context managers (with statements)
- Closures and nested functions
- Generators and async functions

Integer Mode (int)
------------------

The ``int`` mode compiles functions to use native 64-bit integers.

.. code-block:: python

   @justjit.jit(mode='int')
   def fibonacci(n):
       a, b = 0, 1
       for _ in range(n):
           a, b = b, a + b
       return a

   fibonacci(50)  # Returns 12586269025

Supported operations:

- Arithmetic: ``+``, ``-``, ``*``, ``//``, ``%``, ``**``
- Comparison: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
- Bitwise: ``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``
- Range loops: ``for i in range(n)``

LLVM IR:

.. code-block:: llvm

   define i64 @fibonacci(i64 %n) {
     ; Native integer operations
     %add = add i64 %a, %b
     ret i64 %result
   }

Float Mode (float)
------------------

The ``float`` mode compiles functions to use native 64-bit floating point.

.. code-block:: python

   @justjit.jit(mode='float')
   def distance(x1, y1, x2, y2):
       dx = x2 - x1
       dy = y2 - y1
       return (dx * dx + dy * dy) ** 0.5

   distance(0.0, 0.0, 3.0, 4.0)  # Returns 5.0

Supported operations:

- Arithmetic: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
- Comparison: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
- Range loops: ``for i in range(n)``

LLVM IR:

.. code-block:: llvm

   define double @distance(double %x1, double %y1, double %x2, double %y2) {
     %dx = fsub double %x2, %x1
     %dy = fsub double %y2, %y1
     %fadd = fadd double %dx_sq, %dy_sq
     ret double %sqrt_result
   }

Bool Mode (bool)
----------------

The ``bool`` mode compiles functions for boolean logic.

.. code-block:: python

   @justjit.jit(mode='bool')
   def is_valid(a, b):
       return a and not b

   is_valid(True, False)  # Returns True

Supported operations:

- Logical: ``and``, ``or``, ``not``
- Comparison: ``==``, ``!=``

Complex128 Mode (complex128)
----------------------------

The ``complex128`` mode handles double-precision complex numbers.

.. code-block:: python

   @justjit.jit(mode='complex128')
   def mandelbrot_step(z, c):
       return z * z + c

   mandelbrot_step(1+2j, 0.5+0.5j)

The complex number is stored as a struct ``{double real, double imag}``.

Supported operations:

- Arithmetic: ``+``, ``-``, ``*``, ``/``

Complex64 Mode (complex64)
--------------------------

The ``complex64`` mode uses single-precision complex numbers for memory efficiency.

.. code-block:: python

   @justjit.jit(mode='complex64')
   def complex_multiply(a, b):
       return a * b

   complex_multiply(3+4j, 1+2j)  # Returns (-5+10j)

Same operations as ``complex128`` but with 32-bit floats.

Pointer Mode (ptr)
------------------

The ``ptr`` mode enables direct array access via pointers.

.. code-block:: python

   import ctypes

   @justjit.jit(mode='ptr')
   def array_sum(arr, length):
       # arr is a pointer, length is count
       total = 0.0
       for i in range(length):
           total = total + arr[i]
       return total

   # Create a C-compatible array
   data = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
   ptr = ctypes.addressof(data)
   array_sum(ptr, 4)  # Returns 10.0

This mode is useful for NumPy interop and high-performance array operations.

Vec4f Mode (vec4f)
------------------

The ``vec4f`` mode uses SSE SIMD operations on 4 floats simultaneously.

.. code-block:: python

   @justjit.jit(mode='vec4f')
   def vec_add(a, b):
       return a + b

   # Operations on 4 floats at once
   # Input: two <4 x float> vectors
   # Output: <4 x float> sum

**Pointer-Based ABI:**

SIMD modes use a pointer-based ABI for Windows x64 compatibility:

.. code-block:: cpp

   // Actual signature: void fn(float* out, float* a, float* b)
   // Instead of: <4 x float> fn(<4 x float> a, <4 x float> b)

The callable wrapper handles this transparently.

LLVM IR (internal):

.. code-block:: llvm

   define void @vec_add(ptr %out, ptr %a, ptr %b) {
     %vec_a = load <4 x float>, ptr %a, align 16  ; 16-byte alignment for SSE
     %vec_b = load <4 x float>, ptr %b, align 16
     %result = fadd <4 x float> %vec_a, %vec_b
     store <4 x float> %result, ptr %out, align 16
     ret void
   }

Vec8i Mode (vec8i)
------------------

The ``vec8i`` mode uses AVX SIMD operations on 8 integers simultaneously.

.. code-block:: python

   @justjit.jit(mode='vec8i')
   def vec_mul(a, b):
       return a * b

   # Operations on 8 i32 values at once

**Pointer-Based ABI:**

Like vec4f, uses pointer-based ABI:

.. code-block:: cpp

   // Actual signature: void fn(int32_t* out, int32_t* a, int32_t* b)

LLVM IR (internal):

.. code-block:: llvm

   define void @vec_mul(ptr %out, ptr %a, ptr %b) {
     %vec_a = load <8 x i32>, ptr %a, align 32  ; 32-byte alignment for AVX
     %vec_b = load <8 x i32>, ptr %b, align 32
     %result = mul <8 x i32> %vec_a, %vec_b
     store <8 x i32> %result, ptr %out, align 32
     ret void
   }

Optional_f64 Mode (optional_f64)
--------------------------------

The ``optional_f64`` mode handles nullable float values with None support.

.. code-block:: python

   @justjit.jit(mode='optional_f64')
   def safe_divide(a, b):
       if b == 0:
           return None
       return a / b

   safe_divide(10.0, 2.0)  # Returns 5.0
   safe_divide(10.0, 0.0)  # Returns None

The nullable value is stored as a struct ``{i64 has_value, double value}``.

None propagation:

- If any operand is None, the result is None
- Binary operations check both operands for None before computing

LLVM IR:

.. code-block:: llvm

   ; Struct: {i64 has_value, double value}
   define void @safe_divide(ptr %out, ptr %a, ptr %b) {
     %a_val = load {i64, double}, ptr %a
     %b_val = load {i64, double}, ptr %b
     %a_has = extractvalue {i64, double} %a_val, 0
     %b_has = extractvalue {i64, double} %b_val, 0
     %both_have = and i64 %a_has, %b_has
     ; ... compute if both have values
     store {i64, double} %result, ptr %out
     ret void
   }

Int32 and Float32 Modes
-----------------------

The ``int32`` and ``float32`` modes use 32-bit values for C interop and memory efficiency:

.. code-block:: python

   @justjit.jit(mode='int32')
   def small_add(a, b):
       return a + b

   @justjit.jit(mode='float32')
   def ml_operation(a, b):
       return a * b

These modes are useful when:

- Interfacing with C code that uses 32-bit types
- Working with ML frameworks that use float32
- Memory bandwidth is a bottleneck

Choosing the Right Mode
-----------------------

Use this decision tree:

1. **Working with integers?** Use ``int`` mode.
2. **Working with floats?** Use ``float`` mode.
3. **Working with complex numbers?** Use ``complex128`` or ``complex64``.
4. **Need None/nullable values?** Use ``optional_f64``.
5. **Working with arrays directly?** Use ``ptr`` mode.
6. **Need SIMD parallelism?** Use ``vec4f`` or ``vec8i``.
7. **C interop with 32-bit types?** Use ``int32`` or ``float32``.
8. **Mixed types or full Python semantics?** Use ``auto`` (default).

Performance tip: Native modes avoid Python object overhead entirely. For compute-heavy loops, the speedup can be 1,000x to 100,000x compared to the interpreter.
