Performance
===========

This page covers performance characteristics, benchmarks, and optimization tips for JustJIT.

Benchmark Results
-----------------

The following benchmarks compare JustJIT to the standard CPython interpreter on an Intel Core i7-10700K.

Simple Function Calls
^^^^^^^^^^^^^^^^^^^^^

For trivial functions, JustJIT is slower due to the Python-to-native call overhead:

.. code-block:: text

   add(3.0, 4.0) x 1,000,000 calls:
     CPython:  102 ms
     JustJIT:  355 ms
     Speedup:  0.3x (slower)

The function itself runs faster, but crossing the Python/native boundary has overhead.

Loop-Intensive Code
^^^^^^^^^^^^^^^^^^^

For compute-heavy loops, JustJIT shows massive speedups:

.. code-block:: text

   sum_loop(10,000,000):
     CPython:  440 ms
     JustJIT:  0.01 ms
     Speedup:  44,000x

The entire loop runs in native code without returning to Python.

Why Loops Are Fast
------------------

Consider this loop:

.. code-block:: python

   @justjit.jit(mode='int')
   def sum_loop(n):
       total = 0
       for i in range(n):
           total = total + i
       return total

In CPython, each iteration involves:

1. Fetching the next bytecode instruction
2. Dispatching to the opcode handler
3. Creating integer objects for ``i`` and ``total``
4. Type-checking before the addition
5. Creating a new integer object for the result
6. Decrementing reference counts

In JustJIT, the loop compiles to:

.. code-block:: llvm

   range_body:
     %counter = load i64, ptr %range_counter
     %total = load i64, ptr %local_1
     %new_total = add i64 %total, %counter
     store i64 %new_total, ptr %local_1
     %next = add i64 %counter, 1
     store i64 %next, ptr %range_counter
     br label %range_header

This is a tight machine code loop with no Python overhead.

When to Use JustJIT
-------------------

JustJIT is ideal for:

- Numeric loops that iterate many times
- Mathematical computations with known types
- Hot paths that are called frequently
- Algorithms like sorting, searching, or matrix operations
- Generator functions with intensive computation
- Async functions with CPU-bound work between awaits

JustJIT is not ideal for:

- Functions that are called only once
- Code with many type variations
- Heavy use of Python objects (lists, dicts, classes)
- Async generators (not yet supported)

Optimization Tips
-----------------

Use Native Modes
^^^^^^^^^^^^^^^^

Specify a mode when you know the data type:

.. code-block:: python

   # Good: Native integers
   @justjit.jit(mode='int')
   def fast_factorial(n):
       result = 1
       for i in range(2, n + 1):
           result = result * i
       return result

   # Less optimal: Auto mode with type checks
   @justjit.jit
   def slow_factorial(n):
       result = 1
       for i in range(2, n + 1):
           result = result * i
       return result

Avoid Objects in Hot Loops
^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep computation inside the loop simple:

.. code-block:: python

   # Good: All native operations
   @justjit.jit(mode='float')
   def dot_product(n, ptr_a, ptr_b):
       total = 0.0
       for i in range(n):
           total = total + ptr_a[i] * ptr_b[i]
       return total

   # Bad: Creating objects in the loop
   @justjit.jit
   def dot_product_slow(a_list, b_list):
       total = 0.0
       for i in range(len(a_list)):  # len() is a function call
           total = total + a_list[i] * b_list[i]  # List indexing creates objects
       return total

Batch Work Inside JIT Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimize the number of JIT function calls:

.. code-block:: python

   # Good: One call, lots of work
   @justjit.jit(mode='int')
   def sum_batch(n):
       total = 0
       for i in range(n):
           total = total + i
       return total

   result = sum_batch(1_000_000)

   # Bad: Many calls, little work each
   @justjit.jit(mode='int')
   def add_one(x):
       return x + 1

   total = 0
   for i in range(1_000_000):  # This loop is in Python!
       total = add_one(total)

Memory Layout
-------------

For maximum performance with arrays:

1. Use contiguous memory (C arrays or NumPy with C order)
2. Use ``ptr`` mode for direct memory access
3. Avoid creating Python objects during iteration

.. code-block:: python

   import ctypes
   import numpy as np

   @justjit.jit(mode='ptr')
   def array_sum(ptr, n):
       total = 0.0
       for i in range(n):
           total = total + ptr[i]
       return total

   # NumPy interop
   arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
   ptr = arr.ctypes.data
   result = array_sum(ptr, len(arr))

Compilation Time
----------------

JustJIT compiles functions on first use. Compilation time depends on:

- Function complexity (number of bytecode instructions)
- LLVM optimization level

Typical compilation times:

.. code-block:: text

   Simple function (3-5 instructions):  ~5 ms
   Medium function (20-50 instructions): ~20 ms
   Complex function (100+ instructions): ~50 ms

To reduce startup latency, you can warm up functions:

.. code-block:: python

   @justjit.jit(mode='int')
   def my_function(n):
       # ...

   # Warm up during module load
   my_function(0)
