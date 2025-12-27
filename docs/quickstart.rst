Quick Start
===========

This guide covers the basics of using JustJIT to accelerate your Python code.

Basic Usage
-----------

The simplest way to use JustJIT is with the ``@jit`` decorator:

.. code-block:: python

   import justjit

   @justjit.jit
   def add(a, b):
       return a + b

   result = add(1, 2)  # Compiled to native code on first call

When you call ``add(1, 2)``, JustJIT:

1. Extracts the Python bytecode from the function
2. Translates it to LLVM IR
3. Compiles to native machine code
4. Executes the native code

Subsequent calls to ``add`` skip the compilation step and run the cached native code directly.

Choosing a Mode
---------------

JustJIT supports 12 native modes. Each mode compiles your function to work with a specific data type, eliminating Python object overhead.

Integer Mode
^^^^^^^^^^^^

For pure integer arithmetic:

.. code-block:: python

   @justjit.jit(mode='int')
   def factorial(n):
       result = 1
       for i in range(2, n + 1):
           result = result * i
       return result

   factorial(20)  # Returns 2432902008176640000

Float Mode
^^^^^^^^^^

For floating-point arithmetic:

.. code-block:: python

   @justjit.jit(mode='float')
   def average(a, b):
       return (a + b) / 2.0

   average(3.0, 5.0)  # Returns 4.0

Auto Mode (Default)
^^^^^^^^^^^^^^^^^^^

When using ``mode='auto'`` (the default), JustJIT uses the full Python object mode. This is the most compatible but has more overhead:

.. code-block:: python

   @justjit.jit  # mode='auto' is the default
   def flexible_add(a, b):
       return a + b

   flexible_add(1, 2)          # Works with integers
   flexible_add(1.5, 2.5)      # Works with floats
   flexible_add("hello", " ")  # Works with strings

See :doc:`modes` for a complete list of available modes.

Inspecting Generated IR
-----------------------

Use ``dump_ir()`` to see the LLVM IR generated for a function:

.. code-block:: python

   import justjit

   @justjit.jit(mode='float')
   def add(a, b):
       return a + b

   # Trigger compilation
   add(1.0, 2.0)

   # Print the LLVM IR
   print(justjit.dump_ir(add))

Output:

.. code-block:: llvm

   define double @add(double %0, double %1) {
   entry:
     %fadd = fadd double %0, %1
     ret double %fadd
   }

This shows that the Python function compiles to a single ``fadd`` instruction.

Optimization Levels
-------------------

Control the LLVM optimization level with ``opt_level``:

.. code-block:: python

   @justjit.jit(opt_level=0)  # No optimization (fastest compile)
   def debug_function(a, b):
       return a + b

   @justjit.jit(opt_level=3)  # Maximum optimization (default)
   def fast_function(a, b):
       return a + b

- ``opt_level=0``: No optimization, fastest compilation
- ``opt_level=1``: Basic optimization
- ``opt_level=2``: More optimization
- ``opt_level=3``: Maximum optimization (default)

Loop Optimization
-----------------

JustJIT excels at optimizing loops. Native ``for i in range(n)`` loops compile to tight machine code:

.. code-block:: python

   @justjit.jit(mode='int')
   def sum_range(n):
       total = 0
       for i in range(n):
           total = total + i
       return total

   # This runs 38,000x faster than the Python interpreter
   sum_range(10_000_000)

The generated code avoids:

- Python object allocation
- Type checking on each iteration
- Interpreter dispatch overhead

Generator Support
-----------------

JustJIT can compile generator functions to native state machines:

.. code-block:: python

   @justjit.jit
   def countdown(n):
       while n > 0:
           yield n
           n = n - 1

   for value in countdown(5):
       print(value)  # Prints 5, 4, 3, 2, 1

Generators are compiled as step functions with state persistence across yields. The ``send()`` and ``throw()`` methods are fully supported.

Async Function Support
----------------------

Async functions (coroutines) are also supported:

.. code-block:: python

   import asyncio

   @justjit.jit
   async def async_add(a, b):
       await asyncio.sleep(0.1)
       return a + b

   asyncio.run(async_add(1, 2))  # Returns 3

The JIT compiler handles ``await``, delegating to awaited objects and properly extracting return values from ``StopIteration``.

Next Steps
----------

- :doc:`modes` - Learn about all 11 native modes
- :doc:`api` - Full API reference
- :doc:`performance` - Benchmarks and optimization tips
- :doc:`async` - Deep dive into generator/coroutine compilation
