JustJIT Documentation
=====================

**JustJIT** is a Python JIT compiler that uses LLVM ORC to compile Python bytecode directly to native machine code.

.. code-block:: python

   import justjit

   @justjit.jit(mode='int')
   def sum_loop(n):
       total = 0
       for i in range(n):
           total = total + i
       return total

   # 38,000x faster than CPython
   sum_loop(10_000_000)

Features
--------

- Compiles Python bytecode to LLVM IR, then to native machine code
- **12 native compilation modes** for maximum performance (int, float, bool, complex, SIMD, etc.)
- Generator and async function support via state machine compilation
- No interpreter overhead for numeric loops
- Near-complete Python 3.13 opcode coverage (75+ opcodes)
- Simple decorator-based API
- Cross-platform support (Windows x64, macOS arm64, Linux x64)

Quick Start
-----------

Install from PyPI:

.. code-block:: bash

   pip install justjit

Use the ``@jit`` decorator:

.. code-block:: python

   import justjit

   @justjit.jit(mode='float')
   def add(a, b):
       return a + b

   result = add(3.0, 4.0)  # Runs as native code

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   modes

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   performance

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   internals
   async
   cfg

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
