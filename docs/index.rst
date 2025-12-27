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
- 11 native data types for maximum performance
- No interpreter overhead for numeric loops
- Simple decorator-based API
- Cross-platform support (Windows, macOS, Linux)

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
