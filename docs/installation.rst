Installation
============

Requirements
------------

- Python 3.13 or later
- Windows (x64), macOS (arm64), or Linux (x64)

Installation from PyPI
----------------------

The recommended way to install JustJIT is via pip:

.. code-block:: bash

   pip install justjit

The wheel comes with LLVM bundled, so no external dependencies are required.

Verify Installation
-------------------

After installation, verify that JustJIT works correctly:

.. code-block:: python

   import justjit

   @justjit.jit(mode='int')
   def test_add(a, b):
       return a + b

   result = test_add(1, 2)
   print(f"1 + 2 = {result}")  # Output: 1 + 2 = 3

If this runs without errors, JustJIT is installed correctly.

Building from Source
--------------------

To build JustJIT from source, you need:

- LLVM 18 development files
- CMake 3.20 or later
- Python 3.13 development headers
- A C++17 compatible compiler

Clone the repository:

.. code-block:: bash

   git clone https://github.com/magi8101/JustJIT.git
   cd JustJIT

Set up environment variables for LLVM:

.. code-block:: bash

   # Linux/macOS
   export LLVM_DIR=/path/to/llvm/lib/cmake/llvm

   # Windows (PowerShell)
   $env:LLVM_DIR = "C:\path\to\llvm\lib\cmake\llvm"

Build and install:

.. code-block:: bash

   pip install .

Platform-Specific Notes
-----------------------

Windows
^^^^^^^

JustJIT uses static LLVM linking on Windows. The wheel includes all necessary LLVM libraries.

If building from source on Windows, you may need vcpkg for zlib:

.. code-block:: powershell

   $env:ZLIB_ROOT = "C:\vcpkg\installed\x64-windows"

macOS
^^^^^

On macOS, JustJIT dynamically links against LLVM. The wheel bundles the LLVM shared library using delocate.

For building from source, install LLVM via Homebrew or conda-forge:

.. code-block:: bash

   # Using conda-forge
   conda install llvmdev=18.1.8

Linux
^^^^^

On Linux, JustJIT uses the manylinux_2_28 ABI for broad compatibility.

For building from source, install LLVM development packages:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt install llvm-18-dev

   # Fedora/RHEL
   sudo dnf install llvm-devel
