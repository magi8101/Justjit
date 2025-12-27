API Reference
=============

This page documents the public API of JustJIT.

jit
---

The main decorator for JIT-compiling Python functions.

.. py:function:: jit(func=None, *, opt_level=3, vectorize=True, inline=True, parallel=False, lazy=False, mode='auto')

   JIT compile a Python function for aggressive performance optimization.

   :param func: The function to compile. When using ``@jit`` without parentheses, this is the function being decorated.
   :type func: callable, optional
   :param opt_level: LLVM optimization level (0-3). Default is 3 for maximum performance.
   :type opt_level: int
   :param vectorize: Enable loop vectorization. Currently reserved for future use.
   :type vectorize: bool
   :param inline: Enable function inlining. Currently reserved for future use.
   :type inline: bool
   :param parallel: Enable parallelization. Currently reserved for future use.
   :type parallel: bool
   :param lazy: Delay compilation until first call. Currently reserved for future use.
   :type lazy: bool
   :param mode: Compilation mode. See :doc:`modes` for details.
   :type mode: str
   :returns: A JIT-compiled wrapper function.
   :rtype: callable

   **Available modes:**

   - ``'auto'`` - Full Python object mode (default)
   - ``'int'`` - 64-bit integer mode (i64)
   - ``'float'`` - 64-bit float mode (f64)
   - ``'bool'`` - Boolean mode (i1)
   - ``'int32'`` - 32-bit integer mode (i32)
   - ``'float32'`` - 32-bit float mode (f32)
   - ``'complex128'`` - Complex number mode ({f64, f64})
   - ``'complex64'`` - Single-precision complex ({f32, f32})
   - ``'ptr'`` - Pointer mode for array access
   - ``'vec4f'`` - SSE SIMD mode (<4 x f32>)
   - ``'vec8i'`` - AVX SIMD mode (<8 x i32>)
   - ``'optional_f64'`` - Nullable float64 ({i64, f64})

   **Usage without parentheses:**

   .. code-block:: python

      @justjit.jit
      def add(a, b):
          return a + b

   **Usage with parameters:**

   .. code-block:: python

      @justjit.jit(mode='int', opt_level=3)
      def multiply(a, b):
          return a * b

dump_ir
-------

Retrieve the LLVM IR generated for a JIT-compiled function.

.. py:function:: dump_ir(func)

   Dump the LLVM IR for a JIT-compiled function.

   :param func: A JIT-compiled function (decorated with ``@jit``).
   :type func: callable
   :returns: The LLVM IR as a string.
   :rtype: str
   :raises ValueError: If the function is not JIT-compiled.

   **Example:**

   .. code-block:: python

      import justjit

      @justjit.jit(mode='float')
      def add(a, b):
          return a + b

      # Trigger compilation
      add(1.0, 2.0)

      # Get the IR
      ir = justjit.dump_ir(add)
      print(ir)

   **Output:**

   .. code-block:: llvm

      define double @add(double %0, double %1) {
      entry:
        %fadd = fadd double %0, %1
        ret double %fadd
      }

JIT Class
---------

The low-level JIT compiler class. Most users should use the ``@jit`` decorator instead.

.. py:class:: JIT

   Low-level interface to the LLVM ORC JIT compiler.

   .. py:method:: __init__()

      Create a new JIT compiler instance.

   .. py:method:: set_opt_level(level)

      Set the LLVM optimization level.

      :param level: Optimization level (0-3).
      :type level: int

   .. py:method:: get_opt_level()

      Get the current LLVM optimization level.

      :returns: The optimization level.
      :rtype: int

   .. py:method:: set_dump_ir(dump)

      Enable or disable IR capture for debugging.

      :param dump: Whether to capture IR.
      :type dump: bool

   .. py:method:: get_last_ir()

      Get the LLVM IR from the last compiled function.

      :returns: The IR string, or empty string if not available.
      :rtype: str

   .. py:method:: compile(instructions, constants, names, globals_dict, builtins_dict, closure_cells, exception_table, name, param_count=2, total_locals=3, nlocals=3)

      Compile a function to native code using the full Python object mode.

      :param instructions: List of bytecode instruction dicts.
      :param constants: List of constant values.
      :param names: List of attribute/global names.
      :param globals_dict: Function's globals dictionary.
      :param builtins_dict: Builtins dictionary.
      :param closure_cells: List of closure cells.
      :param exception_table: Exception table entries.
      :param name: Function name.
      :param param_count: Number of parameters.
      :param total_locals: Total local variable slots.
      :param nlocals: Number of local variables.
      :returns: True if compilation succeeded.
      :rtype: bool

   .. py:method:: compile_int(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using integer mode.

   .. py:method:: compile_float(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using float mode.

   .. py:method:: compile_bool(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using bool mode.

   .. py:method:: compile_int32(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using int32 mode.

   .. py:method:: compile_float32(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using float32 mode.

   .. py:method:: compile_complex128(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using complex128 mode.

   .. py:method:: compile_complex64(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using complex64 mode.

   .. py:method:: compile_ptr(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using ptr mode.

   .. py:method:: compile_vec4f(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using vec4f mode.

   .. py:method:: compile_vec8i(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using vec8i mode.

   .. py:method:: compile_optional_f64(instructions, constants, name, param_count=2, total_locals=3)

      Compile a function to native code using optional_f64 mode.

   .. py:method:: compile_generator(instructions, constants, names, globals_dict, builtins_dict, closure_cells, exception_table, name, param_count, total_locals, nlocals)

      Compile a generator or async function to a state machine.

      :param instructions: List of bytecode instruction dicts.
      :param constants: List of constant values.
      :param names: List of attribute/global names.
      :param globals_dict: Function's globals dictionary.
      :param builtins_dict: Builtins dictionary.
      :param closure_cells: List of closure cells.
      :param exception_table: Exception table entries.
      :param name: Function name.
      :param param_count: Number of parameters.
      :param total_locals: Total local slots (locals + cells + freevars + stack).
      :param nlocals: Number of local variables.
      :returns: True if compilation succeeded.
      :rtype: bool

   .. py:method:: get_generator_callable(name, param_count, num_locals, gen_name, gen_qualname)

      Get metadata for creating generator/coroutine objects.

      :param name: Function name.
      :param param_count: Number of parameters.
      :param num_locals: Size of locals array.
      :param gen_name: Generator's __name__.
      :param gen_qualname: Generator's __qualname__.
      :returns: Dict with step_func_addr, num_locals, name, qualname.
      :rtype: dict

   .. py:method:: get_callable(name, param_count)

      Get a Python callable for a compiled function.

      :param name: Function name.
      :param param_count: Number of parameters.
      :returns: A callable that invokes the native function.
      :rtype: callable

Wrapper Function Attributes
---------------------------

Functions decorated with ``@jit`` have additional attributes:

.. py:attribute:: _jit_instance

   The underlying ``JIT`` instance used for compilation.

.. py:attribute:: _original_func

   The original Python function before decoration.

.. py:attribute:: _mode

   The compilation mode used ('int', 'float', 'auto', etc.).

.. py:attribute:: _instructions

   The bytecode instructions extracted from the function.
