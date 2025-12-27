Async and Generators
====================

JustJIT compiles Python generators and async functions to native state machines. This page explains how they work under the hood.

Overview
--------

When JustJIT encounters a generator function (one containing ``yield``), it compiles it differently from regular functions. Instead of a single execution, generators need to pause (yield) and resume multiple times.

JustJIT handles this by compiling the generator to a **step function** that implements a state machine. Each call to the step function executes until the next ``yield`` or ``return``.

The Step Function
-----------------

Generator step functions have this signature:

.. code-block:: cpp

   PyObject* step_func(int32_t* state, PyObject** locals, PyObject* sent_value);

**Parameters:**

- ``state``: Pointer to the current state integer, modified by the function
- ``locals``: Array of preserved local variables across yields
- ``sent_value``: Value passed via ``generator.send()``

**State Encoding:**

.. code-block:: text

   State    Meaning
   -----    -------
   0        Initial (generator not started, ignore sent_value)
   1..N     Suspended at yield point N
   -1       Completed (returned)
   -2       Error occurred

The implementation is defined at line 11443 in ``jit_core.cpp``:

.. code-block:: cpp

   bool JITCore::compile_generator(
       nb::list py_instructions, nb::list py_constants,
       nb::list py_names, nb::object py_globals_dict,
       nb::object py_builtins_dict, nb::list py_closure_cells,
       nb::list py_exception_table, const std::string &name,
       int param_count, int total_locals, int nlocals
   );

Stack Depth Simulation
----------------------

One challenge with generators is tracking the stack across yields. JustJIT simulates stack depth during compilation to know how many values need to be saved:

.. code-block:: cpp

   // First pass: simulate stack depth to track depth at each yield
   size_t simulated_depth = 0;
   for (size_t i = 0; i < instructions.size(); ++i) {
       const auto &instr = instructions[i];

       // Track stack effects of each opcode
       if (instr.opcode == op::LOAD_CONST || 
           instr.opcode == op::LOAD_FAST) {
           simulated_depth++;
       } else if (instr.opcode == op::BINARY_OP) {
           if (simulated_depth >= 2) simulated_depth--;
       }
       // ... handle all opcodes

       if (instr.opcode == op::YIELD_VALUE) {
           yield_stack_depth[i] = simulated_depth;
       }
   }

This simulation matches Python's ``dis.stack_effect()`` to ensure correct stack state preservation.

How Yields Work
---------------

When the generated code hits a ``YIELD_VALUE``:

1. **Save locals**: All local variables are stored to the ``locals`` array
2. **Save stack**: Stack values are preserved (their indices tracked by stack depth)
3. **Update state**: Set ``*state`` to this yield's number
4. **Return**: Return the yielded value to the caller

On resume:

1. **Read state**: Load ``*state`` to determine resume point
2. **Switch dispatch**: Jump to the correct basic block
3. **Restore locals**: Reload local variables from ``locals`` array
4. **Handle sent value**: If ``send()`` was used, ``sent_value`` contains it
5. **Continue execution**: Resume from after the yield

Example Generated IR
--------------------

For a generator like:

.. code-block:: python

   def gen():
       yield 1
       yield 2

The step function IR looks like:

.. code-block:: llvm

   define ptr @gen_step(ptr %state_ptr, ptr %locals_ptr, ptr %sent) {
   entry:
     %state = load i32, ptr %state_ptr
     switch i32 %state, label %error [
       i32 0, label %start
       i32 1, label %resume_after_yield_1
       i32 2, label %resume_after_yield_2
     ]

   start:
     ; First execution
     store i32 1, ptr %state_ptr
     ret ptr @py_int_1

   resume_after_yield_1:
     store i32 2, ptr %state_ptr  
     ret ptr @py_int_2

   resume_after_yield_2:
     store i32 -1, ptr %state_ptr
     call void @PyErr_SetNone(ptr @PyExc_StopIteration)
     ret ptr null

   error:
     store i32 -2, ptr %state_ptr
     ret ptr null
   }

JITGeneratorObject
------------------

The ``JITGeneratorObject`` (defined in ``jit_core.h``) wraps the step function:

.. code-block:: cpp

   struct JITGeneratorObject {
       PyObject_HEAD
       int32_t state;              // Current state
       PyObject** locals;          // Preserved variables
       Py_ssize_t num_locals;      // Size of locals array
       GeneratorStepFunc step_func; // The compiled step function
       PyObject* name;             // For repr()
       PyObject* qualname;         // Qualified name
   };

It's a proper Python type that implements:

- ``__iter__()``: Returns self
- ``__next__()``: Calls ``send(None)``
- ``send(value)``: Calls step function with value
- ``throw(exc)``: Raises exception in generator
- ``close()``: Closes generator

Async/Await Support
-------------------

Coroutines use the same state machine approach. The key difference is the ``GET_AWAITABLE`` opcode.

JITGetAwaitable Helper
^^^^^^^^^^^^^^^^^^^^^^

JustJIT implements a C helper (line 166 in ``jit_core.cpp``) for ``await``:

.. code-block:: cpp

   extern "C" PyObject* JITGetAwaitable(PyObject *obj) {
       // Check if it's a native coroutine
       const char* type_name = Py_TYPE(obj)->tp_name;
       if (strcmp(type_name, "coroutine") == 0) {
           Py_INCREF(obj);
           return obj;
       }

       // Check if it's a generator with @types.coroutine
       if (strcmp(type_name, "generator") == 0) {
           PyObject *gi_code = PyObject_GetAttrString(obj, "gi_code");
           // Check CO_ITERABLE_COROUTINE flag
           // ...
       }

       // Otherwise, call __await__()
       PyObject *await_method = PyObject_GetAttrString(obj, "__await__");
       // ...
   }

This handles three cases:

1. Native coroutines (``async def``) - return directly
2. Generators decorated with ``@types.coroutine`` - return as awaitable
3. Objects with ``__await__`` method - call it and return iterator

Coroutine Object
^^^^^^^^^^^^^^^^

``JITCoroutineObject`` (lines 15100-15513 in ``jit_core.cpp``) extends the generator model:

.. code-block:: cpp

   struct JITCoroutineObject {
       PyObject_HEAD
       int32_t state;              // Current state (0=initial, -1=done, -2=error)
       PyObject** locals;          // Preserved variables
       Py_ssize_t num_locals;      // Size of locals array
       GeneratorStepFunc step_func; // Compiled step function
       PyObject* name;             // For repr()
       PyObject* qualname;         // Qualified name
       PyObject* awaiting;         // Currently awaited object (NULL if not awaiting)
   };

The ``awaiting`` field (unique to coroutines) tracks the currently awaited object. When ``await`` is encountered:

1. ``GET_AWAITABLE`` gets the iterator from the awaited object
2. ``awaiting`` is set to this iterator
3. ``SEND`` delegates to the awaited iterator
4. When iterator completes, ``awaiting`` is cleared and execution continues

The ``JITCoroutine_Send()`` method (lines 15244-15351) handles delegation:

.. code-block:: cpp

   PyObject* JITCoroutine_Send(JITCoroutineObject* coro, PyObject* value) {
       // If we're awaiting something, delegate to it first
       if (coro->awaiting != NULL) {
           // Try to send value to the awaited object
           if (is_gen_or_coro) {
               result = send_meth(value);  // Delegate send
           } else {
               result = PyIter_Next(coro->awaiting);  // Iterator
           }
           if (result != NULL) return result;  // Propagate yielded value
           
           // Awaited object finished - extract StopIteration.value
           if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
               // Get return value and continue with it
               value = extract_stop_iteration_value();
               Py_CLEAR(coro->awaiting);
           }
       }
       // Call the step function
       return coro->step_func(&coro->state, coro->locals, value);
   }

``JITCoroutineObject`` implements:

- ``__await__()``: Returns self (coroutine protocol)
- ``__iter__()``: Returns self
- ``__next__()``: Calls ``send(None)``
- ``send(value)``: Send value, delegates to awaited if active
- ``throw(exc)``: Throws into awaited object, then self
- ``close()``: Closes awaited object, then self

Async Generators
----------------

Async generators (``async def`` with ``yield``) are now fully supported. They combine both async/await and generator protocols.

Async Iteration Helpers
^^^^^^^^^^^^^^^^^^^^^^^

JustJIT provides C helpers for async iteration (lines 228-343 in ``jit_core.cpp``):

.. code-block:: cpp

   // GET_AITER opcode: Get async iterator from object
   extern "C" PyObject* JITGetAIter(PyObject *obj) {
       return PyObject_GetAIter(obj);  // Calls __aiter__()
   }

   // GET_ANEXT opcode: Get next awaitable from async iterator  
   extern "C" PyObject* JITGetANext(PyObject *aiter) {
       PyObject *anext_method = PyObject_GetAttrString(aiter, "__anext__");
       return PyObject_CallNoArgs(anext_method);  // Returns awaitable
   }

   // END_ASYNC_FOR opcode: Handle StopAsyncIteration
   extern "C" int JITEndAsyncFor(PyObject *exc) {
       if (PyErr_GivenExceptionMatches(exc, PyExc_StopAsyncIteration)) {
           PyErr_Clear();
           return 1;  // Success - loop ends normally
       }
       return 0;  // Re-raise other exceptions
   }

Async Generator Adapter
^^^^^^^^^^^^^^^^^^^^^^^

Async generators are wrapped with ``_AsyncGeneratorAdapter`` (in ``__init__.py``) that provides the async protocol:

.. code-block:: python

   class _AsyncGeneratorAdapter:
       def __aiter__(self):
           return self
       
       async def __anext__(self):
           try:
               return self._inner.send(None)
           except StopIteration:
               raise StopAsyncIteration
       
       async def asend(self, value): ...
       async def athrow(self, exc): ...
       async def aclose(self): ...

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   import justjit

   @justjit.jit
   async def async_range(n):
       for i in range(n):
           yield i

   async def main():
       async for x in async_range(5):
           print(x)  # Prints 0, 1, 2, 3, 4

   asyncio.run(main())

What's Fully Supported
----------------------

- Basic generators with ``yield``
- ``yield from`` delegation 
- Generator expressions
- ``send()`` and ``throw()`` methods
- Async functions with ``await``
- Nested coroutine calls
- ``CLEANUP_THROW`` opcode (exception handling during throw()/close())
- **Async generators** (``async def`` with ``yield``)
- Async iteration opcodes: ``GET_AITER``, ``GET_ANEXT``, ``END_ASYNC_FOR``

Partial Support
---------------

- Some edge cases with ``close()`` and complex exception chains

Performance Notes
-----------------

Generator overhead comes from:

1. State machine dispatch (switch on state)
2. Local variable save/restore
3. Python object creation for yielded values

For tight loops, consider native int/float mode instead of generators when possible.
