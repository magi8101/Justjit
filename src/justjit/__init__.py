import os
import sys
import dis
import types

# Add DLL directories on Windows before importing the extension
if sys.platform == "win32":
    # Add the package directory itself (contains LLVM DLLs)
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    os.add_dll_directory(_package_dir)

    # Check common LLVM installation paths
    _llvm_paths = [
        os.path.join(os.environ.get("LLVM_DIR", ""), "..", "..", "..", "bin"),
        r"C:\Program Files\LLVM\bin",
        os.path.expanduser(r"~\llvm-project\build\Release\bin"),
    ]
    for _path in _llvm_paths:
        _path = os.path.normpath(_path)
        if os.path.isdir(_path):
            os.add_dll_directory(_path)
            break

# Now import the C++ extension module
from ._core import JIT, create_jit_generator, create_jit_coroutine

__version__ = "0.1.2"
__all__ = ["JIT", "jit", "dump_ir", "create_jit_generator", "create_jit_coroutine"]

# Python code flags
_CO_GENERATOR = 0x20
_CO_COROUTINE = 0x80
_CO_ASYNC_GENERATOR = 0x200

# Generator/coroutine opcodes that we cannot JIT compile
_GENERATOR_OPCODES = {
    "YIELD_VALUE",
    "RETURN_GENERATOR",
    "GEN_START",
    "SEND",
    "END_ASYNC_FOR",
    "GET_AWAITABLE",
    "GET_AITER",
    "GET_ANEXT",
    "GET_YIELD_FROM_ITER",
    "ASYNC_GEN_WRAP",
}

# Exception handling opcodes that we cannot JIT compile (Bug #3)
# Basic exception opcodes (PUSH_EXC_INFO, POP_EXCEPT, CHECK_EXC_MATCH, RAISE_VARARGS, RERAISE)
# are now supported via exception table parsing and error checking.
# With statement opcodes (BEFORE_WITH, WITH_EXCEPT_START) are also supported.
# These remaining opcodes are for more complex constructs.
_EXCEPTION_OPCODES = {"CLEANUP_THROW", "SETUP_FINALLY", "POP_BLOCK"}

# Pattern matching opcodes - the opcodes themselves work, but pattern matching
# creates complex control flow (conditional branches, stack manipulation) that
# causes LLVM verification errors when branches merge with different stack states.
# Keep blocked until proper CFG analysis is implemented.
_PATTERN_MATCHING_OPCODES = {"MATCH_CLASS", "MATCH_KEYS", "MATCH_MAPPING", "MATCH_SEQUENCE"}


def _is_generator_or_coroutine(func):
    """Check if function is a generator, coroutine, or async generator."""
    flags = func.__code__.co_flags
    return bool(flags & (_CO_GENERATOR | _CO_COROUTINE | _CO_ASYNC_GENERATOR))


def _has_unsupported_opcodes(func):
    """Check if function contains opcodes we cannot JIT compile."""
    for instr in dis.get_instructions(func):
        if instr.opname in _GENERATOR_OPCODES:
            return "generator"
        if instr.opname in _EXCEPTION_OPCODES:
            return "exception"
        if instr.opname in _PATTERN_MATCHING_OPCODES:
            return "pattern_matching"
        # All CALL_INTRINSIC_1 args are now supported (1-11)
    return None


def jit(
    func=None,
    *,
    opt_level=3,
    vectorize=True,
    inline=True,
    parallel=False,
    lazy=False,
    mode="auto",
):
    """
    JIT compile a Python function for aggressive performance optimization.

    Args:
        func: The function to compile (when used without parentheses)
        opt_level: LLVM optimization level (0-3, default 3 for maximum performance)
        vectorize: Enable loop vectorization (default True)
        inline: Enable function inlining (default True)
        parallel: Enable parallelization (default False)
        lazy: Delay compilation until first call (default False)
        mode: Compilation mode - 'auto', 'object', or 'int' (default 'auto')
              'int' mode generates native integer code with no Python object overhead

    Example:
        @jit
        def add(a, b):
            return a + b

        @jit(mode='int')  # Pure integer mode - maximum speed
        def mul(a, b):
            return a * b
    """
    if func is None:

        def decorator(f):
            return _create_jit_wrapper(
                f, opt_level, vectorize, inline, parallel, lazy, mode
            )

        return decorator
    return _create_jit_wrapper(func, opt_level, vectorize, inline, parallel, lazy, mode)


def _extract_bytecode(func):
    """Extract bytecode instructions from a Python function."""
    # Opcodes that use argval as a jump target (offset)
    JUMP_OPCODES = {
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "POP_JUMP_IF_NONE",
        "POP_JUMP_IF_NOT_NONE",
        "FOR_ITER",
        "JUMP_BACKWARD_NO_INTERRUPT",
    }

    instructions = []
    for instr in dis.get_instructions(func):
        # Skip CACHE instructions - they're just placeholders for the adaptive interpreter
        if instr.opname == "CACHE":
            continue

        # Handle argval - we only care about integer values for jump targets
        # For non-jump instructions, argval can be the constant VALUE which may overflow int32
        argval = 0
        if instr.opname in JUMP_OPCODES:
            # For jumps, argval is the target offset - pass it
            if hasattr(instr, "argval") and isinstance(instr.argval, int):
                argval = instr.argval
        # For all other opcodes (LOAD_CONST, RETURN_CONST, etc), argval stays 0
        # We use instr.arg as the index into constants/names/locals

        instructions.append(
            {
                "opcode": instr.opcode,
                "arg": instr.arg if instr.arg is not None else 0,
                "argval": argval,
                "offset": instr.offset,
            }
        )
    return instructions


def _extract_constants(func):
    """Extract constant values from code object."""
    # Pass all constants as-is, let C++ side handle them
    return list(func.__code__.co_consts)


def _extract_names(func):
    """Extract names from code object (for LOAD_ATTR, LOAD_GLOBAL, etc)."""
    return list(func.__code__.co_names)


def _extract_globals(func):
    """Extract globals dictionary for runtime lookup.

    Returns the function's __globals__ dict directly so that
    global variable lookups happen at runtime, not compile time.
    This ensures changes to globals after JIT compilation are visible.
    """
    return func.__globals__


def _extract_builtins(func):
    """Extract builtins dict for fallback lookup."""
    import builtins

    return builtins.__dict__


def _extract_closure(func):
    """Extract closure cells from a function.

    Returns a list of cell objects (or empty list if no closure).
    Closure cells contain values captured from enclosing scopes.
    """
    if func.__closure__ is not None:
        return list(func.__closure__)
    return []


def _parse_exception_table(func):
    """Parse Python 3.11+ exception table from code object.

    Returns a list of dicts with keys:
    - start: start offset of protected range
    - end: end offset of protected range
    - target: handler offset (PUSH_EXC_INFO location)
    - depth: stack depth to unwind to
    - lasti: whether to push last instruction offset
    """
    table = func.__code__.co_exceptiontable
    if not table:
        return []

    def read_varint(data, pos):
        """Read a variable-length integer from exception table."""
        val = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            val |= (b & 0x3F) << shift
            shift += 6
            if not (b & 0x40):
                break
        return val, pos

    entries = []
    i = 0
    while i < len(table):
        start, i = read_varint(table, i)
        length, i = read_varint(table, i)
        target, i = read_varint(table, i)
        depth_lasti, i = read_varint(table, i)
        depth = depth_lasti >> 1
        lasti = bool(depth_lasti & 1)

        entries.append(
            {
                "start": start * 2,  # Convert to byte offset
                "end": (start + length) * 2,
                "target": target * 2,
                "depth": depth,
                "lasti": lasti,
            }
        )

    return entries


def _is_simple_generator(func):
    """Check if a generator only uses opcodes the JIT generator compiler supports.
    
    The generator compiler supports most common opcodes. This function returns
    False for opcodes that are NOT yet implemented in compile_generator.
    """
    # Opcodes supported by the generator compiler
    # These opcodes are actually implemented in JITCore::compile_generator()
    supported_opcodes = {
        # Control flow
        "RESUME", "RETURN_GENERATOR", "NOP", "CACHE",
        # Load/store
        "LOAD_CONST", "LOAD_FAST", "LOAD_FAST_CHECK", "STORE_FAST",
        "LOAD_FAST_LOAD_FAST",  # Python 3.13 optimization
        # Global/builtin access
        "LOAD_GLOBAL", "LOAD_ATTR", "PUSH_NULL",
        "CALL",
        # Generator specific
        "YIELD_VALUE", "RETURN_VALUE", "RETURN_CONST",
        # Stack manipulation
        "POP_TOP", "COPY", "SWAP",
        # Arithmetic
        "BINARY_OP",
        # Comparison
        "COMPARE_OP",
        # Looping support
        "GET_ITER", "FOR_ITER", "END_FOR",
        "JUMP_BACKWARD", "JUMP_FORWARD", "JUMP_BACKWARD_NO_INTERRUPT",
        "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE",
        # Collections
        "BUILD_LIST", "BUILD_TUPLE", "BUILD_CONST_KEY_MAP",
        # Subscript
        "BINARY_SUBSCR", "STORE_SUBSCR",
        # Exception handling (supported via exception table)
        "CALL_INTRINSIC_1", "RERAISE", "PUSH_EXC_INFO", "POP_EXCEPT",
        "CHECK_EXC_MATCH",
    }
    
    # Opcodes NOT yet supported - will cause fallback to Python
    unsupported_opcodes = {
        # Complex function calls
        "CALL_FUNCTION_EX", "CALL_KW",
        "LOAD_METHOD",
        "STORE_GLOBAL", "STORE_ATTR",
        # More collections
        "BUILD_MAP", "BUILD_SET",
        "LIST_APPEND", "SET_ADD", "MAP_ADD",
        "UNPACK_SEQUENCE", "UNPACK_EX",
        "DELETE_SUBSCR",
        # Complex exception handling
        "RAISE_VARARGS",
        # Closures
        "LOAD_DEREF", "STORE_DEREF", "LOAD_CLOSURE",
        "COPY_FREE_VARS", "MAKE_CELL",
        # Other
        "IMPORT_NAME", "IMPORT_FROM",
        "MAKE_FUNCTION", "SET_FUNCTION_ATTRIBUTE",
    }
    
    for instr in dis.get_instructions(func):
        if instr.opname in unsupported_opcodes:
            return False
        # All CALL_INTRINSIC_1 args (1-11) are now supported
        # If not in supported and not in unsupported, allow it (unknown new opcode)
    return True


def _create_generator_wrapper(func, opt_level):
    """Create a JIT-compiled wrapper for a generator function.
    
    This compiles the generator into a state machine and returns a factory
    function that creates JIT generator objects when called.
    """
    import functools
    import warnings
    
    # Check if this generator is simple enough for JIT compilation
    if not _is_simple_generator(func):
        # Complex generator - fall back to Python for safety
        warnings.warn(
            f"Generator '{func.__name__}' uses opcodes not yet supported by JIT. "
            f"Using Python implementation (no performance impact for generators).",
            RuntimeWarning,
            stacklevel=4,
        )
        return func
    
    jit_instance = JIT()
    jit_instance.set_opt_level(opt_level)
    
    instructions = _extract_bytecode(func)
    constants = _extract_constants(func)
    names = _extract_names(func)
    globals_dict = _extract_globals(func)
    builtins_dict = _extract_builtins(func)
    closure_cells = _extract_closure(func)
    exception_table = _parse_exception_table(func)
    
    param_count = func.__code__.co_argcount
    nlocals = func.__code__.co_nlocals
    num_cellvars = len(func.__code__.co_cellvars)
    num_freevars = len(func.__code__.co_freevars)
    base_locals = nlocals + num_cellvars + num_freevars
    
    # For generators, we need extra slots for stack persistence across yields
    # The stack may have values that need to survive the yield/resume boundary
    # We allocate co_stacksize additional slots after the regular locals
    max_stack_depth = func.__code__.co_stacksize
    total_locals = base_locals + max_stack_depth
    
    # Compile the generator to a step function
    success = jit_instance.compile_generator(
        instructions,
        constants,
        names,
        globals_dict,
        builtins_dict,
        closure_cells,
        exception_table,
        func.__name__,
        param_count,
        total_locals,
        nlocals,
    )
    
    if not success:
        warnings.warn(
            f"Failed to JIT compile generator '{func.__name__}'. "
            f"Falling back to Python implementation.",
            RuntimeWarning,
            stacklevel=4,
        )
        return func
    
    # Get the step function address and metadata
    gen_info = jit_instance.get_generator_callable(
        func.__name__,
        param_count,
        total_locals,
        func.__name__,
        func.__qualname__,
    )
    
    if gen_info is None:
        return func
    
    step_func_addr = gen_info["step_func_addr"]
    num_locals = gen_info["num_locals"]
    gen_name = gen_info["name"]
    gen_qualname = gen_info["qualname"]
    
    @functools.wraps(func)
    def generator_factory(*args, **kwargs):
        """Factory function that creates a new JIT generator each time it's called."""
        if kwargs:
            # For simplicity, don't support kwargs in generators yet
            # Fall back to original
            return func(*args, **kwargs)
        
        if len(args) != param_count:
            raise TypeError(
                f"{func.__name__}() takes {param_count} positional arguments "
                f"but {len(args)} were given"
            )
        
        # Create a new JIT generator
        gen = create_jit_generator(step_func_addr, num_locals, gen_name, gen_qualname)
        
        # Store arguments in the generator's locals array
        # The generator's step function expects args at indices 0..param_count-1
        for i, arg in enumerate(args):
            gen._set_local(i, arg)
        
        return gen
    
    generator_factory._jit_instance = jit_instance
    generator_factory._original_func = func
    generator_factory._instructions = instructions
    generator_factory._mode = "generator"
    generator_factory._is_jit_generator = True
    
    return generator_factory


def _is_simple_coroutine(func):
    """Check if an async function uses only supported opcodes.
    
    Currently supported async opcodes:
    - GET_AWAITABLE: Get awaitable from object
    - SEND: Core await mechanism  
    - END_SEND: Cleanup after send
    - All regular generator opcodes (RESUME, YIELD_VALUE, etc.)
    """
    # Async functions can use the same opcodes as generators plus async-specific ones
    async_supported = {
        "GET_AWAITABLE", "SEND", "END_SEND",
        # Coroutine entry/exit
        "RETURN_GENERATOR", "RETURN_VALUE", "RETURN_CONST",
        # Resume points
        "RESUME", "YIELD_VALUE",
        # Exception handling (required for coroutines - we'll handle at runtime)
        "CALL_INTRINSIC_1", "RERAISE", "PUSH_EXC_INFO", "POP_EXCEPT",
        # Local/global access
        "LOAD_FAST", "STORE_FAST", "LOAD_CONST", "LOAD_GLOBAL",
        "STORE_GLOBAL", "LOAD_ATTR", "STORE_ATTR", "LOAD_NAME",
        # Python 3.13 combined opcodes
        "LOAD_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST", "LOAD_FAST_CHECK",
        "LOAD_FAST_AND_CLEAR", "STORE_FAST_LOAD_FAST",
        # Operations
        "BINARY_OP", "COMPARE_OP", "UNARY_NEGATIVE", "UNARY_NOT",
        "UNARY_INVERT", "BUILD_LIST", "BUILD_TUPLE", "BUILD_SET",
        "BUILD_MAP", "BUILD_STRING", "BUILD_CONST_KEY_MAP",
        "LIST_APPEND", "SET_ADD", "MAP_ADD", "LIST_EXTEND", 
        "SET_UPDATE", "DICT_MERGE", "DICT_UPDATE", 
        "CALL", "CALL_FUNCTION_EX", "PUSH_NULL",
        "POP_TOP", "COPY", "SWAP", "NOP", "CACHE",
        # Control flow
        "JUMP_BACKWARD", "JUMP_FORWARD", 
        "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE",
        # Subscript
        "BINARY_SUBSCR", "STORE_SUBSCR", "DELETE_SUBSCR",
        # Iteration
        "GET_ITER", "FOR_ITER", "END_FOR",
        # Closures
        "LOAD_DEREF", "STORE_DEREF", "COPY_FREE_VARS", "MAKE_CELL",
    }
    
    for instr in dis.get_instructions(func):
        if instr.opname not in async_supported:
            return False
        # All CALL_INTRINSIC_1 args (1-11) are now supported
    return True


def _create_coroutine_wrapper(func, opt_level):
    """Create a JIT-compiled wrapper for an async function (coroutine).
    
    This compiles the async function into a state machine and returns a factory
    function that creates JIT coroutine objects when called.
    """
    import functools
    import warnings
    
    # Check if this coroutine is simple enough for JIT compilation
    if not _is_simple_coroutine(func):
        # Complex coroutine - fall back to Python for safety
        warnings.warn(
            f"Async function '{func.__name__}' uses opcodes not yet supported by JIT. "
            f"Using Python implementation.",
            RuntimeWarning,
            stacklevel=4,
        )
        return func
    
    jit_instance = JIT()
    jit_instance.set_opt_level(opt_level)
    
    instructions = _extract_bytecode(func)
    constants = _extract_constants(func)
    names = _extract_names(func)
    globals_dict = _extract_globals(func)
    builtins_dict = _extract_builtins(func)
    closure_cells = _extract_closure(func)
    exception_table = _parse_exception_table(func)
    
    param_count = func.__code__.co_argcount
    nlocals = func.__code__.co_nlocals
    num_cellvars = len(func.__code__.co_cellvars)
    num_freevars = len(func.__code__.co_freevars)
    base_locals = nlocals + num_cellvars + num_freevars
    
    # For coroutines, we need extra slots for stack persistence across awaits
    max_stack_depth = func.__code__.co_stacksize
    total_locals = base_locals + max_stack_depth
    
    # Compile the coroutine to a step function (same as generator)
    success = jit_instance.compile_generator(
        instructions,
        constants,
        names,
        globals_dict,
        builtins_dict,
        closure_cells,
        exception_table,
        func.__name__,
        param_count,
        total_locals,
        nlocals,
    )
    
    if not success:
        warnings.warn(
            f"Failed to JIT compile async function '{func.__name__}'. "
            f"Falling back to Python implementation.",
            RuntimeWarning,
            stacklevel=4,
        )
        return func
    
    # Get the step function address and metadata
    coro_info = jit_instance.get_generator_callable(
        func.__name__,
        param_count,
        total_locals,
        func.__name__,
        func.__qualname__,
    )
    
    if coro_info is None:
        return func
    
    step_func_addr = coro_info["step_func_addr"]
    num_locals = coro_info["num_locals"]
    coro_name = coro_info["name"]
    coro_qualname = coro_info["qualname"]
    
    @functools.wraps(func)
    def coroutine_factory(*args, **kwargs):
        """Factory function that creates a new JIT coroutine each time it's called."""
        if kwargs:
            # For simplicity, don't support kwargs in coroutines yet
            return func(*args, **kwargs)
        
        if len(args) != param_count:
            raise TypeError(
                f"{func.__name__}() takes {param_count} positional arguments "
                f"but {len(args)} were given"
            )
        
        # Create a new JIT coroutine
        coro = create_jit_coroutine(step_func_addr, num_locals, coro_name, coro_qualname)
        
        # Store arguments in the coroutine's locals array
        for i, arg in enumerate(args):
            coro._set_local(i, arg)
        
        return coro
    
    coroutine_factory._jit_instance = jit_instance
    coroutine_factory._original_func = func
    coroutine_factory._instructions = instructions
    coroutine_factory._mode = "coroutine"
    coroutine_factory._is_jit_coroutine = True
    
    return coroutine_factory


def _create_jit_wrapper(
    func, opt_level, vectorize, inline, parallel, lazy, mode="auto"
):
    """Create a JIT-compiled wrapper for the given function."""
    import warnings
    import functools

    # Check if this is a generator function
    is_generator = _is_generator_or_coroutine(func)
    
    # Handle coroutines (async def functions)
    flags = func.__code__.co_flags
    if flags & _CO_COROUTINE:
        # Regular coroutine (async def without yield)
        return _create_coroutine_wrapper(func, opt_level)
    
    # Async generators (async def with yield) - not yet supported
    if flags & _CO_ASYNC_GENERATOR:
        warnings.warn(
            f"Async generator function '{func.__name__}' cannot be JIT compiled yet. "
            f"The @jit decorator has no effect on this function.",
            RuntimeWarning,
            stacklevel=3,
        )
        return func

    # Check bytecode for unsupported opcodes
    unsupported = _has_unsupported_opcodes(func)
    if unsupported == "exception":
        # Bug #3 Fix: Detect exception handling and skip JIT compilation
        warnings.warn(
            f"Function '{func.__name__}' uses unsupported exception constructs. "
            f"The @jit decorator has no effect on this function.",
            RuntimeWarning,
            stacklevel=3,
        )
        return func
    if unsupported == "intrinsic":
        # Block functions using unsupported intrinsics (import *, type annotations)
        warnings.warn(
            f"Function '{func.__name__}' uses unsupported intrinsic operations "
            f"(import * or type annotations). The @jit decorator has no effect.",
            RuntimeWarning,
            stacklevel=3,
        )
        return func
    if unsupported == "pattern_matching":
        # Structural pattern matching has complex control flow
        warnings.warn(
            f"Function '{func.__name__}' uses structural pattern matching "
            f"(match/case with class/mapping/sequence patterns). "
            f"The @jit decorator has no effect. Literal matching works fine.",
            RuntimeWarning,
            stacklevel=3,
        )
        return func
    
    # For generators, compile using the generator compilation path
    if is_generator:
        return _create_generator_wrapper(func, opt_level)

    jit_instance = JIT()
    jit_instance.set_opt_level(opt_level)

    instructions = _extract_bytecode(func)
    constants = _extract_constants(func)
    names = _extract_names(func)
    globals_dict = _extract_globals(func)  # Now returns the dict itself
    builtins_dict = _extract_builtins(func)  # For fallback lookup
    closure_cells = _extract_closure(func)
    exception_table = _parse_exception_table(func)  # Bug #3 Fix: Exception handling
    param_count = func.__code__.co_argcount

    # Calculate local slot layout:
    # - nlocals: number of local variables (co_nlocals)
    # - cellvars: variables captured by nested functions (co_cellvars)
    # - freevars: variables from enclosing scope (co_freevars)
    # - total_locals: nlocals + len(cellvars) + len(freevars)
    nlocals = func.__code__.co_nlocals
    num_cellvars = len(func.__code__.co_cellvars)
    num_freevars = len(func.__code__.co_freevars)
    total_locals = nlocals + num_cellvars + num_freevars

    # Determine compilation mode
    use_int_mode = mode == "int"

    compiled_ptr = None

    def wrapper(*args, **kwargs):
        nonlocal compiled_ptr

        if compiled_ptr is None:
            if use_int_mode:
                # Integer mode - pure native i64 operations
                success = jit_instance.compile_int(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_int_callable(func.__name__, param_count)
            else:
                # Object mode - handles Python objects with closure support
                # Bug #4 Fix: Pass globals_dict and builtins_dict for runtime lookup
                # Bug #3 Fix: Pass exception_table for try/except handling
                success = jit_instance.compile(
                    instructions,
                    constants,
                    names,
                    globals_dict,
                    builtins_dict,
                    closure_cells,
                    exception_table,
                    func.__name__,
                    param_count,
                    total_locals,
                    nlocals,
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_callable(func.__name__, param_count)

            if compiled_ptr is None:
                return func(*args, **kwargs)

        try:
            return compiled_ptr(*args, **kwargs)
        except Exception:
            return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper._jit_instance = jit_instance
    wrapper._original_func = func
    wrapper._instructions = instructions
    wrapper._mode = "int" if use_int_mode else "object"
    return wrapper


def dump_ir(func):
    """
    Dump the LLVM IR for a JIT-compiled function.
    
    Args:
        func: A JIT-compiled function (decorated with @jit)
        
    Returns:
        str: The LLVM IR as a string, or None if function wasn't JIT compiled
        
    Example:
        @jit
        def add(a, b):
            return a + b
        
        add(1, 2)  # Trigger compilation
        print(dump_ir(add))
    """
    if not hasattr(func, '_jit_instance'):
        raise ValueError("Function is not a JIT-compiled function. Use @jit decorator first.")
    
    jit_instance = func._jit_instance
    original_func = func._original_func
    
    # Enable IR dump and recompile
    jit_instance.set_dump_ir(True)
    
    # Get compilation parameters
    instructions = func._instructions
    constants = _extract_constants(original_func)
    names = _extract_names(original_func)
    globals_dict = _extract_globals(original_func)
    builtins_dict = _extract_builtins(original_func)
    closure_cells = _extract_closure(original_func)
    exception_table = _parse_exception_table(original_func)
    
    code = original_func.__code__
    param_count = code.co_argcount
    nlocals = code.co_nlocals
    num_cellvars = len(code.co_cellvars)
    num_freevars = len(code.co_freevars)
    total_locals = nlocals + num_cellvars + num_freevars
    
    # Compile with a unique name to capture IR
    ir_name = f"{original_func.__name__}_ir_dump"
    
    if func._mode == "int":
        jit_instance.compile_int(
            instructions, constants, ir_name, param_count, total_locals
        )
    else:
        jit_instance.compile(
            instructions,
            constants,
            names,
            globals_dict,
            builtins_dict,
            closure_cells,
            exception_table,
            ir_name,
            param_count,
            total_locals,
            nlocals,
        )
    
    ir = jit_instance.get_last_ir()
    jit_instance.set_dump_ir(False)
    
    return ir
