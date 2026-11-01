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
            try:
                os.add_dll_directory(_path)
            except OSError:
                pass
    
    # Also check vcpkg paths for zlib dependency
    _vcpkg_paths = [
        r"C:\vcpkg\installed\x64-windows\bin",
        os.path.join(os.environ.get("VCPKG_ROOT", ""), "installed", "x64-windows", "bin"),
    ]
    for _path in _vcpkg_paths:
        _path = os.path.normpath(_path)
        if os.path.isdir(_path):
            try:
                os.add_dll_directory(_path)
            except OSError:
                pass

# Now import the C++ extension module
from ._core import JIT, create_jit_generator, create_jit_coroutine

# InlineCCompiler is only available if Clang support was compiled in
try:
    from ._core import InlineCCompiler
    _HAS_CLANG = True
except ImportError:
    _HAS_CLANG = False
    InlineCCompiler = None

__version__ = "0.1.7"
__all__ = ["JIT", "jit", "dump_ir", "create_jit_generator", "create_jit_coroutine", "InlineCCompiler", "inline_c", "dump_c_ir"]

# Python code flags
_CO_GENERATOR = 0x20
_CO_COROUTINE = 0x80
_CO_ASYNC_GENERATOR = 0x200

# Generator/coroutine opcodes that we cannot JIT compile
# Note: These are opcodes we HAVE now implemented support for (used to detect generator mode)
_GENERATOR_OPCODES = {
    "YIELD_VALUE",
    "RETURN_GENERATOR",
    "GEN_START",
    "SEND",
    "GET_AWAITABLE",
    "GET_YIELD_FROM_ITER",
    # Now supported for async generators:
    "GET_AITER",
    "GET_ANEXT",
    "END_ASYNC_FOR",
    "ASYNC_GEN_WRAP",
}

# Exception handling opcodes that we cannot JIT compile (Bug #3)
# Basic exception opcodes (PUSH_EXC_INFO, POP_EXCEPT, CHECK_EXC_MATCH, RAISE_VARARGS, RERAISE)
# are now supported via exception table parsing and error checking.
# With statement opcodes (BEFORE_WITH, WITH_EXCEPT_START) are also supported.
# CLEANUP_THROW is now supported in compile_generator.
# These remaining opcodes are for more complex constructs.
_EXCEPTION_OPCODES = {"SETUP_FINALLY", "POP_BLOCK"}

# Pattern matching opcodes - now supported with proper CFG analysis and PHI nodes
# These opcodes work correctly with the implemented stack state tracking.
_PATTERN_MATCHING_OPCODES = set()  # All pattern matching opcodes are now supported


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
        "SEND",  # SEND jumps to target when sub-iterator returns
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
        """Read a variable-length integer from exception table.
        
        Python 3.11+ uses a big-endian varint format:
        - Bit 6 (0x40) is the continuation flag
        - Bits 0-5 (0x3F) are the value bits
        - Values are built from high bits to low bits
        """
        b = data[pos]
        pos += 1
        val = b & 0x3F
        while b & 0x40:
            val <<= 6
            b = data[pos]
            pos += 1
            val |= b & 0x3F
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
        "LOAD_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST",  # Python 3.13 combined opcodes
        "STORE_FAST_LOAD_FAST", "LOAD_FAST_AND_CLEAR",   # Python 3.13 combined opcodes
        "LOAD_LOCALS",  # Python 3.13 annotation scope
        "LOAD_FROM_DICT_OR_DEREF", "LOAD_FROM_DICT_OR_GLOBALS",  # Python 3.13 annotation scope
        # Global/builtin access
        "LOAD_GLOBAL", "LOAD_ATTR", "PUSH_NULL",
        "STORE_GLOBAL", "STORE_ATTR",
        "CALL", "CALL_KW", "CALL_FUNCTION_EX",
        "CALL_INTRINSIC_2",  # Two-arg intrinsics (type params, etc.)
        # Generator specific
        "YIELD_VALUE", "RETURN_VALUE", "RETURN_CONST",
        # Stack manipulation
        "POP_TOP", "COPY", "SWAP",
        # Arithmetic
        "BINARY_OP",
        # Comparison and boolean
        "COMPARE_OP", "CONTAINS_OP", "IS_OP", "TO_BOOL",
        # Unary operations
        "UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT",
        # Looping support
        "GET_ITER", "FOR_ITER", "END_FOR",
        "JUMP_BACKWARD", "JUMP_FORWARD", "JUMP_BACKWARD_NO_INTERRUPT",
        "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE",
        # Collections
        "BUILD_LIST", "BUILD_TUPLE", "BUILD_CONST_KEY_MAP",
        "BUILD_MAP", "BUILD_SET",
        "LIST_APPEND", "SET_ADD", "MAP_ADD",
        "LIST_EXTEND", "SET_UPDATE", "DICT_UPDATE", "DICT_MERGE",
        # Subscript and slicing
        "BINARY_SUBSCR", "STORE_SUBSCR", "DELETE_SUBSCR",
        "BUILD_SLICE", "BINARY_SLICE", "STORE_SLICE",
        # Unpacking
        "UNPACK_SEQUENCE", "UNPACK_EX",
        # Exception handling (supported via exception table)
        "CALL_INTRINSIC_1", "RERAISE", "PUSH_EXC_INFO", "POP_EXCEPT",
        "CHECK_EXC_MATCH", "CHECK_EG_MATCH",  # except* exception group matching
        "RAISE_VARARGS",  # raise, raise exc, raise exc from cause
        # Closures
        "LOAD_DEREF", "STORE_DEREF", "LOAD_CLOSURE",
        "COPY_FREE_VARS", "MAKE_CELL",
        # Import
        "IMPORT_NAME", "IMPORT_FROM",
        # Function creation
        "MAKE_FUNCTION", "SET_FUNCTION_ATTRIBUTE",
        # Class support
        "SETUP_ANNOTATIONS", "EXIT_INIT_CHECK",
        # Async with
        "BEFORE_ASYNC_WITH",
    }
    
    # Note: All opcodes that appear in actual bytecode are now supported!
    # - LOAD_METHOD is a pseudo-instruction (emitted as LOAD_ATTR with flag)
    # - RAISE_VARARGS is implemented (line 6513 in jit_core.cpp)
    # - LOAD_FAST_AND_CLEAR is implemented (line 1973, 8644 in jit_core.cpp)
    
    # Allow all opcodes - we have full coverage
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
        "CALL_INTRINSIC_1", "CALL_INTRINSIC_2", "RERAISE", "PUSH_EXC_INFO", "POP_EXCEPT",
        "CHECK_EXC_MATCH", "CHECK_EG_MATCH",  # except* exception group matching
        "RAISE_VARARGS",  # raise, raise exc, raise exc from cause
        # Local/global access
        "LOAD_FAST", "STORE_FAST", "LOAD_CONST", "LOAD_GLOBAL",
        "STORE_GLOBAL", "LOAD_ATTR", "STORE_ATTR", "LOAD_NAME",
        "LOAD_LOCALS", "LOAD_FROM_DICT_OR_DEREF", "LOAD_FROM_DICT_OR_GLOBALS",  # Python 3.13
        # Python 3.13 combined opcodes
        "LOAD_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST", "LOAD_FAST_CHECK",
        "LOAD_FAST_AND_CLEAR", "STORE_FAST_LOAD_FAST",
        # Operations
        "BINARY_OP", "COMPARE_OP", "UNARY_NEGATIVE", "UNARY_NOT",
        "UNARY_INVERT", "BUILD_LIST", "BUILD_TUPLE", "BUILD_SET",
        "BUILD_MAP", "BUILD_STRING", "BUILD_CONST_KEY_MAP",
        "LIST_APPEND", "SET_ADD", "MAP_ADD", "LIST_EXTEND", 
        "SET_UPDATE", "DICT_MERGE", "DICT_UPDATE", 
        "CALL", "CALL_FUNCTION_EX", "CALL_KW", "PUSH_NULL",
        "POP_TOP", "COPY", "SWAP", "NOP", "CACHE",
        # Control flow
        "JUMP_BACKWARD", "JUMP_FORWARD", "JUMP_BACKWARD_NO_INTERRUPT",
        "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE",
        # Coroutine exception handling
        "CLEANUP_THROW",
        # Subscript
        "BINARY_SUBSCR", "STORE_SUBSCR", "DELETE_SUBSCR",
        # Iteration
        "GET_ITER", "FOR_ITER", "END_FOR",
        # Closures
        "LOAD_DEREF", "STORE_DEREF", "COPY_FREE_VARS", "MAKE_CELL",
        # Async with
        "BEFORE_ASYNC_WITH",
        # Class support
        "SETUP_ANNOTATIONS", "EXIT_INIT_CHECK",
    }
    
    # Note: All opcodes are now supported - full Python 3.13 coverage
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


def _create_async_generator_wrapper(func, opt_level):
    """Create a JIT-compiled wrapper for an async generator function.
    
    Async generators are functions that combine both async def and yield.
    They produce values using yield but can also await other coroutines.
    
    The protocol is:
    - __aiter__() returns self
    - __anext__() returns an awaitable that produces the next value
    - asend(value) sends a value to the generator
    - athrow(exc) throws an exception into the generator
    - aclose() closes the generator
    
    For now, we compile async generators using the same mechanism as
    regular generators, but the caller (async for loop) must handle
    the async protocol wrapping.
    """
    import functools
    import warnings
    
    # Compile using the generator compilation path
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
    stack_size = func.__code__.co_stacksize
    total_locals = base_locals + stack_size + 10
    
    # Compile the async generator to a step function
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
            f"Failed to JIT compile async generator '{func.__name__}'. "
            f"Using Python implementation.",
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
    def async_generator_factory(*args, **kwargs):
        """Factory function that creates a new async generator each time it's called."""
        if kwargs:
            # Fall back for kwargs (complex case)
            return func(*args, **kwargs)
        
        if len(args) != param_count:
            raise TypeError(
                f"{func.__name__}() takes {param_count} positional arguments "
                f"but {len(args)} were given"
            )
        
        # Create a JIT generator as the underlying implementation
        # The async generator protocol is handled by wrapping this
        gen = create_jit_generator(step_func_addr, num_locals, gen_name, gen_qualname)
        
        # Store arguments in the generator's locals array
        for i, arg in enumerate(args):
            gen._set_local(i, arg)
        
        # Wrap in an async generator adapter
        return _AsyncGeneratorAdapter(gen, gen_name, gen_qualname)
    
    async_generator_factory._jit_instance = jit_instance
    async_generator_factory._original_func = func
    async_generator_factory._instructions = instructions
    async_generator_factory._mode = "async_generator"
    async_generator_factory._is_jit_async_generator = True
    
    return async_generator_factory


class _AsyncGeneratorAdapter:
    """Adapter that wraps a JIT generator to implement the async generator protocol.
    
    This provides the asynchronous iteration interface (__aiter__, __anext__, etc.)
    on top of a synchronous JIT-compiled generator.
    """
    
    def __init__(self, inner_gen, name, qualname):
        self._inner = inner_gen
        self._name = name
        self._qualname = qualname
        self._closed = False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        
        try:
            # Get next value from inner generator
            value = self._inner.send(None)
            return value
        except StopIteration:
            self._closed = True
            raise StopAsyncIteration
    
    async def asend(self, value):
        if self._closed:
            raise StopAsyncIteration
        
        try:
            result = self._inner.send(value)
            return result
        except StopIteration:
            self._closed = True
            raise StopAsyncIteration
    
    async def athrow(self, exc_type, exc_val=None, exc_tb=None):
        if self._closed:
            raise StopAsyncIteration
        
        try:
            if exc_val is None:
                exc_val = exc_type()
            return self._inner.throw(exc_type, exc_val, exc_tb)
        except StopIteration:
            self._closed = True
            raise StopAsyncIteration
    
    async def aclose(self):
        if not self._closed:
            try:
                self._inner.close()
            except Exception:
                pass
            self._closed = True
    
    def __repr__(self):
        return f"<async_gen_jit {self._qualname} at {id(self):#x}>"


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
    
    # Async generators (async def with yield) - now supported
    if flags & _CO_ASYNC_GENERATOR:
        return _create_async_generator_wrapper(func, opt_level)

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
    use_float_mode = mode == "float"
    use_bool_mode = mode == "bool"
    use_int32_mode = mode == "int32"
    use_float32_mode = mode == "float32"
    use_complex128_mode = mode == "complex128"
    use_ptr_mode = mode == "ptr"
    use_vec4f_mode = mode == "vec4f"
    use_vec8i_mode = mode == "vec8i"
    use_complex64_mode = mode == "complex64"
    use_optional_f64_mode = mode == "optional_f64"

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
            elif use_float_mode:
                # Float mode - pure native f64 operations
                success = jit_instance.compile_float(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_float_callable(func.__name__, param_count)
            elif use_bool_mode:
                # Bool mode - pure native boolean operations
                success = jit_instance.compile_bool(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_bool_callable(func.__name__, param_count)
            elif use_int32_mode:
                # Int32 mode - 32-bit integer for C interop
                success = jit_instance.compile_int32(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_int32_callable(func.__name__, param_count)
            elif use_float32_mode:
                # Float32 mode - 32-bit float for SIMD/ML
                success = jit_instance.compile_float32(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_float32_callable(func.__name__, param_count)
            elif use_complex128_mode:
                # Complex128 mode - native {double,double} struct for complex numbers
                success = jit_instance.compile_complex128(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_complex128_callable(func.__name__, param_count)
            elif use_ptr_mode:
                # Ptr mode - array element access via GEP
                success = jit_instance.compile_ptr(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_ptr_callable(func.__name__, param_count)
            elif use_vec4f_mode:
                # Vec4f mode - SSE SIMD <4 x float>
                success = jit_instance.compile_vec4f(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_vec4f_callable(func.__name__, param_count)
            elif use_vec8i_mode:
                # Vec8i mode - AVX SIMD <8 x i32>
                success = jit_instance.compile_vec8i(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_vec8i_callable(func.__name__, param_count)
            elif use_complex64_mode:
                # Complex64 mode - single-precision complex {float, float}
                success = jit_instance.compile_complex64(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_complex64_callable(func.__name__, param_count)
            elif use_optional_f64_mode:
                # Optional<f64> mode - nullable float64 {i1, f64}
                success = jit_instance.compile_optional_f64(
                    instructions, constants, func.__name__, param_count, total_locals
                )
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_optional_f64_callable(func.__name__, param_count)
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
    wrapper._mode = "int" if use_int_mode else ("float" if use_float_mode else ("bool" if use_bool_mode else ("int32" if use_int32_mode else ("float32" if use_float32_mode else ("complex128" if use_complex128_mode else ("ptr" if use_ptr_mode else ("vec4f" if use_vec4f_mode else ("vec8i" if use_vec8i_mode else ("complex64" if use_complex64_mode else ("optional_f64" if use_optional_f64_mode else "object"))))))))))
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
    elif func._mode == "float":
        jit_instance.compile_float(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "bool":
        jit_instance.compile_bool(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "int32":
        jit_instance.compile_int32(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "float32":
        jit_instance.compile_float32(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "complex128":
        jit_instance.compile_complex128(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "ptr":
        jit_instance.compile_ptr(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "vec4f":
        jit_instance.compile_vec4f(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "vec8i":
        jit_instance.compile_vec8i(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "complex64":
        jit_instance.compile_complex64(
            instructions, constants, ir_name, param_count, total_locals
        )
    elif func._mode == "optional_f64":
        jit_instance.compile_optional_f64(
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


# ============================================================================
# Inline C Compiler - Compile C/C++ code at runtime
# ============================================================================

# Global InlineCCompiler instance (lazy initialized)
_global_c_compiler = None
_global_jit_for_c = None
_last_c_ir = None  # Store last compiled C IR


def dump_c_ir():
    """
    Get the LLVM IR from the last inline_c compilation.
    
    Returns:
        str: LLVM IR string, or None if no compilation has been done
        
    Example:
        result = inline_c('int add(int a, int b) { return a + b; }')
        print(dump_c_ir())  # Prints the LLVM IR
    """
    if _global_c_compiler:
        return _global_c_compiler.get_last_ir()
    return None


def inline_c(code, lang="c", captured_vars=None, include_paths=None, dump_ir=False):
    """
    Compile C/C++ code at runtime and return callable functions.
    
    This uses Clang's Driver API to automatically detect the system's
    C/C++ toolchain (MSVC, MinGW, etc.) without hardcoded paths.
    
    Args:
        code: C or C++ source code to compile
        lang: "c" or "c++" (default: "c")
        captured_vars: dict of Python variables to inject into C code
        include_paths: list of additional include directories
        
    Returns:
        dict containing:
        - 'functions': list of compiled function names
        - Each function name maps to a callable
        
    Example:
        result = inline_c('''
            int add(int a, int b) {
                return a + b;
            }
        ''')
        
        add_func = result['add']
        print(add_func(3, 5))  # Output: 8
        
    For functions requiring includes:
        result = inline_c('''
            #include <stdio.h>
            void hello() { printf("Hello!\\n"); }
        ''')
        # Note: Requires MSVC or MinGW installed on Windows
        
    Raises:
        RuntimeError: If Clang support was not compiled in
        RuntimeError: If C compilation fails
    """
    global _global_c_compiler, _global_jit_for_c
    
    if not _HAS_CLANG:
        raise RuntimeError(
            "inline_c requires Clang support. "
            "Rebuild justjit with JUSTJIT_HAS_CLANG=1 and Clang libraries."
        )
    
    # Lazy init global compiler
    if _global_c_compiler is None:
        _global_jit_for_c = JIT()
        _global_c_compiler = InlineCCompiler(_global_jit_for_c)
        
        # ================================================================
        # RAILGUARD: Auto-detect bundled libc headers from package
        # ================================================================
        try:
            import warnings
            package_dir = os.path.dirname(os.path.abspath(__file__))
            libc_headers_path = os.path.join(package_dir, "vendor", "libc-headers")
            
            if os.path.exists(libc_headers_path):
                # Bundled headers found - add to include path automatically
                _global_c_compiler.add_include_path(libc_headers_path)
            else:
                # RAILGUARD WARNING: vendor folder missing from pip install
                warnings.warn(
                    "JustJIT: Bundled libc headers not found. "
                    "The inline_c function may fail for standard includes like <stdio.h>. "
                    "Please reinstall justjit or provide include_paths manually.",
                    RuntimeWarning,
                    stacklevel=2
                )
        except Exception:
            pass  # Silently continue if railguard check fails
        
        # Auto-add common include paths
        # 1. Current working directory
        _global_c_compiler.add_include_path(os.getcwd())
        
        # 2. Try to get caller's script directory
        try:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_file = frame.f_back.f_back.f_globals.get('__file__')
                if caller_file:
                    caller_dir = os.path.dirname(os.path.abspath(caller_file))
                    if caller_dir and caller_dir != os.getcwd():
                        _global_c_compiler.add_include_path(caller_dir)
        except Exception:
            pass  # Ignore errors in path detection
    
    # Add user-provided include paths
    if include_paths:
        for path in include_paths:
            _global_c_compiler.add_include_path(path)
    
    # Compile and execute
    if captured_vars is None:
        captured_vars = {}
    
    # Enable IR capture if requested
    global _last_c_ir
    if dump_ir and _global_jit_for_c:
        _global_jit_for_c.set_dump_ir(True)
    
    result = _global_c_compiler.compile(code, lang, captured_vars)
    
    # Capture IR if requested
    if dump_ir and _global_jit_for_c:
        _last_c_ir = _global_jit_for_c.get_last_ir()
        _global_jit_for_c.set_dump_ir(False)
    
    return result

