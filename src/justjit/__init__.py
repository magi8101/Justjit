import os
import sys
import dis
import types

# Add DLL directories on Windows before importing the extension
if sys.platform == 'win32':
    # Add the package directory itself (contains LLVM DLLs)
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    os.add_dll_directory(_package_dir)
    
    # Check common LLVM installation paths
    _llvm_paths = [
        os.path.join(os.environ.get('LLVM_DIR', ''), '..', '..', '..', 'bin'),
        r'C:\Program Files\LLVM\bin',
        os.path.expanduser(r'~\llvm-project\build\Release\bin'),
    ]
    for _path in _llvm_paths:
        _path = os.path.normpath(_path)
        if os.path.isdir(_path):
            os.add_dll_directory(_path)
            break

# Now import the C++ extension module
from ._core import JIT

__version__ = "0.1.0"
__all__ = ["JIT", "jit"]

# Python code flags
_CO_GENERATOR = 0x20
_CO_COROUTINE = 0x80
_CO_ASYNC_GENERATOR = 0x200

# Generator/coroutine opcodes that we cannot JIT compile
_GENERATOR_OPCODES = {
    'YIELD_VALUE', 'RETURN_GENERATOR', 'GEN_START', 'SEND',
    'END_ASYNC_FOR', 'GET_AWAITABLE', 'GET_AITER', 'GET_ANEXT',
    'GET_YIELD_FROM_ITER', 'ASYNC_GEN_WRAP'
}

# Exception handling opcodes that we cannot JIT compile (Bug #3)
_EXCEPTION_OPCODES = {
    'PUSH_EXC_INFO', 'POP_EXCEPT', 'CHECK_EXC_MATCH', 'RAISE_VARARGS',
    'RERAISE', 'CLEANUP_THROW', 'SETUP_FINALLY', 'POP_BLOCK',
    'BEFORE_WITH', 'WITH_EXCEPT_START'
}


def _is_generator_or_coroutine(func):
    """Check if function is a generator, coroutine, or async generator."""
    flags = func.__code__.co_flags
    return bool(flags & (_CO_GENERATOR | _CO_COROUTINE | _CO_ASYNC_GENERATOR))


def _has_unsupported_opcodes(func):
    """Check if function contains opcodes we cannot JIT compile."""
    for instr in dis.get_instructions(func):
        if instr.opname in _GENERATOR_OPCODES:
            return 'generator'
        if instr.opname in _EXCEPTION_OPCODES:
            return 'exception'
    return None


def jit(func=None, *, opt_level=3, vectorize=True, inline=True, parallel=False, lazy=False, mode='auto'):
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
            return _create_jit_wrapper(f, opt_level, vectorize, inline, parallel, lazy, mode)
        return decorator
    return _create_jit_wrapper(func, opt_level, vectorize, inline, parallel, lazy, mode)


def _extract_bytecode(func):
    """Extract bytecode instructions from a Python function."""
    instructions = []
    for instr in dis.get_instructions(func):
        # Skip CACHE instructions - they're just placeholders for the adaptive interpreter
        if instr.opname == 'CACHE':
            continue
            
        # Handle argval - we only care about integer values for jump targets
        argval = 0
        if hasattr(instr, 'argval'):
            if isinstance(instr.argval, int):
                argval = instr.argval
            elif instr.opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE', 'JUMP_FORWARD', 'JUMP_BACKWARD'):
                # For jumps, argval should be the target offset
                argval = instr.argval if isinstance(instr.argval, int) else 0
        
        instructions.append({
            'opcode': instr.opcode,
            'arg': instr.arg if instr.arg is not None else 0,
            'argval': argval,
            'offset': instr.offset
        })
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

def _create_jit_wrapper(func, opt_level, vectorize, inline, parallel, lazy, mode='auto'):
    """Create a JIT-compiled wrapper for the given function."""
    import warnings
    
    # Bug #5 Fix: Detect generators/coroutines early and return original function
    if _is_generator_or_coroutine(func):
        warnings.warn(
            f"Generator/coroutine function '{func.__name__}' cannot be JIT compiled. "
            f"The @jit decorator has no effect on this function.",
            RuntimeWarning,
            stacklevel=3
        )
        return func
    
    # Check bytecode for unsupported opcodes (generators, exceptions)
    unsupported = _has_unsupported_opcodes(func)
    if unsupported == 'generator':
        warnings.warn(
            f"Function '{func.__name__}' contains generator/async opcodes. "
            f"The @jit decorator has no effect on this function.",
            RuntimeWarning,
            stacklevel=3
        )
        return func
    elif unsupported == 'exception':
        # Bug #3 Fix: Detect exception handling and skip JIT compilation
        warnings.warn(
            f"Function '{func.__name__}' uses try/except/raise which is not yet supported. "
            f"The @jit decorator has no effect on this function.",
            RuntimeWarning,
            stacklevel=3
        )
        return func
    
    jit_instance = JIT()
    jit_instance.set_opt_level(opt_level)
    
    instructions = _extract_bytecode(func)
    constants = _extract_constants(func)
    names = _extract_names(func)
    globals_dict = _extract_globals(func)  # Now returns the dict itself
    builtins_dict = _extract_builtins(func)  # For fallback lookup
    closure_cells = _extract_closure(func)
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
    use_int_mode = (mode == 'int')
    
    compiled_ptr = None
    
    def wrapper(*args, **kwargs):
        nonlocal compiled_ptr
        
        if compiled_ptr is None:
            if use_int_mode:
                # Integer mode - pure native i64 operations
                success = jit_instance.compile_int(instructions, constants, func.__name__, param_count, total_locals)
                if not success:
                    return func(*args, **kwargs)
                compiled_ptr = jit_instance.get_int_callable(func.__name__, param_count)
            else:
                # Object mode - handles Python objects with closure support
                # Bug #4 Fix: Pass globals_dict and builtins_dict for runtime lookup
                success = jit_instance.compile(instructions, constants, names, globals_dict, builtins_dict, closure_cells, func.__name__, param_count, total_locals, nlocals)
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
    wrapper._mode = 'int' if use_int_mode else 'object'
    return wrapper
