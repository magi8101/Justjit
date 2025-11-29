import os
import sys
import dis
import types

# Now import the C++ extension module
from ._core import JIT

__version__ = "0.1.0"
__all__ = ["JIT", "jit"]


def jit(func=None, *, opt_level=3, vectorize=True, inline=True, parallel=False, lazy=False):
    """
    JIT compile a Python function for aggressive performance optimization.
    
    Args:
        func: The function to compile (when used without parentheses)
        opt_level: LLVM optimization level (0-3, default 3 for maximum performance)
        vectorize: Enable loop vectorization (default True)
        inline: Enable function inlining (default True)
        parallel: Enable parallelization (default False)
        lazy: Delay compilation until first call (default False)
    
    Example:
        @jit
        def add(a, b):
            return a + b
        
        @jit(opt_level=2, vectorize=True)
        def mul(a, b):
            return a * b
    """
    if func is None:
        def decorator(f):
            return _create_jit_wrapper(f, opt_level, vectorize, inline, parallel, lazy)
        return decorator
    return _create_jit_wrapper(func, opt_level, vectorize, inline, parallel, lazy)


def _extract_bytecode(func):
    """Extract bytecode instructions from a Python function."""
    instructions = []
    for instr in dis.get_instructions(func):
        instructions.append({
            'opcode': instr.opcode,
            'arg': instr.arg if instr.arg is not None else 0,
            'argval': instr.argval if hasattr(instr, 'argval') and isinstance(instr.argval, int) else 0,
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
    """Extract global objects referenced by the function."""
    globals_dict = func.__globals__
    names = func.__code__.co_names
    global_values = []
    for name in names:
        if name in globals_dict:
            global_values.append(globals_dict[name])
        else:
            global_values.append(None)  # Will need runtime lookup
    return global_values

def _create_jit_wrapper(func, opt_level, vectorize, inline, parallel, lazy):
    """Create a JIT-compiled wrapper for the given function."""
    jit_instance = JIT()
    jit_instance.set_opt_level(opt_level)
    
    instructions = _extract_bytecode(func)
    constants = _extract_constants(func)
    names = _extract_names(func)
    globals_values = _extract_globals(func)
    param_count = func.__code__.co_argcount
    total_locals = func.__code__.co_nlocals
    
    compiled_ptr = None
    
    def wrapper(*args, **kwargs):
        nonlocal compiled_ptr
        
        if compiled_ptr is None:
            success = jit_instance.compile(instructions, constants, names, globals_values, func.__name__, param_count, total_locals)
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
    return wrapper
