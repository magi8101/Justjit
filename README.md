<div align="center">
  <img src="assets/logo.png" alt="JustJIT Logo" width="200"/>
</div>

# JustJIT

A high-performance Just-In-Time compiler for Python that leverages LLVM's ORC JIT infrastructure to compile Python bytecode to native machine code at runtime.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LLVM 18](https://img.shields.io/badge/LLVM-18.1.8-orange.svg)](https://llvm.org/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Supported Python Features](#supported-python-features)
- [Generator and Coroutine Support](#generator-and-coroutine-support)
- [Control Flow Graph (CFG) Implementation](#control-flow-graph-cfg-implementation)
- [Exception Handling](#exception-handling)
- [Current Limitations](#current-limitations)
- [Building from Source](#building-from-source)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

JustJIT transforms Python functions into optimized native machine code using LLVM. Unlike traditional interpreters that execute bytecode instruction-by-instruction, JustJIT analyzes Python bytecode and generates equivalent LLVM IR, which is then optimized and compiled to native code.

### How It Works

1. **Bytecode Extraction**: Parse Python function bytecode using the `dis` module
2. **IR Generation**: Translate each opcode to LLVM IR, maintaining Python semantics
3. **Optimization**: Apply LLVM optimization passes (O0-O3)
4. **Compilation**: JIT compile to native machine code
5. **Execution**: Call the native function with Python object arguments

---

## Features

### Core Features
- **Python 3.10+ Bytecode Support**: Full support for Python 3.10-3.13 bytecode formats
- **LLVM ORC JIT v2**: Modern JIT infrastructure with lazy compilation support
- **Multiple Compilation Modes**:
  - `object` mode: Full Python object semantics
  - `int` mode: Pure integer arithmetic (10x faster for numeric code)
  - `auto` mode: Automatic detection based on function analysis

### Advanced Features
- **Generator Support**: Full state machine implementation for `yield`
- **Coroutine Support**: Async/await with proper awaitable protocol
- **Exception Handling**: Python 3.11+ exception table parsing
- **Pattern Matching**: `match`/`case` statements with CFG analysis
- **Closure Support**: Captured variables from enclosing scopes
- **Comprehensions**: List, dict, and set comprehensions

---

## Installation

### From PyPI (Recommended)

```bash
pip install justjit
```

### From Source

See [Building from Source](#building-from-source) for detailed instructions.

---

## Quick Start

### Basic Usage

```python
from justjit import jit

@jit
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

result = fibonacci(40)  # Runs as native code
```

### Generator Functions

```python
from justjit import jit

@jit
def count_up(n):
    i = 0
    while i < n:
        yield i
        i += 1

for value in count_up(10):
    print(value)
```

### Async Functions

```python
from justjit import jit
import asyncio

@jit
async def fetch_data():
    await asyncio.sleep(0.1)
    return "data"

result = asyncio.run(fetch_data())
```

### Integer Mode (Maximum Performance)

```python
from justjit import jit

@jit(mode='int')
def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total

# Pure LLVM i64 arithmetic - no Python object overhead
result = sum_range(1000000)
```

---

## Architecture

### Project Structure

```
justjit/
├── src/
│   ├── jit_core.cpp       # Core JIT engine (~12,000 lines)
│   │                      # - Opcode handlers for all Python bytecode
│   │                      # - Generator state machine compiler
│   │                      # - Coroutine compiler with await handling
│   │                      # - CFG analysis and PHI node generation
│   │                      # - Exception table parsing and handling
│   ├── jit_core.h         # JIT engine header
│   │                      # - JITGeneratorObject and JITCoroutineObject types
│   │                      # - BasicBlockInfo and CFGStackState for CFG
│   │                      # - Python C API function declarations
│   ├── bindings.cpp       # nanobind Python bindings
│   ├── opcodes.h          # Python 3.13 opcode definitions
│   └── justjit/
│       └── __init__.py    # Python interface (~800 lines)
│                          # - @jit decorator
│                          # - Bytecode extraction
│                          # - Exception table parsing
│                          # - Fallback handling
├── docs/
│   ├── OPCODES_REFERENCE.md      # Detailed opcode documentation
│   └── ASYNC_IMPLEMENTATION_RESEARCH.md
├── CMakeLists.txt         # Build configuration
├── pyproject.toml         # Python package metadata
├── OPCODE_STATUS.md       # Implementation status tracker
└── CFG_IMPLEMENTATION_PLAN.md    # CFG architecture design
```

### Core Components

| Component | Description |
|-----------|-------------|
| **JITCore** | Main compiler class managing LLVM context, module, and JIT engine |
| **compile_function** | Compiles regular Python functions to native code |
| **compile_generator** | Transforms generators into state machine step functions |
| **compile_coroutine** | Handles async functions with await/send protocol |
| **CFG Analysis** | Control flow graph with PHI nodes for SSA form |

---

## Supported Python Features

### Fully Implemented (95+ opcodes)

#### Control Flow
- `if`/`elif`/`else` statements
- `for` loops with `range()`, lists, tuples
- `while` loops
- `break`/`continue`
- `return` statements

#### Data Types
- Integers, floats, strings, booleans, None
- Lists, tuples, dicts, sets
- Slicing (`x[1:3]`, `x[::2]`)
- Comprehensions (list, dict, set)

#### Operations
- All arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `@`
- All bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
- All comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Membership: `in`, `not in`
- Identity: `is`, `is not`
- Boolean: `and`, `or`, `not`
- Augmented assignment: `+=`, `-=`, `*=`, etc.

#### Functions
- Function calls with positional and keyword arguments
- `*args` and `**kwargs`
- Default arguments
- Closures and nested functions
- Lambda expressions (via nested function)

#### Classes
- Attribute access (`obj.attr`)
- Method calls
- `super()` calls
- `__build_class__` for class creation

#### Exception Handling
- `try`/`except`/`finally`
- `raise` with optional cause
- Exception type matching
- Stack unwinding with proper cleanup

#### Pattern Matching (Python 3.10+)
- `match`/`case` statements
- Sequence patterns: `case [a, b, c]`
- Mapping patterns: `case {"key": value}`
- Class patterns: `case Point(x, y)`
- Wildcard: `case _`
- Guards: `case x if x > 0`

#### Imports
- `import module`
- `from module import name`
- `from module import *`

#### String Formatting
- f-strings: `f"Hello, {name}!"`
- Format specs: `f"{value:.2f}"`
- Conversions: `!r`, `!s`, `!a`

---

## Generator and Coroutine Support

### Generator Implementation

JustJIT compiles generators into **state machine step functions**. Each `yield` creates a suspension point with a unique state ID.

```cpp
// Generator step function signature
PyObject* step_func(int32_t* state, PyObject** locals, PyObject* sent_value);

// State values:
//   0 = initial (not started)
//   1..N = resume points after each yield
//   -1 = completed (returned)
//   -2 = error occurred
```

#### Generator Object Structure

```cpp
struct JITGeneratorObject {
    PyObject_HEAD
    int32_t state;              // Current suspension state
    PyObject** locals;          // Persistent local variables
    Py_ssize_t num_locals;      // Including stack spill slots
    GeneratorStepFunc step_func; // Compiled native function
    PyObject* name;             // For __name__
    PyObject* qualname;         // For __qualname__
};
```

#### How Generator Compilation Works

1. **State Identification**: Each `YIELD_VALUE` opcode gets a unique state ID
2. **State Dispatch**: Entry creates a switch on `*state` to jump to correct resume point
3. **Stack Persistence**: Values are spilled to persistent slots before yield
4. **Resume Restoration**: Stack values are restored from slots after resume

### Coroutine Implementation

Coroutines extend generators with the awaitable protocol:

```cpp
struct JITCoroutineObject {
    PyObject_HEAD
    int32_t state;              // Suspension state
    PyObject** locals;          // Persistent locals
    Py_ssize_t num_locals;
    GeneratorStepFunc step_func;
    PyObject* name;
    PyObject* qualname;
    PyObject* awaiting;         // Currently awaited object
};
```

#### Await Handling

The `SEND` opcode handles await delegation:

1. `GET_AWAITABLE`: Get iterator from awaited object
2. `SEND None`: Start iteration
3. `SEND value`: Forward sent values to sub-iterator
4. `END_SEND`: Handle completion, extract return value

---

## Control Flow Graph (CFG) Implementation

JustJIT implements proper SSA (Static Single Assignment) form using CFG analysis and PHI nodes for complex control flow patterns.

### Why CFG is Needed

Python bytecode simulates a value stack. When control flow merges (e.g., after `if`/`else`), different paths may have pushed different values. LLVM requires each value to have a single definition point, so we use PHI nodes to merge values from different paths.

### CFG Data Structures

```cpp
struct BasicBlockInfo {
    int start_offset;                  // Block start
    int end_offset;                    // Block end
    std::vector<int> predecessors;     // Incoming edges
    std::vector<int> successors;       // Outgoing edges
    int stack_depth_at_entry;          // Expected stack depth
    bool is_exception_handler;         // Handler block flag
    bool needs_phi_nodes;              // Multiple predecessors
    llvm::BasicBlock* llvm_block;      // LLVM representation
};
```

### PHI Node Generation

When compiling pattern matching (`match`/`case`):

1. **Block Analysis**: Identify all case blocks and their targets
2. **Stack Tracking**: Track stack state at each merge point  
3. **PHI Creation**: At merge blocks, create PHI nodes for each stack slot
4. **Value Wiring**: Connect incoming values from each predecessor

---

## Exception Handling

JustJIT parses Python 3.11+ exception tables to handle try/except blocks correctly.

### Exception Table Format

```python
# Exception table entry structure
{
    "start": 10,    # Protected range start (byte offset)
    "end": 50,      # Protected range end
    "target": 100,  # Handler offset (PUSH_EXC_INFO)
    "depth": 2,     # Stack depth to unwind to
    "lasti": False  # Whether to push last instruction
}
```

### Error Checking Pattern

Every Python C API call that can fail is wrapped with error checking:

```cpp
// After each potentially-failing call
auto check_error_and_branch = [&](llvm::Value* result, ...) {
    // Check if result is NULL
    auto is_null = builder->CreateIsNull(result);
    
    // If NULL, unwind stack and jump to handler
    builder->CreateCondBr(is_null, error_block, continue_block);
};
```

---

## Current Limitations

### Not Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| **Async Generators** | ❌ | `async def` with `yield` |
| **`async for`** | ❌ | Needs `GET_AITER`/`GET_ANEXT` |
| **`async with`** | ⚠️ | Basic support, not fully tested |
| **Generator Comprehensions in Generators** | ❌ | Complex stack handling across yields |

### Known Issues

1. **Generator + List Comprehension**: When a generator contains a list comprehension, the stack handling across `YIELD_VALUE` is incomplete. The list builds correctly, but stack spilling/restoration needs refinement.

2. **Deeply Nested Async**: Chains of `await` expressions with complex nesting may have edge cases.

### Fallback Behavior

When JustJIT encounters unsupported opcodes or patterns, it automatically falls back to Python execution with a warning:

```python
RuntimeWarning: Generator 'my_gen' uses opcodes not yet supported by JIT.
Using Python implementation (no performance impact for generators).
```

---

## Building from Source

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CMake | 3.20+ |
| LLVM | 18.1.8+ |
| C++ Compiler | C++17 support |
| nanobind | 2.0.0+ |

### Windows Build

```powershell
# Set LLVM path
$env:LLVM_DIR = "C:\path\to\llvm\build\lib\cmake\llvm"

# Build and install
pip install -e . --no-build-isolation
```

### Linux Build

```bash
# Set LLVM path
export LLVM_DIR=/path/to/llvm/build/lib/cmake/llvm

# Build and install
pip install -e . --no-build-isolation
```

### macOS Build

```bash
# With Homebrew LLVM
export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# Build and install  
pip install -e . --no-build-isolation
```

### CMake Configuration

```bash
cmake -B build \
    -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE=$(which python)
```

---

## API Reference

### `@jit` Decorator

```python
@jit(
    opt_level=3,      # LLVM optimization level (0-3)
    vectorize=True,   # Enable loop vectorization
    inline=True,      # Enable function inlining
    parallel=False,   # Enable parallelization (experimental)
    lazy=False,       # Delay compilation until first call
    mode='auto'       # 'auto', 'object', or 'int'
)
def my_function(...):
    ...
```

### `dump_ir()` Function

```python
from justjit import dump_ir

@dump_ir
def my_func(x):
    return x + 1

# Prints the generated LLVM IR
my_func(5)
```

### Direct API Access

```python
from justjit import JIT

jit = JIT()
jit.set_opt_level(3)
success = jit.compile_function(
    instructions, constants, names,
    globals_dict, builtins_dict,
    closure_cells, exception_table,
    "my_func", param_count=2, total_locals=5, nlocals=5
)
if success:
    callable = jit.get_callable("my_func", param_count=2)
    result = callable(arg1, arg2)
```

---

## Contributing

Contributions are welcome! Please ensure:

1. **Code Style**: Follow existing patterns in the codebase
2. **Testing**: Test with various Python code patterns
3. **Documentation**: Update docs for new features
4. **Compatibility**: Maintain Python 3.10+ and cross-platform support

### Development Workflow

```bash
# Clone the repository
git clone https://github.com/magi8101/justjit.git
cd justjit

# Create development environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install in development mode
pip install -e . --no-build-isolation

# Run tests
python -c "from justjit import jit; @jit\ndef f(x): return x*2\nprint(f(21))"
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [LLVM Project](https://llvm.org/) - Compiler infrastructure
- [Python](https://www.python.org/) - The language we're optimizing
- [nanobind](https://github.com/wjakob/nanobind) - C++/Python bindings

---

## Links

- **Repository**: [github.com/magi8101/justjit](https://github.com/magi8101/justjit)
- **Issues**: [github.com/magi8101/justjit/issues](https://github.com/magi8101/justjit/issues)

