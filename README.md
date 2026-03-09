<div align="center">
  <img src="assets/logo.png" alt="JustJIT Logo" width="200"/>
</div>

# JustJIT

A Just-In-Time compiler for Python that compiles Python bytecode to native machine code using LLVM.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![LLVM 18+](https://img.shields.io/badge/LLVM-18+-orange.svg)](https://llvm.org/)

---

## Installation

```bash
pip install justjit
```

## Quick Start

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

result = fibonacci(40)  # Compiled to native code
```

### Generator Functions

```python
@jit
def count_up(n):
    i = 0
    while i < n:
        yield i
        i += 1

for value in count_up(10):
    print(value)
```

### Async/Await

```python
import asyncio
from justjit import jit

@jit
async def fetch_data():
    await asyncio.sleep(0.1)
    return "data"

result = asyncio.run(fetch_data())
```

### Exception Handling

```python
@jit
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0

print(safe_divide(10, 2))  # 5.0
print(safe_divide(10, 0))  # 0
```

### Float Mode

```python
@jit(mode='float')
def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return (dx * dx + dy * dy) ** 0.5

print(distance(0.0, 0.0, 3.0, 4.0))  # 5.0
```

### List Comprehensions

```python
@jit
def squares(n):
    return [x * x for x in range(n)]

print(squares(5))  # [0, 1, 4, 9, 16]
```

## Features

- **Object Mode** (default): Full Python semantics via C API calls
- **Native Modes**: `int`, `float`, `bool`, `complex128`, `vec4f`, `vec8i` for specialized workloads
- **Generators**: State machine compilation with `yield` support  
- **Coroutines**: Async/await with awaitable protocol
- **Exception Handling**: Try/except/finally with stack unwinding
- **Inline C/C++**: Compile C code at runtime (requires Clang)

## Compilation Modes

| Mode | Type | Use Case |
|------|------|----------|
| `auto` | PyObject* | Default, full Python compatibility |
| `int` | i64 | Pure integer loops |
| `float` | f64 | Floating-point math |
| `complex128` | {f64, f64} | Complex number operations |
| `vec4f` | <4 x f32> | SSE SIMD (4 floats) |
| `vec8i` | <8 x i32> | AVX SIMD (8 ints) |

```python
@jit(mode='int')
def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total
```

## Inline C Compiler

```python
from justjit import inline_c

result = inline_c("""
double square(double x) {
    return x * x;
}
""")

print(result['square'](4.0))  # 16.0
```

## How It Works

### Compilation Pipeline

```
Python Function → Bytecode → CFG Analysis → LLVM IR → Native Code
```

1. **Bytecode Extraction**: Uses Python's `dis` module to extract bytecode instructions
2. **CFG Construction**: Builds control flow graph, identifies basic blocks and jump targets
3. **Stack Depth Analysis**: Dataflow analysis to compute stack depth at each instruction
4. **IR Generation**: Translates each opcode to LLVM IR
5. **Optimization**: Applies LLVM optimization passes (O0-O3)
6. **JIT Compilation**: LLVM ORC JIT compiles to native machine code

### Object Mode Internals

In object mode, Python operations are compiled to C API calls:

```python
# Python code
result = a + b
```

```llvm
; Generated LLVM IR
%result = call ptr @PyNumber_Add(ptr %a, ptr %b)
call void @Py_DECREF(ptr %a)
call void @Py_DECREF(ptr %b)
```

Every API call is wrapped with NULL-checking for exception handling.

### Native Mode Internals

Native modes bypass Python objects entirely:

```python
@jit(mode='int')
def add(a, b):
    return a + b
```

```llvm
; Generated LLVM IR - pure LLVM i64
define i64 @add(i64 %a, i64 %b) {
  %result = add i64 %a, %b
  ret i64 %result
}
```

### Generator State Machine

Generators compile to step functions with explicit state:

```cpp
// Step function signature
PyObject* step(int32_t* state, PyObject** locals, PyObject* sent);

// States: 0=initial, 1..N=resume points, -1=done, -2=error
```

At each `yield`:
1. Stack values spill to persistent `locals` array
2. State advances to next resume point
3. Function returns yielded value

On resume:
1. Switch dispatches to correct state
2. Stack restores from `locals` array
3. Execution continues after yield

### Coroutine Internals

Coroutines extend generators with awaitable protocol:

```cpp
struct JITCoroutineObject {
    int32_t state;           // Suspension state
    PyObject** locals;       // Persistent variables
    PyObject* awaiting;      // Currently awaited object
    GeneratorStepFunc step;  // Compiled step function
};
```

`await` compiles to:
1. `GET_AWAITABLE` - validate and get iterator
2. `SEND` - delegate to awaited object
3. `END_SEND` - extract return value

### Inline C Compiler

Uses embedded Clang to compile C/C++ at runtime:

```
C Code → Clang Frontend → LLVM IR → Same JIT as Python
```

Python variables are captured as C declarations:
```c
// Python: x = 42, arr = numpy.array([1,2,3])
// Generated C preamble:
long long x = 42LL;
double* arr = (double*)0x7fff12345678ULL;
long long arr_len = 3LL;
```

## Current Status

### Working
- Basic control flow (if/else, for, while)
- Arithmetic and comparison operations
- Function calls
- List/dict/set operations  
- Try/except handling
- Simple generators

### Known Limitations
- Native modes (`int`, `float`) only support scalar operations, not array indexing
- Nested list comprehensions in generators may have issues
- Some complex async patterns not fully tested

## Building from Source

Requires: Python 3.10+, LLVM 18+, CMake 3.20+, C++17 compiler

```bash
# Set LLVM path
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm  # Linux/macOS
$env:LLVM_DIR = "C:\path\to\llvm\lib\cmake\llvm"  # Windows

# Build
pip install -e . --no-build-isolation
```

## License

MIT License - see [LICENSE](LICENSE)

## Links
- [Documentation](https://justjit.readthedocs.io)
