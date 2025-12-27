# JustJIT v0.1.4: 11 Native Modes for Python

**Title:** JustJIT v0.1.4: 11 Native Modes for Python

**Slug:** justjit-v014-11-native-modes

**Excerpt:** We just shipped 11 native data types for JustJIT—from SIMD vectors to nullable types. Here's what's new and how it works.

**Tags:** python, jit, llvm, performance, compiler, optimization

---

## Content (copy below)

---

Python is great for productivity, but let's be honest—it's not winning any speed contests. That's why I've been building **JustJIT**, a JIT compiler that takes Python bytecode and compiles it straight to native machine code using LLVM.

Today, I'm releasing **v0.1.4** with **11 native modes**. Let me show you what that means.

---

## The Problem with Python Performance

When you run a Python function, here's what happens:

1. Python parses your code into bytecode
2. The interpreter reads each bytecode instruction
3. Each instruction becomes multiple CPU operations
4. Objects are boxed, unboxed, type-checked... repeatedly

For a simple `a + b`, that's dozens of CPU cycles just to add two numbers.

**JustJIT changes this.** We compile your Python directly to LLVM IR, then to native x86/ARM machine code. No interpreter loop. No boxing. Just raw CPU instructions.

---

## What's New in v0.1.4

### Complex Numbers (complex64)

We now support single-precision complex arithmetic:

```python
@justjit.jit(mode='complex64')
def complex_multiply(a, b):
    return a * b

result = complex_multiply(3+4j, 1+2j)  # (-5+10j)
```

Under the hood, this compiles to a `{float, float}` struct with native floating-point operations. No Python object overhead.

### Nullable Types (optional_f64)

This is a big one. You can now handle `None` values natively:

```python
@justjit.jit(mode='optional_f64')
def safe_divide(a, b):
    if b == 0:
        return None
    return a / b

safe_divide(10.0, 0.0)  # Returns None
safe_divide(10.0, 2.0)  # Returns 5.0
```

The LLVM representation is `{i64, f64}`—a flag indicating if the value exists, plus the actual value. If either operand is `None`, the result is `None`. No exceptions, no try/catch, just clean nullable semantics.

### SIMD Vectors (vec4f, vec8i)

For the performance enthusiasts:

```python
@justjit.jit(mode='vec4f')
def vec_add(a, b):
    return a + b

# Operates on 4 floats simultaneously using SSE
```

This generates actual SIMD instructions (`<4 x float>` in LLVM). Perfect for graphics, physics, or any workload that benefits from parallelism.

---

## All 11 Modes

Here's the complete list:

| Mode | Type | Use Case |
|------|------|----------|
| `int` | i64 | Integer math, loops |
| `float` | f64 | Floating-point math |
| `bool` | i1 | Boolean logic |
| `int32` | i32 | C interop, memory efficiency |
| `float32` | f32 | ML, SIMD preparation |
| `complex128` | {f64, f64} | Complex math (double precision) |
| `complex64` | {f32, f32} | Complex math (single precision) |
| `ptr` | pointer | Direct array/buffer access |
| `vec4f` | <4 x f32> | SSE SIMD operations |
| `vec8i` | <8 x i32> | AVX SIMD operations |
| `optional_f64` | {i64, f64} | Nullable floats |

Each mode generates clean, type-specific LLVM IR.

---

## Real Performance Numbers

Let's look at actual benchmarks:

### Simple Addition (1M calls)

```
CPython:  102 ms
JustJIT:  355 ms
```

Wait, JustJIT is *slower*? Yes—for trivial functions, the Python↔C call overhead dominates. The function itself is fast, but crossing the boundary isn't free.

### Loop Sum (10M iterations)

```
CPython:  440 ms
JustJIT:  0.01 ms
Speedup:  44,000x
```

**This** is where JustJIT shines. The entire loop runs in native code—no interpreter, no object allocation, no type checking. Just a tight machine code loop.

The takeaway: JustJIT is best for **compute-heavy functions**, not tiny utilities.

---

## Under the Hood

Here's what happens when you decorate a function:

```python
@justjit.jit(mode='float')
def add(a, b):
    return a + b
```

**Python bytecode:**
```
LOAD_FAST_LOAD_FAST (a, b)
BINARY_OP (+)
RETURN_VALUE
```

**JustJIT LLVM IR:**
```llvm
define double @add(double %0, double %1) {
entry:
  %fadd = fadd double %0, %1
  ret double %fadd
}
```

Three bytecode instructions become one CPU instruction (`fadd`). That's the power of compilation.

---

## Installation

```bash
pip install justjit==0.1.4
```

Works on:
- **Windows** (x64)
- **macOS** (Apple Silicon)
- **Linux** (x64)

The wheel bundles LLVM 18, so there's no external dependency. Just install and use.

---

## What's Coming Next

- **More optional types** (optional_i64, optional_f32)
- **Async/await support** for coroutines
- **Better control flow** (if/else, while)
- **Function inlining** for even more speed

---

## Try It Out

JustJIT is open source and actively developed. If you're working with numerical Python and want more speed without rewriting everything in C, give it a shot.

```python
import justjit

@justjit.jit(mode='int')
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fibonacci(1000000)  # Blazingly fast
```

Check out the repo: [github.com/magi8101/JustJIT](https://github.com/magi8101/JustJIT)

---

*Built with LLVM 18, nanobind, and an unhealthy obsession with compiler optimization.*
