# JustJIT Future Roadmap

This document outlines the planned evolution of JustJIT from a Python bytecode JIT to a full-featured, GIL-free parallel execution engine.

## Current State (v0.1.x)

- ✅ Python 3.13 bytecode → LLVM IR → native code
- ✅ Object mode (full Python semantics)
- ✅ Integer mode (native int operations)
- ✅ Generator/coroutine support
- ✅ Cross-platform wheels (Linux, macOS, Windows)

---

## Phase 1: Box/Unbox Type System

**Goal:** Bidirectional conversion between Python objects and native types.

```python
@justjit.jit
def compute(x: int, y: float) -> float:
    # x unboxed to i64, y to f64
    result = x * y + 2.5
    # result boxed back to Python float
    return result
```

**Implementation:**
- Type inference from annotations and literals
- `unbox_int()`, `unbox_float()`, `unbox_str()` helpers
- `box_int()`, `box_float()`, `box_str()` for return values
- Stack allocation for small types, heap for large

**Benefits:**
- 10-100x speedup for numeric code
- Foundation for GIL-free execution

---

## Phase 2: Inline C with libclang

**Goal:** Embed C code directly in Python with full interop.

```python
@justjit.jit
def matrix_multiply(a, b, n: int):
    justjit.inline_c("""
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i*n + k] * b[k*n + j];
                }
                result[i*n + j] = sum;
            }
        }
    """)
    return result
```

**Implementation:**
- Parse C with libclang
- Generate LLVM IR from Clang AST
- RAII wrappers for resource management
- Compile to `.bc` (LLVM bitcode)
- Embed bitcode in JIT module

**Python ↔ C Interop:**
```python
# Python variables accessible in C blocks
x = 42
justjit.inline_c("printf('x = %d\\n', x);")

# C results accessible in Python
justjit.inline_c("int result = compute_something();")
print(result)  # Works!
```

---

## Phase 3: True Parallelism & GIL-Free Execution

**Goal:** Multi-threaded execution without GIL constraints.

```python
@justjit.jit(parallel=True)
def parallel_sum(arr: list[int]) -> int:
    # Automatically parallelized across CPU cores
    total = 0
    for x in arr:
        total += x
    return total
```

**Implementation:**
1. **Unbox data** before parallel region
2. **Release GIL** (`Py_BEGIN_ALLOW_THREADS`)
3. **Execute in parallel** (OpenMP or manual threads)
4. **Reacquire GIL** (`Py_END_ALLOW_THREADS`)  
5. **Box result** back to Python

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│                Python Thread (GIL)              │
│  unbox(data) → native arrays                    │
└─────────────────────────────────────────────────┘
         │ Py_BEGIN_ALLOW_THREADS
         ▼
┌─────────────────────────────────────────────────┐
│           Parallel Region (No GIL)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Thread 1 │  │ Thread 2 │  │ Thread N │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
         │ Py_END_ALLOW_THREADS
         ▼
┌─────────────────────────────────────────────────┐
│                Python Thread (GIL)              │
│  box(result) → Python object                    │
└─────────────────────────────────────────────────┘
```

**Constraints:**
- No Python object access during parallel region
- All data must be unboxed before parallelism
- Only native types in hot loops

---

## Implementation Priority

| Phase | Feature | Effort | Impact |
|-------|---------|--------|--------|
| 1 | Box/Unbox | Medium | High |
| 2 | Inline C | High | High |
| 3 | Parallelism | High | Very High |

---

## Technical Requirements

### Phase 1 Dependencies
- Type annotation parsing
- Native type LLVM IR generation

### Phase 2 Dependencies
- libclang integration
- Clang AST → LLVM IR
- LLVM bitcode embedding

### Phase 3 Dependencies
- Thread pool implementation
- Work-stealing scheduler (optional)
- OpenMP integration (optional)

---

## Long-Term Vision

JustJIT aims to become a **Mojo-like** execution engine for Python:
- Python syntax, native performance
- Seamless C/C++ interop
- True multi-core parallelism
- Zero-copy data sharing
- GIL-free hot paths
