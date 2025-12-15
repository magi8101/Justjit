"""
Benchmark comparing JIT-compiled functions vs pure Python.
"""

import time
from justjit import jit


def benchmark(func, args, iterations=100000, name=""):
    """Run a function multiple times and measure execution time."""
    # Warmup
    for _ in range(100):
        func(*args)

    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    per_call_ns = (end - start) / iterations * 1_000_000_000
    print(f"{name:30s} {elapsed_ms:8.2f} ms  ({per_call_ns:6.1f} ns/call)")
    return elapsed_ms


# --- Arithmetic ---
def py_add(a, b):
    return a + b


@jit
def jit_add(a, b):
    return a + b


@jit(mode="int")
def jit_int_add(a, b):
    return a + b


# --- Multiple operations ---
def py_math(a, b):
    c = a + b
    d = a * b
    e = c - d
    return e


@jit
def jit_math(a, b):
    c = a + b
    d = a * b
    e = c - d
    return e


@jit(mode="int")
def jit_int_math(a, b):
    c = a + b
    d = a * b
    e = c - d
    return e


# --- Conditional ---
def py_conditional(x):
    if x > 0:
        return 1
    return 0


@jit
def jit_conditional(x):
    if x > 0:
        return 1
    return 0


@jit(mode="int")
def jit_int_conditional(x):
    if x > 0:
        return 1
    return 0


# --- Comparison ---
def py_compare(a, b):
    if a < b:
        return a
    return b


@jit
def jit_compare(a, b):
    if a < b:
        return a
    return b


@jit(mode="int")
def jit_int_compare(a, b):
    if a < b:
        return a
    return b


# --- Run benchmarks ---
if __name__ == "__main__":
    print("=" * 70)
    print("JustJIT Benchmark - Object Mode vs Integer Mode")
    print("=" * 70)

    # Verify correctness first
    print("\nVerifying correctness...")
    assert py_add(2, 3) == jit_add(2, 3) == jit_int_add(2, 3), "add mismatch"
    assert py_math(5, 3) == jit_math(5, 3) == jit_int_math(5, 3), "math mismatch"
    assert py_conditional(5) == jit_conditional(5) == jit_int_conditional(5), (
        "conditional mismatch"
    )
    assert py_conditional(-5) == jit_conditional(-5) == jit_int_conditional(-5), (
        "conditional mismatch"
    )
    assert py_compare(3, 7) == jit_compare(3, 7) == jit_int_compare(3, 7), (
        "compare mismatch"
    )
    print("All correctness checks passed!\n")

    iterations = 1000000
    print(f"Running {iterations:,} iterations per test...\n")

    print("-" * 70)
    print("Simple Addition (a + b)")
    print("-" * 70)
    py_time = benchmark(py_add, (10, 20), iterations, "Python")
    jit_time = benchmark(jit_add, (10, 20), iterations, "JIT (object mode)")
    int_time = benchmark(jit_int_add, (10, 20), iterations, "JIT (int mode)")
    print(f"{'Object mode speedup:':30s} {py_time / jit_time:.2f}x")
    print(f"{'Integer mode speedup:':30s} {py_time / int_time:.2f}x\n")

    print("-" * 70)
    print("Multiple Operations (a+b, a*b, c-d)")
    print("-" * 70)
    py_time = benchmark(py_math, (5, 3), iterations, "Python")
    jit_time = benchmark(jit_math, (5, 3), iterations, "JIT (object mode)")
    int_time = benchmark(jit_int_math, (5, 3), iterations, "JIT (int mode)")
    print(f"{'Object mode speedup:':30s} {py_time / jit_time:.2f}x")
    print(f"{'Integer mode speedup:':30s} {py_time / int_time:.2f}x\n")

    print("-" * 70)
    print("Conditional (if x > 0)")
    print("-" * 70)
    py_time = benchmark(py_conditional, (5,), iterations, "Python")
    jit_time = benchmark(jit_conditional, (5,), iterations, "JIT (object mode)")
    int_time = benchmark(jit_int_conditional, (5,), iterations, "JIT (int mode)")
    print(f"{'Object mode speedup:':30s} {py_time / jit_time:.2f}x")
    print(f"{'Integer mode speedup:':30s} {py_time / int_time:.2f}x\n")

    print("-" * 70)
    print("Comparison (if a < b)")
    print("-" * 70)
    py_time = benchmark(py_compare, (3, 7), iterations, "Python")
    jit_time = benchmark(jit_compare, (3, 7), iterations, "JIT (object mode)")
    int_time = benchmark(jit_int_compare, (3, 7), iterations, "JIT (int mode)")
    print(f"{'Object mode speedup:':30s} {py_time / jit_time:.2f}x")
    print(f"{'Integer mode speedup:':30s} {py_time / int_time:.2f}x\n")

    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
