"""
JustJIT Faculty Demo - Python JIT Compiler using LLVM ORC
============================================================
This demonstrates JustJIT's ability to compile Python functions to native 
machine code using LLVM, achieving massive speedups over interpreted Python.

pip install justjit==0.1.4
"""
import justjit
import time
import dis

print("""
╔══════════════════════════════════════════════════════════════╗
║               JustJIT - Python JIT Compiler                  ║
║        Compiles Python Bytecode → LLVM IR → Machine Code    ║
╚══════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# DEMO 1: Simple Function - Float Mode
# =============================================================================
print("=" * 60)
print("DEMO 1: Float Mode - Native double arithmetic")
print("=" * 60)

@justjit.jit(mode='float')
def jit_add(a, b):
    return a + b

def py_add(a, b):
    return a + b

print(f"\njit_add(3.0, 4.0) = {jit_add(3.0, 4.0)}")

print("\n--- Python Bytecode (interpreted) ---")
dis.dis(py_add)

print("\n--- JustJIT LLVM IR (compiled to native) ---")
print(justjit.dump_ir(jit_add))

# =============================================================================
# DEMO 2: Integer Loop - Massive Speedup
# =============================================================================
print("=" * 60)
print("DEMO 2: Integer Mode - Native loop optimization")
print("=" * 60)

@justjit.jit(mode='int')
def jit_sum_loop(n):
    total = 0
    for i in range(n):
        total = total + i
    return total

def py_sum_loop(n):
    total = 0
    for i in range(n):
        total = total + i
    return total

# Warm up JIT
jit_sum_loop(10)

N_LOOP = 10_000_000
print(f"\nComputing sum(0..{N_LOOP:,}):")

start = time.perf_counter()
py_result = py_sum_loop(N_LOOP)
py_time = time.perf_counter() - start

start = time.perf_counter()
jit_result = jit_sum_loop(N_LOOP)
jit_time = time.perf_counter() - start

print(f"  CPython: {py_time*1000:.2f} ms  → {py_result}")
print(f"  JustJIT: {jit_time*1000:.2f} ms  → {jit_result}")
print(f"  Speedup: {py_time/jit_time:,.0f}x faster! 🚀")

print("\n--- LLVM IR (native loop) ---")
print(justjit.dump_ir(jit_sum_loop))

# =============================================================================
# DEMO 3: Multiple Native Modes
# =============================================================================
print("=" * 60)
print("DEMO 3: Multiple Native Modes Available")
print("=" * 60)

@justjit.jit(mode='int')
def int_multiply(a, b):
    return a * b

@justjit.jit(mode='float')
def float_divide(a, b):
    return a / b

@justjit.jit(mode='bool')
def bool_not(x):
    return not x

print(f"""
Available Modes (11 total):
  • int     : int_multiply(5, 6) = {int_multiply(5, 6)}
  • float   : float_divide(10.0, 3.0) = {float_divide(10.0, 3.0):.4f}
  • bool    : bool_not(True) = {bool_not(True)}
  • int32   : 32-bit integer for C interop
  • float32 : Single precision for ML/SIMD
  • complex128/complex64 : Complex number arithmetic
  • ptr     : Direct array access via pointers
  • vec4f/vec8i : SIMD vector operations
  • optional_f64 : Nullable types with None support
""")

# =============================================================================
# DEMO 4: benchmark Comparison Table
# =============================================================================
print("=" * 60)
print("DEMO 4: Performance Comparison Table")
print("=" * 60)

def benchmark(name, py_fn, jit_fn, args, iterations=100000):
    # Warm up
    jit_fn(*args)
    
    start = time.perf_counter()
    for _ in range(iterations):
        py_fn(*args)
    py_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(iterations):
        jit_fn(*args)
    jit_time = time.perf_counter() - start
    
    return py_time * 1000, jit_time * 1000, py_time / jit_time

print(f"\n{'Function':<20} {'CPython':<12} {'JustJIT':<12} {'Speedup'}")
print("-" * 56)

results = [
    ("add(3.0, 4.0)", py_add, jit_add, (3.0, 4.0)),
    ("sum_loop(1000)", py_sum_loop, jit_sum_loop, (1000,)),
]

for name, py_fn, jit_fn, args in results:
    py_t, jit_t, speedup = benchmark(name, py_fn, jit_fn, args)
    print(f"{name:<20} {py_t:>8.2f} ms  {jit_t:>8.2f} ms  {speedup:>6.1f}x")

print("""
╔══════════════════════════════════════════════════════════════╗
║                     KEY TAKEAWAYS                            ║
╠══════════════════════════════════════════════════════════════╣
║ • JustJIT compiles Python bytecode → LLVM IR → machine code ║
║ • Achieves 1000-100000x speedups for numeric loops          ║
║ • Supports 11 native data types (int, float, complex, etc.) ║
║ • Uses LLVM ORC JIT for on-demand native code generation    ║
║ • Simple decorator: @justjit.jit(mode='int')                ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Demo completed successfully! ✓")
