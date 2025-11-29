<div align="center">
  <img src="assets/logo.png" alt="JustJIT Logo" width="200"/>
</div>

# JustJIT

Fast Python JIT compiler using LLVM ORC for aggressive runtime optimization.

## Overview

JustJIT is a high-performance Just-In-Time compiler for Python that leverages LLVM's ORC JIT infrastructure to compile Python bytecode to native machine code at runtime. It analyzes Python function bytecode and generates optimized LLVM IR, enabling significant performance improvements for compute-intensive workloads.

## Features

- Python bytecode to LLVM IR compilation
- LLVM optimization levels 0-3 with aggressive optimizations
- Native callable generation via nanobind
- Support for Python 3.8-3.13
- Cross-platform support (Windows, Linux, macOS)
- Zero-overhead C++ integration

## Installation

### Prerequisites

- Python 3.8 or higher
- CMake 3.20 or higher
- LLVM 22.0.0 or compatible version
- C++17 compatible compiler
- nanobind 2.0.0 or higher

### Build from Source

```bash
git clone https://github.com/magi8101/justjit.git
cd justjit
pip install -e .
```

### LLVM Configuration

Set the LLVM installation path during build:

```bash
cmake -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm ..
```

## Usage

### Basic Usage

```python
from justjit import jit

@jit
def add(a, b):
    return a + b

result = add(5, 3)
```

### Custom Optimization Levels

```python
from justjit import jit

@jit(opt_level=3, vectorize=True)
def compute(x, y):
    result = 0
    for i in range(1000):
        result += x * y
    return result
```

### Decorator Options

- `opt_level`: LLVM optimization level (0-3, default: 3)
- `vectorize`: Enable loop vectorization (default: True)
- `inline`: Enable function inlining (default: True)
- `parallel`: Enable parallelization (default: False)
- `lazy`: Delay compilation until first call (default: False)

## Architecture

### Core Components

- **JIT Core**: LLVM ORC JIT engine wrapper with optimization pipeline
- **Bytecode Compiler**: Python bytecode to LLVM IR translator
- **Python Bindings**: nanobind-based Python interface
- **Optimization Pipeline**: Multi-pass LLVM optimization passes

### Supported Python Operations

- Arithmetic operations (add, subtract, multiply, divide, modulo)
- Comparison operations
- Variable loading and storing
- Function calls
- List and tuple operations
- Attribute access
- Control flow (jumps, loops)

## Development

### Project Structure

```
justjit/
├── src/
│   ├── jit_core.cpp       # Core JIT compilation engine
│   ├── jit_core.h         # JIT engine header
│   ├── bindings.cpp       # Python bindings
│   └── justjit/
│       └── __init__.py    # Python package interface
├── docs/
│   └── OPCODES_REFERENCE.md
├── CMakeLists.txt         # Build configuration
├── pyproject.toml         # Python package metadata
└── ERRORS.md              # Build troubleshooting guide
```

### Build System

Uses scikit-build-core for building the C++ extension with CMake integration.

### Testing

```bash
python -m pytest tests/
```

## Current Status

**Development Status**: Alpha

The project is under active development. See `ERRORS.md` for current build status and known issues.

## Contributing

Contributions are welcome. Please ensure:

- Code follows C++17 and Python 3.8+ standards
- All tests pass before submitting PRs
- Documentation is updated for new features
- Cross-platform compatibility is maintained

## License

MIT License

## Authors

JustJIT Contributors

## Links

- [GitHub Repository](https://github.com/magi8101/justjit)
- [Issue Tracker](https://github.com/magi8101/justjit/issues)

## Technical Details

### LLVM Integration

JustJIT uses LLVM ORC v2 JIT API for dynamic compilation:

- ThreadSafeContext for concurrent compilation
- LLJIT builder for JIT stack setup
- Symbol resolution via ExecutionSession
- Native target initialization

### Python C API Integration

Direct integration with CPython internals:

- PyObject manipulation via LLVM function calls
- Reference counting management
- Native type conversions
- Exception handling

### Performance Characteristics

- First call: Compilation overhead + execution
- Subsequent calls: Native execution speed
- Optimization level 3: Maximum performance, higher compile time
- Optimization level 0: Fastest compilation, basic optimizations

## Requirements

### Runtime Dependencies

- numpy >= 1.20.0

### Build Dependencies

- scikit-build-core >= 0.4.3
- nanobind >= 2.0.0
- LLVM development libraries
- zlib (platform-specific)

### Platform-Specific

**Windows:**
- Visual Studio 2022 Build Tools or equivalent
- Windows SDK

**Linux:**
- GCC 7+ or Clang 10+
- Development headers for Python

**macOS:**
- Xcode Command Line Tools
- macOS 10.14 or higher

## Troubleshooting

See `ERRORS.md` for detailed build error documentation and solutions.

### Common Issues

**LLVM not found:**
```bash
cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..
```

**Python version mismatch:**
Ensure Python development headers match runtime version.

**zlib missing:**
Install zlib development package for your platform.

## Roadmap

- Complete bytecode opcode coverage
- Advanced optimization passes
- Multi-threading support
- Ahead-of-time compilation mode
- Profiling and debugging tools
- Extended type inference
- GPU offloading capabilities

## Acknowledgments

Built with:
- LLVM Project
- Python3
- nanobind
