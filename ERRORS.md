# Build Errors Documentation

## Current Status

**Build Status:** FAILING  
**Last Attempt:** November 29, 2025  
**Error Type:** C++ Compilation Error

---

## Error Summary

### Primary Error: C2653 - Namespace/Class Name Error

**Location:** `D:\justjit\src\jit_core.cpp` lines 361, 368, 375, 382, 389  
**Error Code:** C2653  
**Description:** "'JITCore': is not a class or namespace name"

### Affected Methods

All callable creation helper methods are failing to compile:

- `JITCore::create_callable_0(uint64_t func_ptr)` - Line 361
- `JITCore::create_callable_1(uint64_t func_ptr)` - Line 368
- `JITCore::create_callable_2(uint64_t func_ptr)` - Line 375
- `JITCore::create_callable_3(uint64_t func_ptr)` - Line 382
- `JITCore::create_callable_4(uint64_t func_ptr)` - Line 389

### Secondary Errors

- **C2059:** syntax error: '}'
- **C2143:** syntax error: missing ';' before '}'

---

## Previous Errors (Resolved)

### 1. DLL Load Failed Error

**Issue:** `ImportError: DLL load failed while importing _core: The specified module could not be found`

**Root Causes Identified:**

- LLVM built without exceptions (`/EHs-c-`) and without RTTI (`/GR-`)
- Missing zlib dependency (required by LLVM)
- Missing additional LLVM libraries (SandboxIR, ObjCARCOpts, CGData, etc.)
- Incorrect library directory paths using `$(Configuration)` variable

**Attempted Solutions:**

- Added `NB_STATIC` flag to nanobind
- Fixed LLVM library paths to use Release explicitly
- Added all required LLVM component libraries manually
- Located and added zlib: `C:/ProgramData/miniconda3/envs/llvm-dev/Library/lib/zlib.lib`
- Added Windows system libraries (psapi, shell32, ole32, uuid, advapi32, ws2_32)

**Status:** Build never completed successfully to test DLL loading

---

### 2. Linker Errors - Unresolved External Symbols

**Multiple Missing LLVM Libraries:**

- LLVMSandboxIR - Added ✓
- LLVMObjCARCOpts - Added ✓
- LLVMCGData - Added ✓
- LLVMLinker - Added ✓
- LLVMFrontendOpenMP - Added ✓
- LLVMAsmPrinter - Added ✓
- LLVMMCDisassembler - Added ✓
- LLVMOffloading - Does not exist in LLVM build ✗
- LLVMHLSL - Does not exist in LLVM build ✗

**Missing External Dependencies:**

- `compress2` (zlib) - Resolved by adding zlib.lib
- `compressBound` (zlib) - Resolved by adding zlib.lib
- `uncompress` (zlib) - Resolved by adding zlib.lib
- `crc32` (zlib) - Resolved by adding zlib.lib

**Status:** Partially resolved, but build progressed to next error

---

### 3. Multiple Definition Error (LNK2005/LNK1169)

**Issue:** Functions defined multiple times

**Affected Functions:**

- `JITCore::create_callable_0`
- `JITCore::create_callable_1`
- `JITCore::create_callable_2`
- `JITCore::create_callable_3`
- `JITCore::create_callable_4`
- `JITCore::get_callable`

**Root Cause:** `bindings.cpp` was including `jit_core.cpp` directly instead of using a header file

**Solution Applied:**

- Created `jit_core.h` header file with class declarations
- Moved implementation to `jit_core.cpp`
- Changed `bindings.cpp` to include `jit_core.h` instead of `jit_core.cpp`

**Status:** Resolved, but introduced new compilation error

---

## Current Investigation

### Hypothesis

The C2653 error suggests the compiler cannot find the `JITCore` class definition at the point where the methods are being implemented. This could be due to:

1. **Namespace Issues:** Missing or incorrect namespace closure
2. **Header Include Issues:** `jit_core.h` not being properly included
3. **Template/Declaration Mismatch:** nanobind types not being recognized
4. **Compilation Order:** Header dependencies not resolved correctly

### Files Structure

```
src/
 jit_core.h       - Class declaration + method signatures
 jit_core.cpp     - Class implementation
 bindings.cpp     - Python bindings (includes jit_core.h)
```

### Key Code Sections

**jit_core.h** - Class declaration includes:

- Constructor/destructor
- Public methods: `set_opt_level`, `get_opt_level`, `get_callable`, `compile_function`, `lookup_symbol`
- Private helper methods: `create_callable_0` through `create_callable_4`
- Private method: `optimize_module`

**jit_core.cpp** - Expected to have:

- All method implementations with proper `JITCore::` prefix
- Proper namespace closure `namespace justjit { ... }`

---

## Build Configuration

### CMake Configuration

- **CMake Version:** 3.20+
- **C++ Standard:** 17
- **Build Type:** Release (forced)
- **Compiler:** MSVC 19.29 (Visual Studio 2022 Build Tools)

### LLVM Configuration

- **Version:** 22.0.0git
- **Location:** `C:/Users/vetri/llvm-project/build`
- **Build Type:** Release
- **Library Directory:** `C:/Users/vetri/llvm-project/build/Release/lib`
- **Include Directories:**
  - `C:/Users/vetri/llvm-project/llvm/include`
  - `C:/Users/vetri/llvm-project/build/include`

### Python Configuration

- **Version:** 3.13.5
- **Distribution:** Miniconda
- **Binding Library:** nanobind 2.9.2
- **Flags:** `NB_STATIC`, `NOMINSIZE`

### External Dependencies

- **zlib:** `C:/ProgramData/miniconda3/envs/llvm-dev/Library/lib/zlib.lib`
- **Windows libs:** psapi, shell32, ole32, uuid, advapi32, ws2_32

---

## Build Warnings (Non-Critical)

### LLVM Warnings

- C4624: Destructor implicitly defined as deleted for various LLVM operator classes
  - `ConcreteOperator<...>` templates
  - Various operator classes (AShrOperator, LShrOperator, GEPOperator, etc.)

### Type Conversion Warnings

- C4244: Possible loss of data converting `uint64_t` to `unsigned int`
  - Location: `std::pair` constructor in MSVC utility header
  - Context: `InstructionWorklist::push` usage

**Note:** These warnings are from LLVM headers and do not prevent compilation

---

## Next Steps / Debugging Plan

1.  Clean workspace - Removed all test/example files
2.  Review jit_core.cpp implementation structure
3.  Verify namespace closure and class definition
4.  Check for syntax errors around line 361
5.  Ensure all method signatures match header declarations
6.  Verify nanobind include path and type availability

---

## Notes

- Build has never completed successfully
- The project uses nanobind for Python bindings (modern alternative to pybind11)
- LLVM is statically linked to avoid DLL dependencies
- ctypes usage was removed in favor of nanobind's native callable wrapping
- All fixes have been applied systematically but compilation still fails at method implementation stage
