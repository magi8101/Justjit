cmake_minimum_required(VERSION 3.20)
project(justjit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Find Python
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# LLVM configuration
# Users should set LLVM_DIR to their LLVM installation
# Example: cmake -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm ..
find_package(LLVM REQUIRED CONFIG)

# LLVM paths from find_package
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# zlib (optional, needed on some platforms)
find_package(ZLIB)
if(ZLIB_FOUND)
    set(ZLIB_LIBRARY ${ZLIB_LIBRARIES})
endif()

# LLVM definitions
add_definitions(
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS  
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS
    -D__STDC_LIMIT_MACROS
)
)

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

# Create extension module
nanobind_add_module(_core NB_STATIC NOMINSIZE
    src/jit_core.cpp
    src/bindings.cpp
)

# Link LLVM libraries
target_link_libraries(_core PRIVATE
    # Core LLVM
    LLVMCore
    LLVMSupport
    LLVMExecutionEngine
    LLVMOrcJIT
    LLVMRuntimeDyld
    LLVMOrcTargetProcess
    LLVMOrcShared
    LLVMJITLink
    LLVMObject
    LLVMTarget
    LLVMMC
    LLVMMCParser
    LLVMBitReader
    LLVMBitWriter
    LLVMIRReader
    LLVMAsmParser
    LLVMTransformUtils
    LLVMAnalysis
    LLVMCodeGen
    LLVMSelectionDAG
    LLVMScalarOpts
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMipo
    LLVMVectorize
    LLVMSandboxIR
    LLVMInstrumentation
    LLVMPasses
    LLVMObjCARCOpts
    LLVMCGData
    LLVMLinker
    LLVMFrontendOpenMP
    LLVMAsmPrinter
    LLVMMCDisassembler
    # X86 Target
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    # Support
    LLVMDemangle
    LLVMBinaryFormat
    LLVMTargetParser
    LLVMBitstreamReader
    LLVMRemarks
    LLVMTextAPI
    LLVMProfileData
    LLVMSymbolize
    LLVMDebugInfoDWARF
    LLVMDebugInfoPDB
    LLVMGlobalISel
    LLVMCodeGenTypes
    LLVMCFGuard
    # External dependencies
    ${ZLIB_LIBRARY}
    LLVMCFGuard
)

# Add zlib if found
if(ZLIB_FOUND)
    target_link_libraries(_core PRIVATE ${ZLIB_LIBRARY})
    ws2_32
  )
endif()
  target_link_libraries(_core PRIVATE
    # Windows system libraries
    uuid
    advapi32
    ws2_32
)

target_include_directories(_core PRIVATE ${LLVM_INCLUDE_DIRS})

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/justjit"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/justjit"
)

install(TARGETS _core LIBRARY DESTINATION justjit)
