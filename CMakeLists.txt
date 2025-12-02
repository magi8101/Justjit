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

# Force static LLVM linking for standalone wheels
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "Link against static LLVM libraries" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

find_package(LLVM REQUIRED CONFIG)

# LLVM paths from find_package
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM library dir: ${LLVM_LIBRARY_DIRS}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# zlib (optional, needed on some platforms)
find_package(ZLIB)

# LLVM definitions
add_definitions(
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS  
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS
    -DUNICODE
    -D_UNICODE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
)

# Create extension module
nanobind_add_module(_core NB_STATIC NOMINSIZE
    src/jit_core.cpp
    src/bindings.cpp
)

# Use LLVM's llvm_map_components_to_libnames to get the right libraries
# This handles version differences and platform specifics automatically
llvm_map_components_to_libnames(LLVM_LIBS
    Core
    Support
    ExecutionEngine
    OrcJIT
    MCJIT
    RuntimeDyld
    native
    Passes
    IRReader
    AsmParser
    Linker
    ipo
    Instrumentation
    ScalarOpts
    InstCombine
    TransformUtils
    Analysis
    Object
    MC
    MCParser
    MCDisassembler
    BitReader
    BitWriter
    Target
    CodeGen
    SelectionDAG
    AsmPrinter
    GlobalISel
    Demangle
    BinaryFormat
    Remarks
    BitstreamReader
    ProfileData
    AggressiveInstCombine
    Vectorize
    CFGuard
)

message(STATUS "LLVM libraries: ${LLVM_LIBS}")

# Link LLVM libraries
target_link_libraries(_core PRIVATE ${LLVM_LIBS})

# Add zlib if found
if(ZLIB_FOUND)
    target_link_libraries(_core PRIVATE ${ZLIB_LIBRARIES})
endif()

# Platform-specific libraries
if(WIN32)
    target_link_libraries(_core PRIVATE
        psapi
        shell32
        ole32
        uuid
        advapi32
        ws2_32
    )
elseif(UNIX AND NOT APPLE)
    # Linux system libraries needed by LLVM
    find_package(Threads REQUIRED)
    target_link_libraries(_core PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
        rt
        m
    )
    # tinfo/ncurses is optional - some LLVM builds need it, some don't
    find_library(TINFO_LIBRARY NAMES tinfo ncurses ncursesw)
    if(TINFO_LIBRARY)
        target_link_libraries(_core PRIVATE ${TINFO_LIBRARY})
    endif()
elseif(APPLE)
    # macOS system libraries
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    target_link_libraries(_core PRIVATE ${COREFOUNDATION_LIBRARY})
endif()

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/justjit"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/justjit"
)

install(TARGETS _core LIBRARY DESTINATION justjit)
