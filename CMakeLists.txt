cmake_minimum_required(VERSION 3.20)
project(justjit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Find Python
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Link against Python3 to resolve symbols like PyRun_StringFlags on macOS
# This is safe because we link PRIVATE and modern CMake handles the details
# link_libraries(Python3::Python) - REMOVED global link, using target specific below

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)



# LLVM configuration
# conda-forge llvmdev provides DYNAMIC libraries (libLLVM.so / LLVM.dll)
# We link dynamically and bundle the shared lib into the wheel

# On Windows, LLVM may have been built with LibXml2 dependency
# Find it before LLVM to satisfy LLVMExports.cmake requirements
if(WIN32)
    find_package(LibXml2 QUIET)
    if(LibXml2_FOUND)
        message(STATUS "Found LibXml2: ${LIBXML2_LIBRARIES}")
    else()
        # Create an empty imported target to satisfy LLVM dependency
        # LLVM may not actually need it at runtime for our use case
        if(NOT TARGET LibXml2::LibXml2)
            add_library(LibXml2::LibXml2 INTERFACE IMPORTED)
            message(STATUS "Created stub LibXml2::LibXml2 target for LLVM dependency")
        endif()
    endif()
endif()

# Set policy to allow missing files in imported targets (e.g., LLVMTestingAnnotations on Alpine)
# This is needed because some LLVM packages don't include all testing libraries
if(POLICY CMP0111)
    cmake_policy(SET CMP0111 OLD)
endif()

find_package(LLVM REQUIRED CONFIG)

# Workaround: Create stub for LLVMTestingAnnotations if missing (Alpine LLVM packaging issue)
if(NOT TARGET LLVMTestingAnnotations)
    add_library(LLVMTestingAnnotations INTERFACE IMPORTED)
    message(STATUS "Created stub LLVMTestingAnnotations target (not provided by LLVM package)")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM library dir: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")

# Include LLVM headers
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Additional Windows-specific definitions
if(WIN32)
    add_definitions(
        -D_CRT_SECURE_NO_DEPRECATE
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_DEPRECATE
        -D_SCL_SECURE_NO_WARNINGS
    )
endif()

# Find Clang for InlineCCompiler support
# Clang is typically installed alongside LLVM
find_package(Clang CONFIG HINTS ${LLVM_DIR}/../clang QUIET)
if(Clang_FOUND)
    message(STATUS "Found Clang - enabling InlineCCompiler")
    add_definitions(-DJUSTJIT_HAS_CLANG=1)
    include_directories(${CLANG_INCLUDE_DIRS})
    # Check for clang-cpp (monolithic shared library, preferred for distro builds)
    if(TARGET clang-cpp)
        set(CLANG_LIBS clang-cpp)
    elseif(TARGET clang::clang-cpp)
        set(CLANG_LIBS clang::clang-cpp)
    elseif(TARGET clangFrontend)
        # Components as non-namespaced targets
        set(CLANG_LIBS
            clangFrontend
            clangCodeGen
            clangSema
            clangAST
            clangParse
            clangLex
            clangAnalysis
            clangEdit
            clangSerialization
            clangBasic
            clangSupport
        )
    elseif(TARGET clang::clangFrontend)
        # Components as namespaced targets (common in some configs)
        set(CLANG_LIBS
            clang::clangFrontend
            clang::clangCodeGen
            clang::clangSema
            clang::clangAST
            clang::clangParse
            clang::clangLex
            clang::clangAnalysis
            clang::clangEdit
            clang::clangSerialization
            clang::clangBasic
            clang::clangSupport
        )
    else()
        # Fallback to manual list (likely to fail if not found above, but tries -l...)
        message(WARNING "Clang targets not found, attempting link via library names")
         set(CLANG_LIBS
            clangFrontend
            clangCodeGen
            clangSema
            clangAST
            clangParse
            clangLex
            clangAnalysis
            clangEdit
            clangSerialization
            clangBasic
            clangSupport
        )
    endif()
else()
    message(STATUS "Clang not found - InlineCCompiler will be disabled")
    set(CLANG_LIBS "")
endif()

# ============================================================================
# Find Clang resource headers and libc++ for embedded stdlib support
# This allows inline C/C++ to compile without system headers (e.g., MSVC)
# ============================================================================
if(Clang_FOUND)
    # Get LLVM major version for path construction
    string(REGEX MATCH "^[0-9]+" LLVM_MAJOR_VERSION "${LLVM_PACKAGE_VERSION}")
    
    # Search paths for Clang resource headers (stddef.h, stdarg.h, stdint.h, etc.)
    # Derive source tree path from LLVM_DIR (which is ${source}/build/lib/cmake/llvm)
    get_filename_component(LLVM_SOURCE_ROOT "${LLVM_DIR}/../../../.." ABSOLUTE)
    set(CLANG_RESOURCE_SEARCH_PATHS
        "${LLVM_LIBRARY_DIR}/clang/${LLVM_MAJOR_VERSION}/include"
        "${CMAKE_PREFIX_PATH}/lib/clang/${LLVM_MAJOR_VERSION}/include"
        "${CMAKE_PREFIX_PATH}/Library/lib/clang/${LLVM_MAJOR_VERSION}/include"
        "/usr/lib64/clang/${LLVM_MAJOR_VERSION}/include"
        "/usr/lib/clang/${LLVM_MAJOR_VERSION}/include"
        "${LLVM_SOURCE_ROOT}/clang/lib/Headers"                        # LLVM source tree build
    )
    
    set(CLANG_RESOURCE_INCLUDE "")
    foreach(path ${CLANG_RESOURCE_SEARCH_PATHS})
        if(EXISTS "${path}/stddef.h")
            set(CLANG_RESOURCE_INCLUDE "${path}")
            message(STATUS "Found Clang resource headers: ${path}")
            break()
        endif()
    endforeach()
    
    # Search paths for libc++ headers
    set(LIBCXX_SEARCH_PATHS
        "${CMAKE_PREFIX_PATH}/include/c++/v1"
        "${CMAKE_PREFIX_PATH}/Library/include/c++/v1"
        "/usr/include/c++/v1"
        "${LLVM_SOURCE_ROOT}/libcxx/include"                           # LLVM source tree build
    )

    
    set(LIBCXX_INCLUDE "")
    foreach(path ${LIBCXX_SEARCH_PATHS})
        # libc++ needs both __config and __config_site (generated during build)
        if(EXISTS "${path}/__config")
            # Check if __config_site exists (required for libc++ to work)
            # In source builds, this file is generated during libc++ configuration
            if(EXISTS "${path}/__config_site")
                set(LIBCXX_INCLUDE "${path}")
                message(STATUS "Found libc++ headers: ${path}")
                break()
            else()
                message(STATUS "libc++ at ${path} missing __config_site (not configured)")
            endif()
        endif()
    endforeach()

    
    if(NOT CLANG_RESOURCE_INCLUDE)
        message(WARNING "Clang resource headers not found - inline C may require system headers")
    endif()
    if(NOT LIBCXX_INCLUDE)
        message(WARNING "libc++ headers not found - inline C++ stdlib may not work")
    endif()
endif()


# Find zlib (required by LLVM)
find_package(ZLIB REQUIRED)


# Create the Python extension module
nanobind_add_module(_core NB_STATIC NOMINSIZE
    src/jit_core.cpp
    src/bindings.cpp
    src/raii_wrapper.cpp
)
target_link_libraries(_core PRIVATE Python::Module)

if(APPLE)
  # Do NOT link against libpython explicitly on macOS, as it causes 'delocate' to bundle 
  # a second copy of the Python library, leading to "initialization of _core did not return an extension module".
  # Instead, use dynamic lookup for undefined symbols (standard for Python extensions).
  target_link_options(_core PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

# Pass embedded header paths to source code via compile definitions
if(CLANG_RESOURCE_INCLUDE)
    target_compile_definitions(_core PRIVATE
        JUSTJIT_CLANG_RESOURCE_DIR="${CLANG_RESOURCE_INCLUDE}")
endif()
if(LIBCXX_INCLUDE)
    target_compile_definitions(_core PRIVATE
        JUSTJIT_LIBCXX_DIR="${LIBCXX_INCLUDE}")
endif()

# Embedded musl libc headers for self-contained inline C (no system dependencies)
set(EMBEDDED_LIBC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/libc-headers")
if(EXISTS "${EMBEDDED_LIBC_DIR}/stdio.h")
    target_compile_definitions(_core PRIVATE
        JUSTJIT_EMBEDDED_LIBC_DIR="${EMBEDDED_LIBC_DIR}")
    message(STATUS "Using embedded musl libc headers: ${EMBEDDED_LIBC_DIR}")
else()
    message(WARNING "Embedded libc headers not found - inline C may require system headers")
endif()

# ============================================================================
# Optional Platform SDK Headers (for platform-specific code like winsock2.h)
# These are detected at build time and used as fallback after embedded musl
# ============================================================================

# Windows SDK (for winsock2.h, windows.h, etc.)
if(WIN32)
    set(WINSDK_INCLUDE "")
    set(MSVC_INCLUDE "")
    
    # Find Windows SDK
    file(GLOB WINSDK_VERSIONS "C:/Program Files (x86)/Windows Kits/10/Include/*")
    if(WINSDK_VERSIONS)
        list(SORT WINSDK_VERSIONS)
        list(REVERSE WINSDK_VERSIONS)
        list(GET WINSDK_VERSIONS 0 WINSDK_LATEST)
        if(EXISTS "${WINSDK_LATEST}/ucrt")
            set(WINSDK_INCLUDE "${WINSDK_LATEST}")
            target_compile_definitions(_core PRIVATE
                JUSTJIT_WINSDK_UCRT_DIR="${WINSDK_INCLUDE}/ucrt"
                JUSTJIT_WINSDK_SHARED_DIR="${WINSDK_INCLUDE}/shared"
                JUSTJIT_WINSDK_UM_DIR="${WINSDK_INCLUDE}/um")
            message(STATUS "Found Windows SDK headers: ${WINSDK_INCLUDE}")
        endif()
    endif()
    
    # Find MSVC headers
    file(GLOB MSVC_VERSIONS "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*")
    if(MSVC_VERSIONS)
        list(SORT MSVC_VERSIONS)
        list(REVERSE MSVC_VERSIONS)
        list(GET MSVC_VERSIONS 0 MSVC_LATEST)
        if(EXISTS "${MSVC_LATEST}/include")
            set(MSVC_INCLUDE "${MSVC_LATEST}/include")
            target_compile_definitions(_core PRIVATE
                JUSTJIT_MSVC_INCLUDE_DIR="${MSVC_INCLUDE}")
            message(STATUS "Found MSVC headers: ${MSVC_INCLUDE}")
        endif()
    endif()
endif()

# Linux system headers (for platform-specific code)
if(UNIX AND NOT APPLE)
    if(EXISTS "/usr/include")
        target_compile_definitions(_core PRIVATE
            JUSTJIT_LINUX_INCLUDE_DIR="/usr/include")
        message(STATUS "Found Linux system headers: /usr/include")
    endif()
endif()

# macOS SDK headers (for platform-specific code)
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE XCRUN_RESULT
    )
    if(XCRUN_RESULT EQUAL 0 AND EXISTS "${MACOS_SDK_PATH}/usr/include")
        target_compile_definitions(_core PRIVATE
            JUSTJIT_MACOS_SDK_DIR="${MACOS_SDK_PATH}/usr/include")
        message(STATUS "Found macOS SDK headers: ${MACOS_SDK_PATH}/usr/include")
    endif()
endif()


# Link against LLVM libraries
# Windows: LLVM-C.dll only exposes C API, so we must link static component libs for C++ API
# Linux/macOS: Can use the unified libLLVM shared library

# Map all required LLVM components to library names
llvm_map_components_to_libnames(LLVM_LIBS
    Core
    Support
    OrcJIT
    ExecutionEngine
    RuntimeDyld
    native
    Passes
    TransformUtils
    Analysis
    Target
    Object
    MC
    IRReader
    InstCombine
    ScalarOpts
    Vectorize
    IPO
    Linker
    BitWriter
    BitReader
    MCParser
    ObjCARCOpts
    AggressiveInstCombine
    CodeGen
    SelectionDAG
    AsmPrinter
    MIRParser
    GlobalISel
    DebugInfoDWARF
    DebugInfoCodeView
    DebugInfoMSF
    DebugInfoPDB
    Symbolize
    Demangle
    TextAPI
    BinaryFormat
    Remarks
    ProfileData
    Coverage
    LTO
    Extensions
    CFGuard
    Coroutines
    JITLink
    OrcTargetProcess
    OrcShared
    WindowsDriver
    WindowsManifest
)

message(STATUS "LLVM libraries to link: ${LLVM_LIBS}")

if(WIN32)
    # Windows: Must use static component libraries for C++ API
    target_link_libraries(_core PRIVATE ${LLVM_LIBS})
    # Link Clang libraries if available
    if(Clang_FOUND)
        target_link_libraries(_core PRIVATE ${CLANG_LIBS})
    endif()
else()
    # Linux/macOS: Try to use the unified shared library first
    find_library(LLVM_SHARED_LIB NAMES LLVM LLVM-${LLVM_VERSION_MAJOR} PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
    if(LLVM_SHARED_LIB)
        message(STATUS "Found LLVM shared library: ${LLVM_SHARED_LIB}")
        target_link_libraries(_core PRIVATE ${LLVM_SHARED_LIB})
    else()
        # Fallback: use component libraries
        target_link_libraries(_core PRIVATE ${LLVM_LIBS})
    endif()
    # Link Clang libraries if available
    if(Clang_FOUND)
        target_link_libraries(_core PRIVATE ${CLANG_LIBS})
    endif()
endif()

# Link zlib
target_link_libraries(_core PRIVATE ZLIB::ZLIB)

# Platform-specific system libraries
if(WIN32)
    target_link_libraries(_core PRIVATE
        version
        psapi
        shell32
        ole32
        uuid
        advapi32
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(_core PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
        m
    )
elseif(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    target_link_libraries(_core PRIVATE ${COREFOUNDATION_LIBRARY})
endif()

# Install the Python extension
install(TARGETS _core LIBRARY DESTINATION justjit)

# The LLVM shared library will be bundled by wheel repair tools:
# - auditwheel (Linux) - bundles .so files
# - delocate (macOS) - bundles .dylib files  
# - delvewheel (Windows) - bundles .dll files