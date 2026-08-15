cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0175)
    cmake_policy(SET CMP0175 NEW)
endif()
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
project(SemanticSQLite VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(SEMEXT_ONNX "Build with ONNX Runtime embedding backend (default ON)" ON)

# ---------------------------------------------------------------------------
# Vendored SQLite: an UNMODIFIED upstream amalgamation (sqlite3.c/.h) and CLI
# shell (shell.c), generated from the official sqlite/sqlite mirror
# (github.com/diskerror/sqlite fork) via `make sqlite3.c shell.c sqlite3.h`.
# See vendor/sqlite/UPSTREAM_COMMIT.txt for the exact commit + version.
#
# shell.c's own header comment documents `#define main sqlite3_shell` as the
# supported way to use it "as a subroutine" from another project — that's
# exactly what we do in src/main.c, so upstream needs ZERO patches.
# ---------------------------------------------------------------------------
set(SQLITE_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor/sqlite")

# Shell build flags mirroring what SQLite's own Makefile uses for the CLI
# (readline history/editing, FTS5, JSON1, R-Tree, etc. are already baked
# into the amalgamation by configure/mkshellc; here we just add the
# essentials for a standalone build outside their Makefile).
set(SHELL_DEFS
    -DSQLITE_ENABLE_FTS5
    -DSQLITE_ENABLE_RTREE
    -DSQLITE_ENABLE_JSON1
    -DSQLITE_ENABLE_DBSTAT_VTAB
    -DSQLITE_ENABLE_EXPLAIN_COMMENTS
    -DHAVE_READLINE=0
)

add_library(sqlite3_amalgamation STATIC
    "${SQLITE_VENDOR_DIR}/sqlite3.c"
)
target_include_directories(sqlite3_amalgamation PUBLIC "${SQLITE_VENDOR_DIR}")
target_compile_options(sqlite3_amalgamation PRIVATE ${SHELL_DEFS})
if(APPLE)
    target_link_libraries(sqlite3_amalgamation PUBLIC "-framework CoreFoundation")
endif()
find_package(Threads REQUIRED)
target_link_libraries(sqlite3_amalgamation PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)

# shell.c compiled as its own object, main() renamed to sqlite3_shell() so
# it becomes a callable subroutine (upstream-documented pattern — see the
# comment above int main() in vendor/sqlite/shell.c). Our custom functions
# are wired in via shell.c's own SQLITE_SHELL_EXTFUNCS hook (see
# include/shell_extfuncs_hook.h) rather than sqlite3_auto_extension(),
# because auto_extension from a wrapper main() forces early library init
# and trips shell.c's own verify_uninitialized() self-check.
add_library(sqlite_shell_obj OBJECT
    "${SQLITE_VENDOR_DIR}/shell.c"
)
target_include_directories(sqlite_shell_obj PRIVATE
    "${SQLITE_VENDOR_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
)
target_compile_options(sqlite_shell_obj PRIVATE
    ${SHELL_DEFS}
    -Dmain=sqlite3_shell
    -DSQLITE_SHELL_EXTFUNCS=SEMEXT
    -include
    "${CMAKE_SOURCE_DIR}/include/shell_extfuncs_hook.h"
)

# ---------------------------------------------------------------------------
# libllama (llama.cpp) — used by EMBED() to run GGUF embedding models.
# macOS/MacPorts: `port install llama.cpp` puts it under /opt/local.
# Debian/Linux: build llama.cpp from source (`cmake --install` to /usr/local)
#   or install a distro package if available. Also searches
#   /opt/llama.cpp/build (a common in-tree build location that never got
#   'cmake --install'ed). Set -DLLAMA_ROOT=/path/to/llama.cpp/build to
#   point at a specific build tree if none of the above match.
# ---------------------------------------------------------------------------
set(LLAMA_ROOT "" CACHE PATH "Root directory of a llama.cpp build/install (searched for lib/ and include/)")
find_library(LLAMA_LIBRARY NAMES llama
    PATHS ${LLAMA_ROOT} /opt/local/lib /usr/local/lib /usr/lib
          /opt/llama.cpp/build/bin /opt/llama.cpp/build/lib
    PATH_SUFFIXES lib)
find_path(LLAMA_INCLUDE_DIR NAMES llama.h
    PATHS ${LLAMA_ROOT} /opt/local/include /usr/local/include /usr/include
          /opt/llama.cpp/include
    PATH_SUFFIXES include)
if(NOT LLAMA_LIBRARY OR NOT LLAMA_INCLUDE_DIR)
    message(FATAL_ERROR
        "libllama not found. On macOS: 'sudo port install llama.cpp'. "
        "On Debian/Linux: build llama.cpp from source and 'cmake --install'. "
        "If it's built but not installed to a standard path, pass "
        "-DLLAMA_ROOT=/path/to/llama.cpp/build. "
        "EMBED() requires libllama; DMPHON/EMBEDDING_SIM/EMBEDDING_DIST do not.")
endif()
message(STATUS "libllama: ${LLAMA_LIBRARY} (headers: ${LLAMA_INCLUDE_DIR})")
get_filename_component(LLAMA_LIBRARY_DIR "${LLAMA_LIBRARY}" DIRECTORY)

# llama.h #includes "ggml.h", which in an un-installed in-tree llama.cpp
# build lives in a SEPARATE directory (ggml/include, a sibling of the repo
# root, not under LLAMA_INCLUDE_DIR) — only a `cmake --install`ed tree
# flattens ggml's headers alongside llama.h. Search for it near
# LLAMA_INCLUDE_DIR so in-tree builds (e.g. /opt/llama.cpp/build without
# an install step) still compile.
find_path(GGML_INCLUDE_DIR NAMES ggml.h
    PATHS "${LLAMA_INCLUDE_DIR}" "${LLAMA_INCLUDE_DIR}/../ggml/include"
          "${LLAMA_ROOT}/ggml/include" /opt/llama.cpp/ggml/include
          /opt/local/include /usr/local/include /usr/include
    NO_DEFAULT_PATH)
if(NOT GGML_INCLUDE_DIR)
    # Fall back to a normal search (covers installed layouts where ggml.h
    # sits right next to llama.h, i.e. GGML_INCLUDE_DIR == LLAMA_INCLUDE_DIR).
    set(GGML_INCLUDE_DIR "${LLAMA_INCLUDE_DIR}")
endif()
message(STATUS "ggml headers: ${GGML_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# ONNX Runtime + tokenizers-cpp (optional, for the ONNX embedding backend)
# ---------------------------------------------------------------------------
set(SEMEXT_ONNX_SOURCES "")
set(SEMEXT_ONNX_LIBS "")
set(SEMEXT_ONNX_INCLUDES "")

if(SEMEXT_ONNX)
    # ---- ONNX Runtime (vendored or auto-fetched, same as Ragger) ----
    set(ORT_VERSION "1.24.4" CACHE STRING "ONNX Runtime version")
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(ORT_PKG "onnxruntime-osx-arm64-${ORT_VERSION}")
        else()
            set(ORT_PKG "onnxruntime-osx-x86_64-${ORT_VERSION}")
        endif()
        set(ORT_LIB_NAME "libonnxruntime.dylib")
    elseif(UNIX)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(ORT_PKG "onnxruntime-linux-aarch64-${ORT_VERSION}")
        else()
            set(ORT_PKG "onnxruntime-linux-x64-${ORT_VERSION}")
        endif()
        set(ORT_LIB_NAME "libonnxruntime.so")
    else()
        message(FATAL_ERROR "Unsupported platform for ONNX Runtime")
    endif()

    set(ORT_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime")
    if(EXISTS "${ORT_VENDOR_DIR}/lib/${ORT_LIB_NAME}"
       AND EXISTS "${ORT_VENDOR_DIR}/include/onnxruntime_cxx_api.h")
        set(ORT_DIR "${ORT_VENDOR_DIR}")
        message(STATUS "ONNX Runtime: using vendored ${ORT_DIR}")
    else()
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}.tgz")
        set(ORT_CACHE "${CMAKE_BINARY_DIR}/_deps/onnxruntime/${ORT_PKG}")
        if(NOT EXISTS "${ORT_CACHE}/lib/${ORT_LIB_NAME}")
            message(STATUS "ONNX Runtime: fetching ${ORT_URL}")
            file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/_deps/${ORT_PKG}.tgz" SHOW_PROGRESS STATUS _ort_dl)
            list(GET _ort_dl 0 _ort_rc)
            if(NOT _ort_rc EQUAL 0)
                message(FATAL_ERROR "ONNX Runtime download failed: ${_ort_dl}")
            endif()
            file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/_deps/${ORT_PKG}.tgz"
                 DESTINATION "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
        endif()
        set(ORT_DIR "${ORT_CACHE}")
        message(STATUS "ONNX Runtime: ${ORT_DIR}")
    endif()
    set(ORT_LIB "${ORT_DIR}/lib/${ORT_LIB_NAME}")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
    )
    if(APPLE)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_SONAME "@rpath/libonnxruntime.dylib")
    endif()

    # ---- tokenizers-cpp (vendored, same subtree as Ragger) ----
    set(TOKENIZERS_VENDOR "${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp")
    if(NOT EXISTS "${TOKENIZERS_VENDOR}/CMakeLists.txt")
        message(FATAL_ERROR "vendor/tokenizers-cpp not found. Copy or symlink from Ragger/vendor/tokenizers-cpp.")
    endif()
    add_subdirectory("${TOKENIZERS_VENDOR}")

    set(SEMEXT_ONNX_SOURCES src/embed_onnx.cpp)
    list(APPEND SEMEXT_ONNX_LIBS onnxruntime tokenizers_cpp)
    list(APPEND SEMEXT_ONNX_INCLUDES "${ORT_DIR}/include")
    message(STATUS "ONNX embedding backend: enabled")
else()
    message(STATUS "ONNX embedding backend: disabled (-DSEMEXT_ONNX=OFF)")
endif()

# ---------------------------------------------------------------------------
# Our added functions: DMPHON(), EMBEDDING_SIM()/EMBEDDING_DIST(), EMBED()
# ---------------------------------------------------------------------------
add_library(semext STATIC
    src/ext_functions.c
    src/double_metaphone.cpp
    src/double_metaphone_capi.cpp
    src/embed.c
    ${SEMEXT_ONNX_SOURCES}
)
target_include_directories(semext PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(semext PRIVATE "${SQLITE_VENDOR_DIR}" "${LLAMA_INCLUDE_DIR}" "${GGML_INCLUDE_DIR}" ${SEMEXT_ONNX_INCLUDES})
target_link_libraries(semext PUBLIC sqlite3_amalgamation "${LLAMA_LIBRARY}" ${SEMEXT_ONNX_LIBS})
if(SEMEXT_ONNX)
    target_compile_definitions(semext PUBLIC SEMEXT_HAVE_ONNX=1)
endif()

# ---------------------------------------------------------------------------
# sqlite-ext executable: our tiny main() pre-registers semext's functions
# via sqlite3_auto_extension(), then hands off to the real shell.
# ---------------------------------------------------------------------------
add_executable(sqlite-ext
    src/main.c
    $<TARGET_OBJECTS:sqlite_shell_obj>
)
target_include_directories(sqlite-ext PRIVATE "${SQLITE_VENDOR_DIR}")
target_link_libraries(sqlite-ext PRIVATE semext sqlite3_amalgamation)

# On Linux, shared libs found via full path (libllama.so, libonnxruntime.so)
# are NOT automatically embedded into the binary's runtime search path the
# way macOS embeds absolute dylib paths — the linker only records the
# soname, so the loader falls back to ldconfig's cache / LD_LIBRARY_PATH,
# which won't include non-standard install locations (e.g. a llama.cpp
# built from source into /usr/local, or a vendored onnxruntime under
# vendor/onnxruntime/lib). Set an explicit RPATH so `sqlite-ext` finds them
# without requiring `ldconfig` or LD_LIBRARY_PATH gymnastics.
if(UNIX AND NOT APPLE)
    set(SEMEXT_RPATH_DIRS "${LLAMA_LIBRARY_DIR}")
    if(SEMEXT_ONNX)
        list(APPEND SEMEXT_RPATH_DIRS "${ORT_DIR}/lib")
    endif()
    set_target_properties(sqlite-ext PROPERTIES
        BUILD_RPATH   "${SEMEXT_RPATH_DIRS}"
        INSTALL_RPATH "${SEMEXT_RPATH_DIRS}"
    )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix" FORCE)
install(TARGETS sqlite-ext RUNTIME DESTINATION bin)
