cmake_minimum_required(VERSION 3.24)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
project(SemanticSQLite VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Our added functions: DMPHON() and EMBEDDING_SIM()/EMBEDDING_DIST()
# ---------------------------------------------------------------------------
add_library(semext STATIC
    src/ext_functions.c
    src/double_metaphone.cpp
    src/double_metaphone_capi.cpp
)
target_include_directories(semext PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(semext PRIVATE "${SQLITE_VENDOR_DIR}")
target_link_libraries(semext PUBLIC sqlite3_amalgamation)

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

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix" FORCE)
install(TARGETS sqlite-ext RUNTIME DESTINATION bin)
