cmake_minimum_required(VERSION 3.24)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
project(SemanticSQLite VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# SQLite3
# ---------------------------------------------------------------------------
find_package(SQLite3 REQUIRED)

# ---------------------------------------------------------------------------
# Eigen3 (for cosine similarity on embedding blobs)
# ---------------------------------------------------------------------------
if(APPLE)
    find_package(Eigen3 3.4 REQUIRED NO_MODULE
        HINTS /opt/local/share/eigen3/cmake
    )
else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

# ---------------------------------------------------------------------------
# sqlite-ext executable
# ---------------------------------------------------------------------------
add_executable(sqlite-ext
    src/main.cpp
    src/double_metaphone.cpp
    src/ext_functions.cpp
)
target_include_directories(sqlite-ext PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(sqlite-ext PRIVATE
    SQLite::SQLite3
    Eigen3::Eigen
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix" FORCE)
install(TARGETS sqlite-ext RUNTIME DESTINATION bin)
