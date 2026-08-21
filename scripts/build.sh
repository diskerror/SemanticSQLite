#!/bin/bash
# build.sh — Build SemanticSQLite (semqlite) from source
#
# Usage: ./scripts/build.sh [clean] [--no-onnx]
#   clean     — remove build directory first
#   --no-onnx — disable ONNX Runtime embedding backend

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m'

# rustup installs cargo under ~/.cargo/bin and writes an env file to source.
[ -r "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

missing=()
check() { command -v "$1" &>/dev/null || missing+=("$1"); }

check cmake
check make
check c++
check pkg-config
# Rust toolchain — required by vendor/tokenizers-cpp (ONNX backend).
check cargo
check rustc

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required tools:${NC} ${missing[*]}"
    echo ""
    case " ${missing[*]} " in
        *" cargo "*|*" rustc "*)
            echo "    Install Rust with rustup (recommended):"
            echo "      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
            echo "    Then open a new shell, or run:  . \$HOME/.cargo/env"
            echo "" ;;
    esac
    echo "    System packages on macOS: sudo port install cmake pkg-config rust"
    echo "    System packages on Debian/Ubuntu: sudo apt install build-essential cmake pkg-config"
    exit 1
fi

cd "$(dirname "$0")/.."

ONNX_FLAG="ON"
DO_CLEAN=0

for arg in "$@"; do
    case "$arg" in
        clean)      DO_CLEAN=1 ;;
        --no-onnx)  ONNX_FLAG="OFF" ;;
    esac
done

BUILD_DIR="build"

if [ "$DO_CLEAN" = "1" ]; then
    echo "[+] Clean build"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect platform-specific cmake flags
CMAKE_FLAGS=""
OS="$(uname -s)"
case "$OS" in
    Darwin)
        # MacPorts Boost — find latest installed version
        BOOST_BASE="/opt/local/libexec/boost"
        if [ -d "$BOOST_BASE" ]; then
            BOOST_DIR=$(ls -d "$BOOST_BASE"/[0-9]* 2>/dev/null | sort -V | tail -1)
            if [ -n "$BOOST_DIR" ]; then
                CMAKE_FLAGS="-DBOOST_ROOT=$BOOST_DIR"
            fi
        fi
        JOBS=$(/usr/sbin/sysctl -n hw.ncpu)
        ;;
    Linux)
        JOBS=$(nproc)
        ;;
    *)
        JOBS=4
        ;;
esac

# Use local c_lib if CMakeUserPresets.json exists,
# otherwise FetchContent pulls it from GitHub.
LOCAL_C_LIB=""
if [ -f ../CMakeUserPresets.json ]; then
    C_LIB_PATH=$(python3 -c "import json; d=json.load(open('../CMakeUserPresets.json')); print(d['configurePresets'][0].get('cacheVariables',{}).get('FETCHCONTENT_SOURCE_DIR_C_LIB',''))" 2>/dev/null || true)
    if [ -n "$C_LIB_PATH" ] && [ -d "$C_LIB_PATH" ]; then
        LOCAL_C_LIB="-DFETCHCONTENT_SOURCE_DIR_C_LIB=$C_LIB_PATH"
    fi
fi

if [ ! -f Makefile ] || [ ../CMakeLists.txt -nt Makefile ]; then
    echo "[+] Configuring (SEMEXT_ONNX=$ONNX_FLAG)..."
    cmake .. $CMAKE_FLAGS $LOCAL_C_LIB -DSEMEXT_ONNX="$ONNX_FLAG"
fi

echo "[+] Building with $JOBS threads..."
make -j"$JOBS"

echo ""
echo "✓ Built: $(pwd)/semqlite"
echo "  Install with: sudo cmake --install build"
