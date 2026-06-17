#!/usr/bin/env bash
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   macOS project setup: .venv, local OpenBLAS-linked NumPy build, PIC deps.
#   NumPy is compiled for multithreaded BLAS (OPENBLAS_NUM_THREADS) control.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUMPY_VERSION="${NUMPY_VERSION:-2.4.6}"
INSTALL_ROOT="${INSTALL_ROOT:-$SCRIPT_DIR}"
VENV_DIR="$INSTALL_ROOT/.venv"
BUILD_ROOT="$INSTALL_ROOT/.build"
SRC_DIR="$BUILD_ROOT/numpy"
BUILD_REQUIREMENTS="$INSTALL_ROOT/requirements-build.txt"
PIC_REQUIREMENTS="$INSTALL_ROOT/requirements.txt"

clone_numpy_source() {
    mkdir -p "$BUILD_ROOT"
    if [[ -d "$SRC_DIR/.git" ]]; then
        echo "Reusing existing NumPy source at ${SRC_DIR}"
        git -C "$SRC_DIR" fetch --depth 1 origin "refs/tags/v${NUMPY_VERSION}:refs/tags/v${NUMPY_VERSION}" 2>/dev/null || true
        git -C "$SRC_DIR" checkout -f "v${NUMPY_VERSION}"
        git -C "$SRC_DIR" submodule update --init --recursive
        return
    fi
    echo "Cloning NumPy v${NUMPY_VERSION} into ${SRC_DIR}"
    git clone \
        --branch "v${NUMPY_VERSION}" \
        --depth 1 \
        --recurse-submodules \
        https://github.com/numpy/numpy.git \
        "$SRC_DIR"
}

setup_venv_and_build_toolchain() {
    echo "=== 1. Checking Python version ==="
    python3 - <<'PY'
import sys

minimum = (3, 11)
if sys.version_info < minimum:
    detected = ".".join(map(str, sys.version_info[:3]))
    required = ".".join(map(str, minimum))
    raise SystemExit(
        f"Python {required}+ is required for NumPy 2.4.x; found {detected}."
    )
PY

    echo "=== 2. Creating Python virtual environment (.venv) ==="
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    PYTHON="$VENV_DIR/bin/python"
    export PATH="$VENV_DIR/bin:$PATH"

    "$PYTHON" -m pip install --upgrade pip

    if [[ ! -f "$BUILD_REQUIREMENTS" ]]; then
        echo "ERROR: ${BUILD_REQUIREMENTS} not found." >&2
        exit 1
    fi
    echo "=== 3. Installing NumPy build toolchain from requirements-build.txt ==="
    "$PYTHON" -m pip install -r "$BUILD_REQUIREMENTS"

    if ! command -v ninja >/dev/null 2>&1; then
        echo "ERROR: ninja is not on PATH. Expected it in .venv/bin." >&2
        exit 1
    fi
}

echo "=== 0. Checking Xcode Command Line Tools ==="
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Xcode Command Line Tools are required (clang, git)."
    echo "A dialog should open; complete the install, then re-run this script."
    xcode-select --install
    exit 1
fi

setup_venv_and_build_toolchain

echo "=== 4. Installing system dependencies via Homebrew ==="
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required. Install it from https://brew.sh and re-run this script." >&2
    exit 1
fi

brew install openblas pkg-config gfortran git

BREW_PREFIX="$(brew --prefix)"
export PKG_CONFIG_PATH="${BREW_PREFIX}/opt/openblas/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "=== 5. Verifying OpenBLAS pkg-config ==="
if ! pkg-config --exists openblas; then
    echo "OpenBLAS was not found via pkg-config." >&2
    echo "Expected pkg-config file under: ${BREW_PREFIX}/opt/openblas/lib/pkgconfig" >&2
    exit 1
fi

echo "=== 6. Removing stale NumPy source trees in project root ==="
if [[ -d "$INSTALL_ROOT/numpy" ]]; then
    echo "Removing ${INSTALL_ROOT}/numpy (can cause ImportError or wrong BLAS backend)."
    rm -rf "$INSTALL_ROOT/numpy"
fi
if [[ -d "$INSTALL_ROOT/builddir" ]]; then
    echo "Removing ${INSTALL_ROOT}/builddir (stale meson build directory)."
    rm -rf "$INSTALL_ROOT/builddir"
fi

if "$PYTHON" -c "import numpy" 2>/dev/null; then
    EXISTING_BLAS="$("$PYTHON" -c '
import numpy as np
print(np.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"])
' 2>/dev/null || true)"
    if [[ -n "$EXISTING_BLAS" ]]; then
        EXISTING_LC="$(printf '%s' "$EXISTING_BLAS" | tr '[:upper:]' '[:lower:]')"
        if [[ "$EXISTING_LC" != "openblas" ]]; then
            echo "Note: Replacing existing NumPy (BLAS backend: ${EXISTING_BLAS})."
            echo "      macOS often defaults to Apple Accelerate unless OpenBLAS is forced at build time."
        fi
    fi
fi

echo "=== 7. Preparing NumPy ${NUMPY_VERSION} source under .build/ ==="
clone_numpy_source

echo "=== 8. Building and installing NumPy with OpenBLAS ==="
cd "$SRC_DIR"

"$PYTHON" -m pip install . \
    --verbose \
    --no-build-isolation \
    -Csetup-args=-Dblas=openblas \
    -Csetup-args=-Dlapack=openblas \
    -Csetup-args=-Dallow-noblas=false

echo "=== 9. Verifying NumPy / OpenBLAS install ==="
cd "$INSTALL_ROOT"

BLAS_NAME="$("$PYTHON" -c '
import numpy as np
config = np.__config__.show(mode="dicts")
print(config["Build Dependencies"]["blas"]["name"])
')"

printf 'NumPy %s installed with BLAS backend: %s\n' \
    "$("$PYTHON" -c 'import numpy as np; print(np.__version__)')" \
    "$BLAS_NAME"

BLAS_LC="$(printf '%s' "$BLAS_NAME" | tr '[:upper:]' '[:lower:]')"
if [[ "$BLAS_LC" != "openblas" ]]; then
    echo "ERROR: Expected OpenBLAS, but NumPy reports BLAS backend '${BLAS_NAME}'." >&2
    exit 1
fi

if [[ -f "$PIC_REQUIREMENTS" ]]; then
    echo "=== 10. Installing PIC dependencies from requirements.txt ==="
    "$PYTHON" -m pip install -r "$PIC_REQUIREMENTS"
fi

echo "===================================================="
echo "Setup complete. Local OpenBLAS NumPy is in .venv."
echo "Activate with:"
printf '  source %q\n' "$VENV_DIR/bin/activate"
echo "Set thread count, e.g.: export OPENBLAS_NUM_THREADS=4"
echo "===================================================="
