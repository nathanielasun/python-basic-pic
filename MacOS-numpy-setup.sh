#!/usr/bin/env bash
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   Setup script for NumPy with OpenBLAS on macOS for multithreading.
#   Installs system dependencies, sets up a Python virtual environment,
#   and compiles and installs NumPy with OpenBLAS.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUMPY_VERSION="${NUMPY_VERSION:-2.4.6}"
INSTALL_ROOT="${INSTALL_ROOT:-$SCRIPT_DIR}"
VENV_DIR="$INSTALL_ROOT/.venv"
BUILD_ROOT="$(mktemp -d)"
SRC_DIR="$BUILD_ROOT/numpy"

cleanup() {
    rm -rf "$BUILD_ROOT"
}
trap cleanup EXIT

echo "=== 1. Checking Xcode Command Line Tools ==="
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Xcode Command Line Tools are required (clang, git)."
    echo "A dialog should open; complete the install, then re-run this script."
    xcode-select --install
    exit 1
fi

echo "=== 2. Installing System Dependencies via Homebrew ==="
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required. Install it from https://brew.sh and re-run this script." >&2
    exit 1
fi

brew install openblas pkg-config gfortran git

BREW_PREFIX="$(brew --prefix)"
export PKG_CONFIG_PATH="${BREW_PREFIX}/opt/openblas/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "=== 3. Verifying OpenBLAS pkg-config ==="
if ! pkg-config --exists openblas; then
    echo "OpenBLAS was not found via pkg-config." >&2
    echo "Expected pkg-config file under: ${BREW_PREFIX}/opt/openblas/lib/pkgconfig" >&2
    exit 1
fi

echo "=== 4. Checking Python version ==="
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

echo "=== 5. Creating Python virtual environment ==="
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"

"$PYTHON" -m pip install --upgrade pip

echo "=== 5a. Removing stale NumPy source trees in project root ==="
# A leftover numpy/ clone (from manual builds) shadows the installed package
# when Python adds the current working directory to sys.path.
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

echo "=== 6. Cloning NumPy ${NUMPY_VERSION} ==="
git clone \
    --branch "v${NUMPY_VERSION}" \
    --depth 1 \
    --recurse-submodules \
    https://github.com/numpy/numpy.git \
    "$SRC_DIR"

echo "=== 7. Installing NumPy build requirements ==="
"$PYTHON" -m pip install -r "$SRC_DIR/requirements/build_requirements.txt"

if ! command -v ninja >/dev/null 2>&1; then
    echo "ERROR: ninja is not on PATH." >&2
    echo "meson-python requires ninja >= 1.8.2. pip installs it into .venv/bin;" >&2
    echo "ensure .venv/bin is on PATH before building (this script does that automatically)." >&2
    exit 1
fi

echo "=== 8. Building and installing NumPy with OpenBLAS ==="
cd "$SRC_DIR"

"$PYTHON" -m pip install . \
    --verbose \
    --no-build-isolation \
    -Csetup-args=-Dblas=openblas \
    -Csetup-args=-Dlapack=openblas \
    -Csetup-args=-Dallow-noblas=false

echo "=== 9. Verifying installation ==="
# Verification must run outside the NumPy source tree; otherwise Python may
# import the incomplete clone from sys.path instead of site-packages.
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

echo "===================================================="
echo "NumPy ${NUMPY_VERSION} was installed with OpenBLAS."
echo "Activate it with:"
printf '  source %q\n' "$VENV_DIR/bin/activate"
echo "===================================================="
