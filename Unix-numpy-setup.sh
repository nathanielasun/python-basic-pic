#!/usr/bin/env bash
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   Setup script for NumPy with OpenBLAS on Linux for multithreading.
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

install_fedora_openblas_pc() {
    # Fedora/RHEL ship OpenBLAS without an openblas.pc file.
    local pc_dir="${HOME}/.local/lib/pkgconfig"
    mkdir -p "$pc_dir"
    cat > "${pc_dir}/openblas.pc" <<'EOF'
prefix=/usr
includedir=${prefix}/include
libdir=${prefix}/lib64

Name: openblas
Description: OpenBLAS is an optimized BLAS library
Version: 0.3.0
Cflags: -I${includedir}/openblas
Libs: -L${libdir} -lopenblas
EOF
    export PKG_CONFIG_PATH="${pc_dir}:${PKG_CONFIG_PATH:-}"
}

echo "=== 1. Detecting OS and installing system dependencies ==="

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
else
    echo "Cannot identify Linux distribution." >&2
    exit 1
fi

case "${ID:-}" in
    debian|ubuntu)
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            gfortran \
            libopenblas-dev \
            liblapack-dev \
            pkg-config \
            git \
            python3-dev \
            python3-venv
        ;;

    fedora)
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y \
            gcc-gfortran \
            openblas-devel \
            lapack-devel \
            pkgconfig \
            git \
            python3-devel
        install_fedora_openblas_pc
        ;;

    rhel|centos|rocky|almalinux)
        if command -v dnf >/dev/null 2>&1; then
            PKG_MANAGER=dnf
        elif command -v yum >/dev/null 2>&1; then
            PKG_MANAGER=yum
        else
            echo "Neither dnf nor yum was found." >&2
            exit 1
        fi

        sudo "$PKG_MANAGER" groupinstall -y "Development Tools"
        sudo "$PKG_MANAGER" install -y \
            gcc-gfortran \
            openblas-devel \
            lapack-devel \
            pkgconfig \
            git \
            python3-devel
        install_fedora_openblas_pc
        ;;

    arch)
        sudo pacman -Syu --noconfirm --needed \
            base-devel gcc-fortran openblas pkgconf git python
        ;;

    *)
        echo "Unsupported distribution: ${ID:-unknown}" >&2
        exit 1
        ;;
esac

echo "=== 2. Verifying OpenBLAS pkg-config ==="
if ! pkg-config --exists openblas; then
    echo "OpenBLAS was not found via pkg-config." >&2
    echo "Set PKG_CONFIG_PATH to the directory containing openblas.pc and re-run." >&2
    exit 1
fi

echo "=== 3. Checking Python version ==="
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

echo "=== 4. Creating Python virtual environment ==="
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"

"$PYTHON" -m pip install --upgrade pip

if [[ -d "$INSTALL_ROOT/numpy" ]]; then
    echo "Removing stale ${INSTALL_ROOT}/numpy source tree."
    rm -rf "$INSTALL_ROOT/numpy"
fi
if [[ -d "$INSTALL_ROOT/builddir" ]]; then
    echo "Removing stale ${INSTALL_ROOT}/builddir."
    rm -rf "$INSTALL_ROOT/builddir"
fi

echo "=== 5. Cloning NumPy ${NUMPY_VERSION} ==="
git clone \
    --branch "v${NUMPY_VERSION}" \
    --depth 1 \
    --recurse-submodules \
    https://github.com/numpy/numpy.git \
    "$SRC_DIR"

echo "=== 6. Installing NumPy build requirements ==="
"$PYTHON" -m pip install -r "$SRC_DIR/requirements/build_requirements.txt"

if ! command -v ninja >/dev/null 2>&1; then
    echo "ERROR: ninja is not on PATH. Ensure .venv/bin is on PATH." >&2
    exit 1
fi

echo "=== 7. Building and installing NumPy with OpenBLAS ==="
cd "$SRC_DIR"

"$PYTHON" -m pip install . \
    --verbose \
    --no-build-isolation \
    -Csetup-args=-Dblas=openblas \
    -Csetup-args=-Dlapack=openblas \
    -Csetup-args=-Dallow-noblas=false

cd "$INSTALL_ROOT"

echo "=== 8. Verifying installation ==="
# Run outside the NumPy source tree so Python loads site-packages, not the clone.
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
