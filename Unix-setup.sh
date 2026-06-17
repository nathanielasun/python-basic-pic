#!/usr/bin/env bash
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   Linux project setup: .venv, local OpenBLAS-linked NumPy build, PIC deps.
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

install_fedora_openblas_pc() {
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

setup_venv_and_build_toolchain() {
    echo "=== 2. Checking Python version ==="
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

    echo "=== 3. Creating Python virtual environment (.venv) ==="
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
    echo "=== 4. Installing NumPy build toolchain from requirements-build.txt ==="
    "$PYTHON" -m pip install -r "$BUILD_REQUIREMENTS"

    if ! command -v ninja >/dev/null 2>&1; then
        echo "ERROR: ninja is not on PATH. Expected it in .venv/bin." >&2
        exit 1
    fi
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

setup_venv_and_build_toolchain

echo "=== 5. Verifying OpenBLAS pkg-config ==="
if ! pkg-config --exists openblas; then
    echo "OpenBLAS was not found via pkg-config." >&2
    echo "Set PKG_CONFIG_PATH to the directory containing openblas.pc and re-run." >&2
    exit 1
fi

if [[ -d "$INSTALL_ROOT/numpy" ]]; then
    echo "Removing stale ${INSTALL_ROOT}/numpy source tree."
    rm -rf "$INSTALL_ROOT/numpy"
fi
if [[ -d "$INSTALL_ROOT/builddir" ]]; then
    echo "Removing stale ${INSTALL_ROOT}/builddir."
    rm -rf "$INSTALL_ROOT/builddir"
fi

echo "=== 6. Preparing NumPy ${NUMPY_VERSION} source under .build/ ==="
clone_numpy_source

echo "=== 7. Building and installing NumPy with OpenBLAS ==="
cd "$SRC_DIR"

"$PYTHON" -m pip install . \
    --verbose \
    --no-build-isolation \
    -Csetup-args=-Dblas=openblas \
    -Csetup-args=-Dlapack=openblas \
    -Csetup-args=-Dallow-noblas=false

cd "$INSTALL_ROOT"

echo "=== 8. Verifying NumPy / OpenBLAS install ==="
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
    echo "=== 9. Installing PIC dependencies from requirements.txt ==="
    "$PYTHON" -m pip install -r "$PIC_REQUIREMENTS"
fi

echo "===================================================="
echo "Setup complete. Local OpenBLAS NumPy is in .venv."
echo "Activate with:"
printf '  source %q\n' "$VENV_DIR/bin/activate"
echo "Set thread count, e.g.: export OPENBLAS_NUM_THREADS=4"
echo "===================================================="
