# python-basic-pic

Basic Python 3D particle-in-cell (PIC) implementation — a scaffold for a future C++ transition.

## Requirements

- **Python 3.11+** (tested with Python 3.14)
- A C/C++ toolchain and build tools (installed automatically by the setup scripts below)
- **OpenBLAS** for multithreaded linear algebra

This project builds **NumPy from source linked against OpenBLAS** instead of using a prebuilt wheel. On macOS, the default PyPI wheel uses Apple Accelerate, which does not give the same OpenBLAS threading control needed for PIC workloads.

## NumPy setup

Run the setup script for your platform from the repository root. Each script:

1. Installs system dependencies (OpenBLAS, compilers, `pkg-config`, etc.)
2. Creates or reuses a `.venv` virtual environment
3. Clones NumPy **v2.4.6** into a temporary directory
4. Builds and installs NumPy with `-Dblas=openblas -Dlapack=openblas -Dallow-noblas=false`
5. Verifies that the installed BLAS backend is `openblas` (not Apple Accelerate)

### macOS

Requires [Homebrew](https://brew.sh) and Xcode Command Line Tools.

```bash
./MacOS-numpy-setup.sh
source .venv/bin/activate
```

### Linux (Debian/Ubuntu, Fedora, RHEL/CentOS, Arch)

```bash
./Unix-numpy-setup.sh
source .venv/bin/activate
```

Fedora and RHEL-family distros ship OpenBLAS without a `openblas.pc` file; the Linux script creates a local pkg-config stub when needed.

### Windows

Requires Python, Git, and [MSYS2](https://www.msys2.org/) (installed via `winget` if missing). Run PowerShell as Administrator if package installs fail.

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   # if needed
.\Windows-numpy-setup.ps1
.\.venv\Scripts\Activate.ps1
```

### Optional environment variables

All setup scripts accept:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUMPY_VERSION` | `2.4.6` | NumPy release tag to build |
| `INSTALL_ROOT` | repository root | Directory containing `.venv` |

Example:

```bash
NUMPY_VERSION=2.4.6 ./Unix-numpy-setup.sh
```

### Verify the install

After activating `.venv`:

```bash
python -c "import numpy as np; c=np.__config__.show(mode='dicts'); print(np.__version__, c['Build Dependencies']['blas']['name'])"
```

Expected output includes `openblas` as the BLAS backend (not `accelerate` or `mkl`).

### OpenBLAS threading

Control BLAS thread count at runtime with `OPENBLAS_NUM_THREADS`:

```bash
export OPENBLAS_NUM_THREADS=4
```

## Running tests

OpenBLAS build and multithreading tests live in `tests/`:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

These tests check that NumPy is linked against OpenBLAS, that basic linear algebra works, and that OpenBLAS threading behaves as expected.

## Known issues and troubleshooting (macOS)

These problems were encountered while setting up NumPy on Apple Silicon macOS during development of this project. The macOS setup script (`MacOS-numpy-setup.sh`) includes guards for each; this section documents symptoms and fixes if you hit them manually.

### Apple Accelerate linked instead of OpenBLAS

**Symptom:** `np.show_config()` or `np.__config__.show(mode='dicts')` reports `blas: accelerate`, even though Homebrew OpenBLAS is installed. Grepping config output for the string `openblas` can still match (Accelerate entries include `"openblas configuration": "unknown"`), so that check is misleading.

**Cause:** On macOS, NumPy’s build system prefers Apple Accelerate unless BLAS is explicitly set to OpenBLAS at compile time.

**Fix:** Build with explicit meson options and disallow the no-BLAS fallback:

```bash
pip install . --no-build-isolation \
  -Csetup-args=-Dblas=openblas \
  -Csetup-args=-Dlapack=openblas \
  -Csetup-args=-Dallow-noblas=false
```

Verify with the BLAS **name**, not a text grep:

```bash
python -c "import numpy as np; print(np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])"
```

Expected: `openblas`.

### `meson-python: error: Could not find ninja`

**Symptom:** Metadata generation fails during `pip install` with no ninja on PATH.

**Cause:** `ninja` is installed into `.venv/bin` by pip, but meson-python looks for the `ninja` **executable** on `PATH`. If `.venv/bin` is not on `PATH`, the build fails even though `pip show ninja` succeeds.

**Fix:** Activate the venv or prepend `.venv/bin` to `PATH` before building:

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
```

The setup script does this automatically after creating the venv.

### `ImportError: cannot import name 'version' from 'numpy'`

**Symptom:** Immediately after a successful `pip install`, `import numpy` fails with an error pointing at a path inside a `numpy/` source directory.

**Cause:** Verification (or any `import numpy`) was run with the shell’s current directory inside a NumPy **source clone**. Python prepends the cwd to `sys.path`, so it loads the incomplete tree instead of the wheel in `site-packages`.

**Fix:** `cd` to the project root (or any directory without a `numpy/` package folder) before importing. The setup script changes back to `INSTALL_ROOT` before verification.

### Stale `numpy/` directory in the project root

**Symptom:** Intermittent import errors, wrong NumPy version, or wrong BLAS backend depending on which directory you run Python from.

**Cause:** A manual `git clone` of NumPy into the repo root (e.g. `./numpy`) was left behind from an earlier build attempt.

**Fix:** Remove the source tree; only the install in `.venv/lib/.../site-packages` should remain:

```bash
rm -rf numpy builddir
```

The setup script removes these directories if present before building.

### OpenBLAS not found via pkg-config

**Symptom:** Meson configures BLAS as missing or falls back to Accelerate / internal routines.

**Cause:** Homebrew installs OpenBLAS, but `pkg-config` does not find it unless `PKG_CONFIG_PATH` includes Homebrew’s OpenBLAS pkgconfig directory.

**Fix:**

```bash
export PKG_CONFIG_PATH="$(brew --prefix)/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"
pkg-config --exists openblas && echo OK
```

The setup script sets this and aborts early if `openblas` is not visible to pkg-config.

### Xcode Command Line Tools not installed

**Symptom:** `clang` or `git` not found; compiler sanity checks fail.

**Fix:**

```bash
xcode-select --install
```

Re-run the setup script after the installer finishes.

### Building from `main` without a release tag

**Symptom:** Installed version looks like `2.6.0.dev0+git...`; behavior differs from the pinned release; BLAS defaults may differ.

**Cause:** Cloning the default branch instead of a stable tag (this project pins **v2.4.6**).

**Fix:** Use the setup script, or clone explicitly:

```bash
git clone --branch v2.4.6 --depth 1 --recurse-submodules https://github.com/numpy/numpy.git
```

Override with `NUMPY_VERSION=2.4.6 ./MacOS-numpy-setup.sh` if needed.

## Project layout

```
src/                  Simulation code (grids, particles)
tests/                NumPy OpenBLAS build and threading tests
MacOS-numpy-setup.sh  macOS NumPy + OpenBLAS setup
Unix-numpy-setup.sh   Linux NumPy + OpenBLAS setup
Windows-numpy-setup.ps1  Windows NumPy + OpenBLAS setup
```
