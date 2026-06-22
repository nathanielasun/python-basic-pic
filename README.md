# python-basic-pic

Basic Python 3D particle-in-cell (PIC) implementation — a scaffold for a future C++ transition.

## Requirements

- **Python 3.11+** (tested with Python 3.14)
- A C/C++ toolchain and build tools (installed automatically by the setup scripts below)
- **OpenBLAS** for multithreaded linear algebra

This project builds **NumPy from source linked against OpenBLAS** instead of using a prebuilt wheel. On macOS, the default PyPI wheel uses Apple Accelerate, which does not give the same OpenBLAS threading control needed for PIC workloads.

## Setup

Run **one** platform setup script from the repository root. Each script creates `.venv`, builds OpenBLAS-linked NumPy into it, then installs PIC dependencies from `requirements.txt`.

| Platform | Command |
|----------|---------|
| macOS (Homebrew + Xcode CLI) | `./MacOS-setup.sh` then `source .venv/bin/activate` |
| Linux | `./Unix-setup.sh` then `source .venv/bin/activate` |
| Windows (MSYS2) | `.\Windows-setup.ps1` then `.\.venv\Scripts\Activate.ps1` |

Optional: `NUMPY_VERSION=2.4.6` and `INSTALL_ROOT=/path/to/repo` override defaults.

**Verify** (BLAS name should be `openblas`):

```bash
python -c "import numpy as np; print(np.__version__, np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])"
```

**Runtime threading:** examples default to Numba for particle work; keep `OPENBLAS_NUM_THREADS=1` when using Numba (the driver sets this if unset). See [`tests/test_numpy_openblas.py`](tests/test_numpy_openblas.py).

**Requirements files:** `requirements-build.txt` (meson, ninja, …) then `requirements.txt` (SciPy, h5py, matplotlib). NumPy is built by the setup scripts and is intentionally omitted from both files.

**Troubleshooting** (Accelerate instead of OpenBLAS, missing ninja, import errors, pkg-config, manual rebuild): see **[docs/numpy-openblas-troubleshooting.html](docs/numpy-openblas-troubleshooting.html)**.

## Running tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

If NumPy is already built and you only need PIC dependencies: `pip install -r requirements.txt`.

## Running simulations

# Note (Nate - 06/18/2026) These example files are for quick current testing and are NOT optimized
# These are likely to be changed (or removed) in the future as common simulation handles are written

Electrostatic PIC examples live under [`examples/`](examples/). Each script shares CLI flags via [`examples/common.py`](examples/common.py). Output animations go to [`animations/`](animations/) (gitignored).

| Script | Physics scenario |
|--------|------------------|
| [`01_kr_rf_plasma.py`](examples/01_kr_rf_plasma.py) | Uniform 10 GHz RF on Kr⁺ / e⁻ plasma |
| [`02_hydrogen_thermal.py`](examples/02_hydrogen_thermal.py) | Thermal H⁺ / e⁻, self-fields only |
| [`03_propagating_plane_wave.py`](examples/03_propagating_plane_wave.py) | Propagating 800 nm plane wave |
| [`04_gaussian_laser_pulse.py`](examples/04_gaussian_laser_pulse.py) | Gaussian laser pulse along +z |
| [`05_high_density_dc_bias.py`](examples/05_high_density_dc_bias.py) | High-density Kr⁺ / e⁻ with DC bias |
| [`06_elliptical_rf.py`](examples/06_elliptical_rf.py) | Elliptically polarized RF drive |

```bash
python examples/01_kr_rf_plasma.py --no-animate
python examples/02_hydrogen_thermal.py --steps 100 --no-animate
```

Flags: `--steps N`, `--seed N`, `--backend {numba,numpy}`, `--threads N`, `--no-animate`, `--frame-subsample N`, `--output-dir PATH`.

### Electromagnetic (Yee grid) examples

Full Maxwell PIC on [`YeeGrid`](src/grids/YeeGrid.py) with high-density (10²⁶–10²⁷ m⁻³) relativistic electrons and **Higuera–Cary** pushes (no Boris). Shared driver: [`examples/em_common.py`](examples/em_common.py).

| Script | Physics scenario |
|--------|------------------|
| [`em_01_kr_rf_relativistic.py`](examples/em_01_kr_rf_relativistic.py) | 10 GHz RF on relativistic Kr⁺ / e⁻ |
| [`em_02_hydrogen_relativistic_thermal.py`](examples/em_02_hydrogen_relativistic_thermal.py) | Hot H⁺ / e⁻, self-fields only |
| [`em_03_propagating_plane_wave.py`](examples/em_03_propagating_plane_wave.py) | 800 nm E+B plane wave on plasma |
| [`em_04_gaussian_laser_pulse.py`](examples/em_04_gaussian_laser_pulse.py) | Gaussian laser pulse + matched B |
| [`em_05_high_density_dc_bias.py`](examples/em_05_high_density_dc_bias.py) | 10²⁷ m⁻³ plasma, DC E + guide B |
| [`em_06_elliptical_rf.py`](examples/em_06_elliptical_rf.py) | Elliptical RF with axial guide B |

```bash
python examples/em_01_kr_rf_relativistic.py --no-animate --steps 100 --backend numba
```

EM examples use the same CLI flags as electrostatic runs (`--backend`, `--threads`, etc.). With `--backend numba`, particle ρ deposit, staggered E/B gather, Esirkepov J deposit, and Higuera–Cary push use parallel Numba kernels; FDTD field updates remain NumPy. Set `OPENBLAS_NUM_THREADS=1` when using Numba (handled automatically in the drivers).

## Documentation

- **[API reference](docs/index.html)** — module docs; serve with `cd docs && python3 -m http.server 8000`
- **[OpenBLAS NumPy troubleshooting](docs/numpy-openblas-troubleshooting.html)** — setup errors and manual rebuild

## Project layout

```
src/                  Top-level PIC modules (Pushers, Particle, pic_animation)
  fields/             Prescribed E/B sources, wave frames, field I/O
  grids/              Electrostatic & Yee grids, CIC kernels, grid helpers
examples/             Electrostatic (01–06) and EM Yee (em_01–em_06) drivers
tests/                Unit tests (kernels, fields, integrator, grids)
docs/                 API reference + OpenBLAS setup troubleshooting
animations/           Exported MP4 output (gitignored)
data/                 CSV field fixtures used by tests
.build/numpy/         NumPy source clone (gitignored)
.venv/                Project virtualenv
MacOS-setup.sh        macOS setup
Unix-setup.sh         Linux setup
Windows-setup.ps1     Windows setup
```
