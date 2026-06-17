#!/usr/bin/env pwsh
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   Windows project setup: .venv, local OpenBLAS-linked NumPy build, PIC deps.
#   NumPy is compiled for multithreaded BLAS (OPENBLAS_NUM_THREADS) control.
# NOTE: Run as Administrator if package installs fail. You may also need:
#       Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$NumpyVersion = if ($env:NUMPY_VERSION) { $env:NUMPY_VERSION } else { "2.4.6" }
$InstallRoot = if ($env:INSTALL_ROOT) { $env:INSTALL_ROOT } else { $ScriptDir }
$VenvDir = Join-Path $InstallRoot ".venv"
$Msys64 = "C:\msys64"
$Bash = Join-Path $Msys64 "usr\bin\bash.exe"
$BuildRoot = Join-Path $InstallRoot ".build"
$SrcDir = Join-Path $BuildRoot "numpy"
$BuildRequirements = Join-Path $InstallRoot "requirements-build.txt"
$PicRequirements = Join-Path $InstallRoot "requirements.txt"

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

try {
    Write-Host "=== 1. Checking prerequisites ==="
    if (-not (Test-Command python)) {
        throw "Python was not found on PATH. Install Python from https://www.python.org/downloads/ and re-run."
    }
    if (-not (Test-Command git)) {
        throw "Git was not found on PATH. Install Git for Windows and re-run."
    }

    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $major, $minor = $pyVersion.Split(".")
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
        throw "Python 3.11+ is required for NumPy 2.4.x; found $pyVersion."
    }

    Write-Host "=== 2. Creating Python virtual environment (.venv) ==="
    if (-not (Test-Path $VenvDir)) {
        python -m venv $VenvDir
    }
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    $env:Path = (Join-Path $VenvDir "Scripts") + ";" + $env:Path
    & $Python -m pip install --upgrade pip

    if (-not (Test-Path $BuildRequirements)) {
        throw "requirements-build.txt not found at $BuildRequirements"
    }
    Write-Host "=== 3. Installing NumPy build toolchain from requirements-build.txt ==="
    & $Python -m pip install -r $BuildRequirements

    if (-not (Test-Command ninja)) {
        throw "ninja is not on PATH. Expected it in .venv\Scripts after installing requirements-build.txt."
    }

    Write-Host "=== 4. Installing MSYS2 via Winget (if needed) ==="
    if (-not (Test-Path $Bash)) {
        if (-not (Test-Command winget)) {
            throw "MSYS2 is not installed and winget is unavailable. Install MSYS2 from https://www.msys2.org/ manually."
        }
        winget install --id MSYS2.MSYS2 -e --accept-package-agreements --accept-source-agreements
    }
    if (-not (Test-Path $Bash)) {
        throw "MSYS2 bash was not found at $Bash. Complete the MSYS2 install and re-run."
    }

    Write-Host "=== 5. Installing toolchain and OpenBLAS inside MSYS2 ==="
    $packages = @(
        "mingw-w64-x86_64-gcc",
        "mingw-w64-x86_64-gfortran",
        "mingw-w64-x86_64-openblas",
        "mingw-w64-x86_64-pkg-config"
    )
    $packageList = ($packages -join " ")
    & $Bash -lc "pacman -Syu --noconfirm"
    & $Bash -lc "pacman -S --noconfirm --needed $packageList"

    Write-Host "=== 6. Updating Windows PATH for OpenBLAS pkg-config ==="
    $env:Path = "$Msys64\mingw64\bin;$Msys64\usr\bin;" + $env:Path
    $env:PKG_CONFIG_PATH = "$Msys64\mingw64\lib\pkgconfig"

    Write-Host "=== 7. Verifying OpenBLAS pkg-config ==="
    & $Bash -lc "pkg-config --exists openblas"
    if ($LASTEXITCODE -ne 0) {
        throw "OpenBLAS was not found via pkg-config under $env:PKG_CONFIG_PATH"
    }

    Write-Host "=== 8. Preparing NumPy $NumpyVersion source under .build/ ==="
    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
    if (Test-Path (Join-Path $SrcDir ".git")) {
        Write-Host "Reusing existing NumPy source at $SrcDir"
        git -C $SrcDir fetch --depth 1 origin "refs/tags/v$NumpyVersion:refs/tags/v$NumpyVersion" 2>$null
        git -C $SrcDir checkout -f "v$NumpyVersion"
        git -C $SrcDir submodule update --init --recursive
    }
    else {
        Write-Host "Cloning NumPy v$NumpyVersion into $SrcDir"
        git clone `
            --branch "v$NumpyVersion" `
            --depth 1 `
            --recurse-submodules `
            https://github.com/numpy/numpy.git `
            $SrcDir
    }

    Write-Host "=== 9. Building and installing NumPy with OpenBLAS ==="
    Push-Location $SrcDir
    try {
        & $Python -m pip install . `
            --verbose `
            --no-build-isolation `
            -Csetup-args=-Dblas=openblas `
            -Csetup-args=-Dlapack=openblas `
            -Csetup-args=-Dallow-noblas=false
    }
    finally {
        Pop-Location
    }

    Write-Host "=== 10. Verifying NumPy / OpenBLAS install ==="
    $blasName = & $Python -c @"
import numpy as np
print(np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])
"@
    $numpyVersion = & $Python -c "import numpy as np; print(np.__version__)"
    Write-Host "NumPy $numpyVersion installed with BLAS backend: $blasName"

    if ($blasName.ToLower() -ne "openblas") {
        throw "Expected OpenBLAS, but NumPy reports BLAS backend '$blasName'."
    }

    if (Test-Path $PicRequirements) {
        Write-Host "=== 11. Installing PIC dependencies from requirements.txt ==="
        & $Python -m pip install -r $PicRequirements
    }

    Write-Host "===================================================="
    Write-Host "Setup complete. Local OpenBLAS NumPy is in .venv."
    Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
    Write-Host 'Set thread count, e.g.: $env:OPENBLAS_NUM_THREADS = 4'
    Write-Host "===================================================="
}
