#!/usr/bin/env pwsh
# Author: Nathaniel Sun
# Date: 2026-06-17
# Description:
#   Setup script for NumPy with OpenBLAS on Windows for multithreading.
#   Installs MSYS2 (OpenBLAS + pkg-config), sets up a Python virtual environment,
#   and compiles and installs NumPy with OpenBLAS.
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
$BuildRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("numpy-build-" + [guid]::NewGuid().ToString())
$SrcDir = Join-Path $BuildRoot "numpy"

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

    Write-Host "=== 2. Installing MSYS2 via Winget (if needed) ==="
    if (-not (Test-Path $Bash)) {
        if (-not (Test-Command winget)) {
            throw "MSYS2 is not installed and winget is unavailable. Install MSYS2 from https://www.msys2.org/ manually."
        }
        winget install --id MSYS2.MSYS2 -e --accept-package-agreements --accept-source-agreements
    }
    if (-not (Test-Path $Bash)) {
        throw "MSYS2 bash was not found at $Bash. Complete the MSYS2 install and re-run."
    }

    Write-Host "=== 3. Installing Toolchain and OpenBLAS inside MSYS2 ==="
    $packages = @(
        "mingw-w64-x86_64-gcc",
        "mingw-w64-x86_64-gfortran",
        "mingw-w64-x86_64-openblas",
        "mingw-w64-x86_64-pkg-config"
    )
    $packageList = ($packages -join " ")
    & $Bash -lc "pacman -Syu --noconfirm"
    & $Bash -lc "pacman -S --noconfirm --needed $packageList"

    Write-Host "=== 4. Updating Windows Path Environment ==="
    $env:Path = "$Msys64\mingw64\bin;$Msys64\usr\bin;" + $env:Path
    $env:PKG_CONFIG_PATH = "$Msys64\mingw64\lib\pkgconfig"

    Write-Host "=== 5. Verifying OpenBLAS pkg-config ==="
    & $Bash -lc "pkg-config --exists openblas"
    if ($LASTEXITCODE -ne 0) {
        throw "OpenBLAS was not found via pkg-config under $env:PKG_CONFIG_PATH"
    }

    Write-Host "=== 6. Creating Python virtual environment ==="
    if (-not (Test-Path $VenvDir)) {
        python -m venv $VenvDir
    }
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    $env:Path = (Join-Path $VenvDir "Scripts") + ";" + $env:Path
    & $Python -m pip install --upgrade pip

    Write-Host "=== 7. Cloning NumPy $NumpyVersion ==="
    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
    git clone `
        --branch "v$NumpyVersion" `
        --depth 1 `
        --recurse-submodules `
        https://github.com/numpy/numpy.git `
        $SrcDir

    Write-Host "=== 8. Installing NumPy build requirements ==="
    & $Python -m pip install -r (Join-Path $SrcDir "requirements\build_requirements.txt")

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

    Write-Host "=== 10. Verifying installation ==="
    $blasName = & $Python -c @"
import numpy as np
print(np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])
"@
    $numpyVersion = & $Python -c "import numpy as np; print(np.__version__)"
    Write-Host "NumPy $numpyVersion installed with BLAS backend: $blasName"

    if ($blasName.ToLower() -ne "openblas") {
        throw "Expected OpenBLAS, but NumPy reports BLAS backend '$blasName'."
    }

    Write-Host "===================================================="
    Write-Host "NumPy $NumpyVersion was installed with OpenBLAS."
    Write-Host "Activate it with: .\$VenvDir\Scripts\Activate.ps1"
    Write-Host "===================================================="
}
finally {
    if (Test-Path $BuildRoot) {
        Remove-Item -Recurse -Force $BuildRoot
    }
}
