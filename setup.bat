@echo off
setlocal
title Meme Classification System - Setup
cd /d "%~dp0"

echo ============================================================
echo  Meme Classification System - Setup
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo Make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    python --version
    pause
    exit /b 1
)

echo [1/6] Python found:
python --version
echo.

if not exist "venv\Scripts\python.exe" (
    echo [2/6] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 goto :error
) else (
    echo [2/6] Virtual environment already exists.
)
set "PY=venv\Scripts\python.exe"
echo.

echo [3/6] Upgrading pip...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :error
echo.

set "HAS_GPU=0"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "HAS_GPU=1"

set "INSTALL_GPU=N"
if "%HAS_GPU%"=="0" (
    echo [4/6] No NVIDIA GPU detected. Using CPU PyTorch.
    goto :install_requirements
)

"%PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 (
    echo [4/6] CUDA PyTorch is already installed. Skipping download.
    goto :install_requirements
)

echo [4/6] NVIDIA GPU detected, but PyTorch is CPU-only.
set /p "INSTALL_GPU=Install CUDA PyTorch? Downloads about 2.6 GB [y/N]: "
if /i not "%INSTALL_GPU:~0,1%"=="y" goto :install_requirements
echo.

echo [5/6] Installing CUDA PyTorch...
"%PY%" -m pip install --force-reinstall --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
if errorlevel 1 (
    echo [WARN] CUDA install failed. Falling back to the CPU build.
) else (
    echo CUDA PyTorch installed.
)
echo.

:install_requirements
echo [6/6] Installing project requirements...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.

echo Creating project folders...
if not exist "static\uploads" mkdir "static\uploads"
if not exist "models" mkdir "models"
if not exist "data\train" mkdir "data\train"
if not exist "data\test" mkdir "data\test"
echo.

echo Checking Tesseract OCR...
where tesseract >nul 2>nul
if errorlevel 1 (
    if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
        echo Tesseract found at %ProgramFiles%\Tesseract-OCR\tesseract.exe
        echo The app will use this path automatically.
    ) else (
        echo [WARN] Tesseract OCR was not found.
        echo Install it from https://github.com/UB-Mannheim/tesseract/wiki
        echo Then set TESSERACT_CMD to the tesseract.exe path.
    )
) else (
    echo Tesseract OCR found.
)
echo.

echo Verifying PyTorch...
"%PY%" -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto :error
echo.

echo ============================================================
echo  Setup complete.
echo.
echo  Start the app:
echo    venv\Scripts\python.exe app.py
echo.
echo  Then open http://127.0.0.1:5000/
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed. Check the messages above.
pause
exit /b 1
