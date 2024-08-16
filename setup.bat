@echo off
setlocal enabledelayedexpansion

REM Check if Python 3.10+ is installed
python --version 2>nul | findstr /r "^Python 3\.1[0-9]" >nul
if errorlevel 1 (
    echo Python 3.10 or higher is required. Please install it and try again.
    pause
    exit /b 1
)

REM Clone the torchchat repository if not already cloned
if not exist "torchchat" (
    git clone https://github.com/pytorch/torchchat.git
    if errorlevel 1 (
        echo Failed to clone the torchchat repository. Please check your internet connection.
        pause
        exit /b 1
    )
)

cd torchchat

REM Set up the virtual environment
python -m venv .venv
if errorlevel 1 (
    echo Failed to create a virtual environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip and install dependencies
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

REM Run the installation script
.\install_requirements.sh
if errorlevel 1 (
    echo Installation of requirements failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo All setup steps have been completed successfully.
echo You can now use torchchat by running the appropriate commands.
pause
