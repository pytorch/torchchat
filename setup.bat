@echo off
setlocal

REM === Basic Setup ===

echo Setting up the environment...
REM Update the package manager (Assuming Chocolatey for Windows)
choco upgrade chocolatey -y

REM === Install Dependencies ===

echo Installing dependencies...

REM Install CMake
choco install cmake -y

REM Install Boost Libraries
choco install boost-msvc-14.1 -y

REM Install other required packages (example: Git, Python)
choco install git -y
choco install python -y

REM === Install Project-Specific Tools ===

echo Installing project-specific tools...

REM Install specific Python packages using pip
python -m pip install --upgrade pip
pip install -r requirements.txt

REM === Project Configuration ===

echo Configuring the project...

REM Create build directory
if not exist "build" (
    mkdir build
)

REM Run CMake to configure the project
cd build
cmake ..

REM === Build the Project ===

echo Building the project...
cmake --build .

REM === Post-Build Steps ===

echo Running post-build steps...

REM Example: Running tests, setting environment variables, etc.
REM Assuming you have a test script or command
ctest

REM === Final Steps ===

cd ..
echo Setup completed successfully!

REM === Keep the Command Prompt Open After Execution ===
echo The script has completed. Press any key to exit...
pause >nul
