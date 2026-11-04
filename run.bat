@echo off
rem Launch the OpenBCI x BioProxal flow validation dashboard.
rem Usage: run.bat [app.py arguments], e.g. run.bat --sim   or   run.bat --band ble
cd /d %~dp0

python -c "import numpy, serial" >nul 2>&1
if errorlevel 1 (
    echo Missing numpy or pyserial. Install the dependencies first:
    echo     python -m pip install -r requirements.txt
    exit /b 1
)

python app.py %*
exit /b %errorlevel%
