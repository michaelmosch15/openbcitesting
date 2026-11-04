# Launch the OpenBCI x BioProxal flow validation dashboard.
# Usage: .\run.ps1 [app.py arguments], e.g. .\run.ps1 --sim   or   .\run.ps1 --band ble
Set-Location $PSScriptRoot

python -c "import numpy, serial" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing numpy or pyserial. Install the dependencies first:"
    Write-Host "    python -m pip install -r requirements.txt"
    exit 1
}

python app.py @args
exit $LASTEXITCODE
