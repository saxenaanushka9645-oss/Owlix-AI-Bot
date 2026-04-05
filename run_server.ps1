# Owlix — Start FastAPI backend on http://127.0.0.1:8000
# Run from the project folder (where main.py lives), or double-click.
Set-Location $PSScriptRoot

# Activate venv if it exists one level up
$venvPy = "..\venv\Scripts\python.exe"
$localPy = "python"
$py = if (Test-Path $venvPy) { $venvPy } else { $localPy }

Write-Host "Starting Owlix backend..." -ForegroundColor Cyan
& $py -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
