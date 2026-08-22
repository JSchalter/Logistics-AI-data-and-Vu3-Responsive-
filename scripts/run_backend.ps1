$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt"
}

$env:PYTHONPATH = $projectRoot
& $python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
