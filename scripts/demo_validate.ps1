$env:DATA_MODE='demo'
$env:PYTHONPATH=(Resolve-Path "$PSScriptRoot\..").Path
python scripts/train.py
python -m pytest -q
