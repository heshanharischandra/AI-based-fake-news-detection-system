$pythonPath = Join-Path $PSScriptRoot ".venv311/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error "Python environment not found at $pythonPath"
    exit 1
}
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
