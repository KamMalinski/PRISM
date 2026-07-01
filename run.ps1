$ErrorActionPreference = "Stop"

$pythonVenv = ".\.venv\Scripts\python.exe"

function Get-SystemPython {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $exe = & $launcher.Source -3 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) {
            return $exe.Trim()
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $exe = & $python.Source -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) {
            return $exe.Trim()
        }
    }

    throw "Could not find a working Python 3 installation."
}

function New-Venv {
    param([string]$systemPython)

    $root = (Resolve-Path .).Path
    $venvPath = Join-Path $root ".venv"
    if (Test-Path $venvPath) {
        $target = (Resolve-Path $venvPath).Path
        if (-not $target.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw ".venv path is outside the project directory: $target"
        }
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    & $systemPython -m venv .venv
}

$systemPython = Get-SystemPython
$venvOk = $false

if (Test-Path $pythonVenv) {
    try {
        & $pythonVenv --version *> $null
        $venvOk = ($LASTEXITCODE -eq 0)
    } catch {
        $venvOk = $false
    }
}

if (-not $venvOk) {
    New-Venv $systemPython
}

& $pythonVenv -m pip install -r requirements.txt
$env:PYTHONPATH = "$(Get-Location)\src"
& $pythonVenv -m schematic_generator
