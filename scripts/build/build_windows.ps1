$ErrorActionPreference = "Stop"

$bootstrap = Join-Path $PSScriptRoot "bootstrap.py"
$launcher = Get-Command py -ErrorAction SilentlyContinue

if ($launcher) {
    & $launcher.Source -3 $bootstrap --platform windows
} else {
    $python = Get-Command python -ErrorAction Stop
    & $python.Source $bootstrap --platform windows
}

exit $LASTEXITCODE
