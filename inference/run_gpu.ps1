param(
    [string]$PythonExe = "C:\Tools\anaconda3\python.exe",
    [string]$ModelPath = "",
    [int]$MaxNewTokens = 256,
    [string]$ModelDeviceMap = "auto"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ModelPath) {
    $ModelPath = $repoRoot
}

$hfHome = Join-Path $repoRoot ".hf_cache"
if (-not (Test-Path $hfHome)) {
    New-Item -ItemType Directory -Force -Path $hfHome | Out-Null
}

$env:HF_HOME = $hfHome
$env:MODEL_PATH = $ModelPath
$env:MAX_NEW_TOKENS = [string]$MaxNewTokens
$env:MODEL_DEVICE_MAP = $ModelDeviceMap

Write-Host "[INFO] PythonExe=$PythonExe"
Write-Host "[INFO] MODEL_PATH=$env:MODEL_PATH"
Write-Host "[INFO] MAX_NEW_TOKENS=$env:MAX_NEW_TOKENS"
Write-Host "[INFO] MODEL_DEVICE_MAP=$env:MODEL_DEVICE_MAP"
Write-Host "[INFO] HF_HOME=$env:HF_HOME"

& $PythonExe (Join-Path $PSScriptRoot "generate.py")
