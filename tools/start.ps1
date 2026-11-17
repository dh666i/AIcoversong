param([switch]$NoOpen, [switch]$Full)
# Starts the page. Default: the Chinese page (cover_app.py). -Full: the complete Applio interface (app.py).
# This file is ASCII on purpose: Windows PowerShell 5 reads scripts without a BOM in the ANSI code page.
$ErrorActionPreference = 'Stop'
$coverRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $coverRoot 'env\Scripts\python.exe'
$appPath = Join-Path $coverRoot 'Applio'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment is missing: $pythonPath. Run tools\setup.ps1 first."
}
$logsPath = Join-Path $coverRoot 'logs'
New-Item -ItemType Directory -Path $logsPath -Force | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = Join-Path $coverRoot 'cache\huggingface'
$env:TORCH_HOME = Join-Path $coverRoot 'cache\torch'
$env:AUDIO_SEPARATOR_MODEL_DIR = Join-Path $appPath 'rvc\models\uvr'
$env:GRADIO_TEMP_DIR = Join-Path $coverRoot 'cache\gradio'
$env:NUMBA_CACHE_DIR = Join-Path $coverRoot 'cache\numba'
$env:UV_CACHE_DIR = Join-Path $coverRoot 'cache\uv'
$env:TEMP = Join-Path $coverRoot 'cache\temp'
$env:TMP = $env:TEMP
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:ALL_PROXY = ''
$env:NO_PROXY = '*'
$env:PATH = "$appPath;$env:PATH"
foreach ($cachePath in @($env:HF_HOME, $env:TORCH_HOME, $env:GRADIO_TEMP_DIR, $env:NUMBA_CACHE_DIR, $env:TEMP)) {
    New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
}
Set-Location -LiteralPath $appPath
$existing = $null
try {
    $existing = Invoke-WebRequest -Uri 'http://127.0.0.1:6969/config' -UseBasicParsing -TimeoutSec 2
} catch { }
if ($existing -and $existing.StatusCode -eq 200) {
    # 'cover-recording-prompt' is on the Chinese page, 'applio-shell' on the complete Applio interface.
    $runningFull = $existing.Content.Contains('applio-shell')
    $runningCover = $existing.Content.Contains('cover-recording-prompt')
    if (($Full -and $runningFull) -or (-not $Full -and $runningCover)) {
        Write-Host 'Already running: http://127.0.0.1:6969'
        if (-not $NoOpen) { Start-Process 'http://127.0.0.1:6969' }
        exit 0
    }
    if ($runningFull -or $runningCover) {
        Write-Host 'The other AIcoversong interface is running on port 6969. Stop it first (tools\stop.ps1), then start again.'
        exit 1
    }
}
$PID | Set-Content -LiteralPath (Join-Path $coverRoot 'logs\server-launcher.pid')
$script = if ($Full) { 'app.py' } else { 'cover_app.py' }
$appArguments = @('-u', $script, '--server-name', '127.0.0.1', '--port', '6969')
if (-not $NoOpen) { $appArguments += '--open' }
& $pythonPath @appArguments
exit $LASTEXITCODE
