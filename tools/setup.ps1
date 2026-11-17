param([switch]$DownloadModels)
# ASCII for Windows PowerShell 5 compatibility.
$ErrorActionPreference = 'Stop'
$installRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $installRoot
if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: winget install --id astral-sh.uv --exact ; then open a new terminal.'
}
if (-not (Test-Path -LiteralPath (Join-Path $installRoot 'Applio\cover_app.py'))) {
    throw 'Application source is missing. Extract or clone the complete AIcoversong repository.'
}
$env:UV_CACHE_DIR = Join-Path $installRoot 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $installRoot 'runtime\python'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
foreach ($relative in @('logs', 'cache', 'output', 'recordings\MyVoice')) {
    New-Item -ItemType Directory -Path (Join-Path $installRoot $relative) -Force | Out-Null
}
function Invoke-Uv([string[]]$Arguments) {
    & uv @Arguments
    if ($LASTEXITCODE -ne 0) { throw "uv failed ($LASTEXITCODE). Setup stopped; existing recordings and models were not removed." }
}
Invoke-Uv @('python', 'install', '3.12.10')
foreach ($name in @('env', 'separator-env')) {
    $envPython = Join-Path $installRoot "$name\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $envPython)) {
        Invoke-Uv @('venv', '--python', '3.12.10', (Join-Path $installRoot $name))
    }
    & $envPython -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
    if ($LASTEXITCODE -ne 0) { throw "Unexpected Python version in $name. Use a fresh installation directory." }
}
$appPython = Join-Path $installRoot 'env\Scripts\python.exe'
$separatorPython = Join-Path $installRoot 'separator-env\Scripts\python.exe'
# Install CUDA wheels from PyTorch, not an arbitrary additional package index.
Invoke-Uv @('pip', 'install', '--python', $appPython, '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.11.0+cu128', 'torchaudio==2.11.0+cu128')
Invoke-Uv @('pip', 'install', '--python', $separatorPython, '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.11.0+cu128', 'torchvision==0.26.0+cu128')
Invoke-Uv @('pip', 'install', '--python', $appPython, '-r', 'requirements\app-windows-py312.txt')
Invoke-Uv @('pip', 'install', '--python', $separatorPython, '-r', 'requirements\separator-windows-py312.txt')
Invoke-Uv @('pip', 'check', '--python', $appPython)
Invoke-Uv @('pip', 'check', '--python', $separatorPython)
& $appPython -c 'import torch,gradio,soundfile; print("UI",gradio.__version__,"Torch",torch.__version__,"CUDA",torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Application import check failed.' }
& $separatorPython -c 'from audio_separator.separator import Separator; import torch; print("Separator ready; CUDA",torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Separation environment import check failed.' }
if ($DownloadModels) {
    Write-Host 'Downloading public model/tool assets. Model-specific terms: docs\MODELS.md'
    & $appPython (Join-Path $PSScriptRoot 'download_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Asset download did not complete. Re-run tools\download_assets.py to resume.' }
} else {
    Write-Host 'Dependencies installed. Download assets with env\Scripts\python.exe tools\download_assets.py before processing audio.'
}
Write-Host 'Setup complete. Start with tools\start.ps1 or the start batch file.'
