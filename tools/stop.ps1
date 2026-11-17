$ErrorActionPreference = 'Stop'
$coverRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $coverRoot 'logs\server-launcher.pid'
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host 'No AIcoversong server was started by this launcher.'
    exit 0
}
$serverLauncherPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$serverLauncherPid"
$expectedScript = Join-Path $PSScriptRoot 'start.ps1'
if (-not $serverProcess) {
    Write-Host 'AIcoversong is already stopped.'
    exit 0
}
if (-not $serverProcess.CommandLine -or $serverProcess.CommandLine.IndexOf($expectedScript, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw 'The saved process is not this AIcoversong launcher. No process was stopped.'
}
& taskkill.exe /PID $serverLauncherPid /T /F
if ($LASTEXITCODE -ne 0) { throw 'Could not stop AIcoversong.' }
Write-Host 'AIcoversong stopped.'
