$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pidFile = Join-Path $projectRoot "logs\platform.pid"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "platform.pid not found. YOLO Platform may already be stopped."
} else {
    $pidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($pidText) {
        $platformPid = [int]$pidText
        $process = Get-Process -Id $platformPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-Host "Process $platformPid is not running. Removed stale pid file."
        } else {
            & taskkill /F /T /PID $platformPid *>&1 | Out-Null
            Start-Sleep -Milliseconds 300
            Write-Host "YOLO Platform process $platformPid stopped."
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
