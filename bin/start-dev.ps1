$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$frontendRoot = Join-Path $projectRoot "frontend"
$logDir = Join-Path $projectRoot "logs"
$backendPidFile = Join-Path $logDir "dev-backend.pid"
$frontendPidFile = Join-Path $logDir "dev-frontend.pid"
$srcPath = Join-Path $projectRoot "src"
$dbPath = Join-Path $projectRoot "yolo_state.sqlite"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$python = $env:YOLO_PYTHON
if (-not $python) {
    $python = "python"
}

if (-not (Test-Path -LiteralPath $backendPidFile)) {
    $backendCommand = @(
        "`$env:YOLO_DB_PATH='$dbPath'"
        "`$env:YOLO_HOST='127.0.0.1'"
        "`$env:YOLO_PORT='8765'"
        "`$env:PYTHONPATH='$srcPath'"
        "Set-Location '$projectRoot'"
        "& '$python' -m backend.api"
    ) -join "; "
    $backend = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) -RedirectStandardOutput (Join-Path $logDir "dev-backend.stdout.log") -RedirectStandardError (Join-Path $logDir "dev-backend.stderr.log") -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $backendPidFile -Value $backend.Id -Encoding ascii
}

if (-not (Test-Path -LiteralPath $frontendPidFile)) {
    $frontend = Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") -WorkingDirectory $frontendRoot -RedirectStandardOutput (Join-Path $logDir "dev-frontend.stdout.log") -RedirectStandardError (Join-Path $logDir "dev-frontend.stderr.log") -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $frontendPidFile -Value $frontend.Id -Encoding ascii
}

Write-Host "Development servers started."
Write-Host "Frontend: http://127.0.0.1:5173/"
Write-Host "Backend:  http://127.0.0.1:8765/"
