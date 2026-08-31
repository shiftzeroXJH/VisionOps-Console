$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$logDir = Join-Path $projectRoot "logs"
$pidFile = Join-Path $logDir "platform.pid"
$stdoutLog = Join-Path $logDir "platform.stdout.log"
$stderrLog = Join-Path $logDir "platform.stderr.log"
$dbPath = Join-Path $projectRoot "yolo_state.sqlite"
$srcPath = Join-Path $projectRoot "src"
$frontendIndex = Join-Path $projectRoot "frontend\dist\index.html"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-ConfiguredPython {
    param(
        [string]$DbPath
    )

    if ($env:YOLO_PYTHON) {
        return $env:YOLO_PYTHON
    }

    if (Test-Path -LiteralPath $DbPath) {
        try {
            $pythonCommand = Get-Command python.exe -ErrorAction Stop
            $reader = Join-Path (Split-Path -Parent $DbPath) 'bin\read-yolo-python.py'
            $configured = & $pythonCommand.Source $reader $DbPath 2>$null
            if ($LASTEXITCODE -eq 0 -and $configured) {
                return ([string]($configured -join '')).Trim()
            }
            throw 'The global yolo_python setting could not be read'
        } catch {
            throw "Unable to read global yolo_python setting from $DbPath : $($_.Exception.Message)"
        }
    }

    return "python"
}

if (-not (Test-Path -LiteralPath $frontendIndex)) {
    Write-Host "Frontend build not found: $frontendIndex"
    Write-Host "Run: cd frontend && npm ci && npm run build"
    exit 1
}

if (Test-Path -LiteralPath $pidFile) {
    $existingPidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($existingPidText) {
        $existingProcess = Get-Process -Id ([int]$existingPidText) -ErrorAction SilentlyContinue
        if ($null -ne $existingProcess) {
            Write-Host "YOLO Platform is already running."
            Write-Host "PID: $existingPidText"
            Write-Host "URL: http://127.0.0.1:8765/"
            exit 0
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

$python = Get-ConfiguredPython -DbPath $dbPath

$command = @(
    "`$env:YOLO_DB_PATH='$dbPath'"
    "`$env:YOLO_HOST='0.0.0.0'"
    "`$env:YOLO_PORT='8765'"
    "`$env:YOLO_PYTHON='$python'"
    "`$env:PYTHONPATH='$srcPath'"
    "Set-Location '$projectRoot'"
    "& '$python' -m backend.api"
) -join "; "

$process = Start-Process `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

Write-Host "YOLO Platform started."
Write-Host "Python: $python"
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://127.0.0.1:8765/"
Write-Host "DB: $dbPath"
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
