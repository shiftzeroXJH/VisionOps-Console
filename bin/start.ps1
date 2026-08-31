$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$frontendRoot = Join-Path $projectRoot "frontend"
$logDir = Join-Path $projectRoot "logs"
$backendPidFile = Join-Path $logDir "dev-backend.pid"
$frontendPidFile = Join-Path $logDir "dev-frontend.pid"
$srcPath = Join-Path $projectRoot "src"
$dbPath = Join-Path $projectRoot "yolo_state.sqlite"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Test-ManagedProcess {
    param(
        [string]$PidFile,
        [string[]]$ExpectedProcessNames
    )

    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $false
    }

    $pidText = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if (-not $pidText) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return $false
    }

    $process = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return $false
    }

    if ($ExpectedProcessNames -notcontains $process.ProcessName) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        return $false
    }

    return $true
}

function Get-ConfiguredPython {
    param(
        [string]$ProjectRoot,
        [string]$DbPath
    )

    if ($env:YOLO_PYTHON) {
        return $env:YOLO_PYTHON
    }

    if (Test-Path -LiteralPath $DbPath) {
        try {
            $pythonCommand = Get-Command python.exe -ErrorAction Stop
            $reader = Join-Path $ProjectRoot 'bin\read-yolo-python.py'
            $configured = & $pythonCommand.Source $reader $DbPath 2>$null
            if ($LASTEXITCODE -eq 0 -and $configured) {
                return ([string]($configured -join '')).Trim()
            }
            throw 'The global yolo_python setting could not be read'
        } catch {
            throw "Unable to read global yolo_python setting from $DbPath : $($_.Exception.Message)"
        }
    }

    # Before the first global setting is saved, let Python resolve from PATH.
    return "python"
}

$python = Get-ConfiguredPython -ProjectRoot $projectRoot -DbPath $dbPath

if (-not (Test-ManagedProcess -PidFile $backendPidFile -ExpectedProcessNames @("powershell", "pwsh"))) {
    $backendCommand = @(
        "`$env:YOLO_DB_PATH='$dbPath'"
        "`$env:YOLO_HOST='127.0.0.1'"
        "`$env:YOLO_PORT='8765'"
        "`$env:YOLO_PYTHON='$python'"
        "`$env:PYTHONPATH='$srcPath'"
        "Set-Location '$projectRoot'"
        "& '$python' -m backend.api"
    ) -join "; "
    $backend = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) -RedirectStandardOutput (Join-Path $logDir "dev-backend.stdout.log") -RedirectStandardError (Join-Path $logDir "dev-backend.stderr.log") -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $backendPidFile -Value $backend.Id -Encoding ascii
}

if (-not (Test-ManagedProcess -PidFile $frontendPidFile -ExpectedProcessNames @("cmd", "node", "npm"))) {
    $frontend = Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") -WorkingDirectory $frontendRoot -RedirectStandardOutput (Join-Path $logDir "dev-frontend.stdout.log") -RedirectStandardError (Join-Path $logDir "dev-frontend.stderr.log") -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $frontendPidFile -Value $frontend.Id -Encoding ascii
}

Write-Host "Using Python: $python"

Write-Host "Development servers started."
Write-Host "Frontend: http://127.0.0.1:5173/"
Write-Host "Backend:  http://127.0.0.1:8765/"
