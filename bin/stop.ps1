$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pidFiles = @(
    (Join-Path $projectRoot "logs\dev-frontend.pid"),
    (Join-Path $projectRoot "logs\dev-backend.pid")
)

foreach ($pidFile in $pidFiles) {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        continue
    }
    $pidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($pidText) {
        $process = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            if ($process.ProcessName -in @("powershell", "pwsh", "cmd", "node", "npm")) {
                & taskkill /F /T /PID ([int]$pidText) *>&1 | Out-Null
                Write-Host "Stopped process $pidText."
            } else {
                Write-Host "Skipped stale pid $pidText ($($process.ProcessName))."
            }
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
