# Start the trading assistant orchestrator as a supervised hidden background process.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CommonScript = Join-Path $ProjectRoot "scripts\start-common.ps1"

if (-not (Test-Path $CommonScript)) {
    throw "Missing shared startup helpers at $CommonScript"
}

. $CommonScript
Import-AssistantEnvFile -EnvFile (Join-Path $ProjectRoot ".env")
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
$RunDir = Join-Path $ProjectRoot "run"
$LogFile = Join-Path $LogDir "orchestrator.log"
$ErrFile = Join-Path $LogDir "orchestrator.err.log"
$PidFile = Join-Path $RunDir "orchestrator.pid"
$LockFile = Join-Path $RunDir "orchestrator.supervisor.lock"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

$SupervisorLock = Enter-OrchestratorSupervisorLock -LockFile $LockFile
if (-not $SupervisorLock) {
    Write-Host "Trading assistant supervisor already running."
    return
}

try {
    $Pythonw = Resolve-OrchestratorPythonw -ProjectRoot $ProjectRoot
    if (-not $Pythonw) {
        $message = "No usable pythonw interpreter found. Checked .venv, venv, then PATH."
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $line = "[$timestamp] ERROR: $message"
        Add-Content -Path $ErrFile -Value $line -Encoding ASCII
        throw $message
    }

    if (Test-OrchestratorAlreadyRunning -PidFile $PidFile) {
        Write-Host "Trading assistant already running and healthy."
        return
    }

    $UvicornHost = "127.0.0.1"
    $env:BIND_HOST = $UvicornHost
    $env:UVICORN_HOST = $UvicornHost

    $Arguments = @(
        "-m", "uvicorn", "trading_assistant.orchestrator.app:app",
        "--host", $UvicornHost,
        "--port", "8000"
    )

    $RestartDelaySeconds = 5
    $MaxRestartDelaySeconds = 60

    while ($true) {
        Rotate-OrchestratorLog -LogPath $LogFile
        Rotate-OrchestratorLog -LogPath $ErrFile

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $proc = Start-Process `
            -WindowStyle Hidden `
            -FilePath $Pythonw `
            -ArgumentList $Arguments `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError $ErrFile `
            -PassThru

        Set-Content -Path $PidFile -Value $proc.Id -Encoding ASCII
        Write-Host "[$timestamp] Started trading assistant supervisor child (PID $($proc.Id)) using $Pythonw"

        if (-not (Wait-OrchestratorHealthy -TimeoutSeconds 60)) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "[$timestamp] Health check failed. Restarting in $RestartDelaySeconds second(s)."
            if (-not $proc.HasExited) {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds $RestartDelaySeconds
            $RestartDelaySeconds = [Math]::Min($RestartDelaySeconds * 2, $MaxRestartDelaySeconds)
            continue
        }

        $RestartDelaySeconds = 5
        Wait-Process -Id $proc.Id

        $exitCode = 0
        try {
            $proc.Refresh()
            $exitCode = $proc.ExitCode
        } catch {
        }

        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] Trading assistant exited with code $exitCode. Restarting in $RestartDelaySeconds second(s)."
        Start-Sleep -Seconds $RestartDelaySeconds
        $RestartDelaySeconds = [Math]::Min($RestartDelaySeconds * 2, $MaxRestartDelaySeconds)
    }
} finally {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Exit-OrchestratorSupervisorLock -LockHandle $SupervisorLock -LockFile $LockFile
}
