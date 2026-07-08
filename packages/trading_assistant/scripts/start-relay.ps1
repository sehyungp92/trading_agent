# Start the local trading relay as a supervised hidden background process.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start-relay.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CommonScript = Join-Path $ProjectRoot "scripts\start-common.ps1"

if (-not (Test-Path $CommonScript)) {
    throw "Missing shared startup helpers at $CommonScript"
}

. $CommonScript
Import-AssistantEnvFile -EnvFile (Join-Path $ProjectRoot ".env")

$RelayAppDir = Join-Path $ProjectRoot "src"
if (-not (Test-Path (Join-Path $RelayAppDir "trading_assistant\relay_ingress\app.py"))) {
    throw "Missing assistant relay ingress under $RelayAppDir"
}

$LogDir = Join-Path $ProjectRoot "logs"
$RunDir = Join-Path $ProjectRoot "run"
$DataDir = Join-Path $ProjectRoot "data"
$LogFile = Join-Path $LogDir "relay.log"
$ErrFile = Join-Path $LogDir "relay.err.log"
$PidFile = Join-Path $RunDir "relay.pid"
$LockFile = Join-Path $RunDir "relay.supervisor.lock"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

$SupervisorLock = Enter-OrchestratorSupervisorLock -LockFile $LockFile
if (-not $SupervisorLock) {
    Write-Host "Trading relay supervisor already running."
    return
}

try {
    $Pythonw = Resolve-OrchestratorPythonw -ProjectRoot $ProjectRoot
    if (-not $Pythonw) {
        $message = "No usable pythonw interpreter found. Checked package and repo .venv, venv, then PATH."
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $line = "[$timestamp] ERROR: $message"
        Add-Content -Path $ErrFile -Value $line -Encoding ASCII
        throw $message
    }

    $RelayHost = if ($env:RELAY_HOST) { $env:RELAY_HOST } else { "127.0.0.1" }
    $RelayPort = if ($env:RELAY_PORT) { $env:RELAY_PORT } else { "8001" }
    $HealthHost = if ($RelayHost -eq "0.0.0.0") { "127.0.0.1" } else { $RelayHost }
    $RelayHealthUrl = "http://{0}:{1}/health" -f $HealthHost, $RelayPort

    if (-not $env:RELAY_DB_PATH) {
        $env:RELAY_DB_PATH = Join-Path $DataDir "relay.db"
    }
    if (-not $env:TRADING_MODE -and -not $env:TRADING_ENV) {
        $env:TRADING_MODE = "paper"
    }
    $TradingMode = if ($env:TRADING_MODE) { $env:TRADING_MODE.ToLowerInvariant() } elseif ($env:TRADING_ENV) { $env:TRADING_ENV.ToLowerInvariant() } else { "paper" }
    $RelayNetworkMode = if ($env:RELAY_NETWORK_MODE) { $env:RELAY_NETWORK_MODE.ToLowerInvariant() } else { "" }
    $LoopbackRelayHosts = @("127.0.0.1", "localhost", "::1")
    $LoopbackAllowedModes = @("local_direct", "private_interface", "secure_tunnel", "tunnel")
    $AllowsLoopbackRelay = ($env:ALLOW_LOOPBACK_RELAY -eq "1") -or ($LoopbackAllowedModes -contains $RelayNetworkMode)
    if (($TradingMode -in @("paper", "live", "prod", "production")) -and ($LoopbackRelayHosts -contains $RelayHost.ToLowerInvariant()) -and (-not $AllowsLoopbackRelay)) {
        throw "RELAY_HOST must not be loopback in $TradingMode mode unless RELAY_NETWORK_MODE is local_direct/private_interface/secure_tunnel/tunnel or ALLOW_LOOPBACK_RELAY=1."
    }

    $pythonPathEntries = @(
        $RelayAppDir
    )
    if ($env:PYTHONPATH) {
        $pythonPathEntries += $env:PYTHONPATH
    }
    $env:PYTHONPATH = ($pythonPathEntries -join [System.IO.Path]::PathSeparator)

    if (Test-RelayAlreadyRunning -PidFile $PidFile -HealthUrl $RelayHealthUrl) {
        Write-Host "Trading relay already running and healthy."
        return
    }

    $Arguments = @(
        "-m", "uvicorn", "trading_assistant.relay_ingress.app:app",
        "--app-dir", $RelayAppDir,
        "--host", $RelayHost,
        "--port", $RelayPort
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
        Write-Host "[$timestamp] Started trading relay supervisor child (PID $($proc.Id)) using $Pythonw"

        if (-not (Wait-OrchestratorHealthy -TimeoutSeconds 60 -Url $RelayHealthUrl)) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Write-Host "[$timestamp] Relay health check failed. Restarting in $RestartDelaySeconds second(s)."
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
        Write-Host "[$timestamp] Trading relay exited with code $exitCode. Restarting in $RestartDelaySeconds second(s)."
        Start-Sleep -Seconds $RestartDelaySeconds
        $RestartDelaySeconds = [Math]::Min($RestartDelaySeconds * 2, $MaxRestartDelaySeconds)
    }
} finally {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Exit-OrchestratorSupervisorLock -LockHandle $SupervisorLock -LockFile $LockFile
}
