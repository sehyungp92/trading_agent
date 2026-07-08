# Shared helpers for starting the trading assistant services on Windows.

$Script:OrchestratorHealthUrl = "http://127.0.0.1:8000/ready"
$Script:OrchestratorCommandLinePattern = "trading_assistant.orchestrator.app:app"
$Script:RelayHealthUrl = "http://127.0.0.1:8001/health"
$Script:RelayCommandLinePattern = "trading_assistant.relay_ingress.app:app"

function Import-AssistantEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvFile
    )

    if (-not (Test-Path $EnvFile)) {
        return
    }

    foreach ($line in Get-Content -Path $EnvFile -ErrorAction Stop) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $separator = $trimmed.IndexOf("=")
        if ($separator -le 0) {
            continue
        }

        $key = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        if (-not $key) {
            continue
        }

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Enter-OrchestratorSupervisorLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockFile
    )

    $lockDir = Split-Path -Parent $LockFile
    if ($lockDir) {
        New-Item -ItemType Directory -Path $lockDir -Force | Out-Null
    }

    try {
        return [System.IO.File]::Open(
            $LockFile,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        return $null
    }
}

function Exit-OrchestratorSupervisorLock {
    param(
        [AllowNull()]
        [object]$LockHandle,
        [Parameter(Mandatory = $true)]
        [string]$LockFile
    )

    if ($LockHandle) {
        try {
            $LockHandle.Close()
            $LockHandle.Dispose()
        } catch {
        }
    }

    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}

function Resolve-OrchestratorPythonw {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $repoRoot = $null
    try {
        $repoRoot = Split-Path -Parent (Split-Path -Parent $ProjectRoot)
    } catch {
        $repoRoot = $null
    }

    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\pythonw.exe")
    )

    if ($repoRoot -and $repoRoot -ne $ProjectRoot) {
        $candidates += @(
            (Join-Path $repoRoot ".venv\Scripts\pythonw.exe"),
            (Join-Path $repoRoot "venv\Scripts\pythonw.exe")
        )
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $command = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command pythonw -ErrorAction SilentlyContinue
    }
    if ($command) {
        return $command.Source
    }

    return $null
}

function Rotate-OrchestratorLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [int]$MaxSizeMB = 10,
        [int]$MaxFiles = 5
    )

    if (-not (Test-Path $LogPath)) {
        return
    }

    $file = Get-Item $LogPath
    if ($file.Length -lt ($MaxSizeMB * 1MB)) {
        return
    }

    for ($i = $MaxFiles; $i -ge 1; $i--) {
        $src = "$LogPath.$i"
        if ($i -eq $MaxFiles) {
            if (Test-Path $src) {
                Remove-Item $src -Force
            }
        } else {
            $dst = "$LogPath.$($i + 1)"
            if (Test-Path $src) {
                Rename-Item $src $dst -Force
            }
        }
    }

    Rename-Item $LogPath "$LogPath.1" -Force
}

function Test-OrchestratorHealthy {
    param(
        [string]$Url = $Script:OrchestratorHealthUrl,
        [int]$TimeoutSeconds = 2
    )

    try {
        $response = Invoke-WebRequest `
            -Uri $Url `
            -TimeoutSec $TimeoutSeconds `
            -UseBasicParsing `
            -ErrorAction Stop
        if ($response.StatusCode -ne 200) {
            return $false
        }
        $content = $response.Content
        if ($content -is [byte[]]) {
            $content = [System.Text.Encoding]::UTF8.GetString($content)
        }
        $payload = $content | ConvertFrom-Json -ErrorAction Stop
        return $payload.status -eq "ok"
    } catch {
        return $false
    }
}

function Wait-OrchestratorHealthy {
    param(
        [int]$TimeoutSeconds = 60,
        [string]$Url = $Script:OrchestratorHealthUrl
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-OrchestratorHealthy -Url $Url) {
            return $true
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Get-OrchestratorProcessFromPidFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        return $null
    }

    try {
        $pidText = (Get-Content $PidFile -Raw -ErrorAction Stop).Trim()
        if (-not $pidText) {
            return $null
        }
        return Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

function Get-OrchestratorProcessMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    } catch {
        return $null
    }
}

function Test-OrchestratorProcessCommandLine {
    param(
        [string]$ProcessName,
        [string]$CommandLine,
        [string]$AppPattern = $Script:OrchestratorCommandLinePattern
    )

    if (-not $commandLine) {
        return $false
    }

    $normalizedName = $ProcessName.ToLowerInvariant()
    $isPythonProcess = $normalizedName -eq "python.exe" -or $normalizedName -eq "pythonw.exe"
    if (-not $isPythonProcess) {
        return $false
    }

    $normalizedCommandLine = $commandLine.ToLowerInvariant()
    $hasUvicornCommand = $normalizedCommandLine -match "(^|\\s)-m\\s+uvicorn(\\s|$)"
    $hasTargetApp = $normalizedCommandLine.Contains(
        $AppPattern.ToLowerInvariant()
    )
    return ($hasUvicornCommand -and $hasTargetApp)
}

function Test-OrchestratorProcessRecordMatches {
    param(
        [AllowNull()]
        [object]$ProcessRecord
    )

    if (-not $ProcessRecord) {
        return $false
    }

    return Test-OrchestratorProcessCommandLine `
        -ProcessName ([string]$ProcessRecord.Name) `
        -CommandLine ([string]$ProcessRecord.CommandLine)
}

function Find-OrchestratorProcess {
    try {
        return Get-CimInstance Win32_Process `
            -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" `
            -ErrorAction Stop |
            Where-Object { Test-OrchestratorProcessRecordMatches $_ } |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function Test-OrchestratorAlreadyRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile,
        [string]$HealthUrl = $Script:OrchestratorHealthUrl
    )

    $process = Get-OrchestratorProcessFromPidFile -PidFile $PidFile
    if ($process) {
        if (Test-OrchestratorHealthy -Url $HealthUrl) {
            return $true
        }
    }

    if ((-not $process) -and (Test-Path $PidFile)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-OrchestratorHealthy -Url $HealthUrl)) {
        return $false
    }

    $fallbackProcess = Find-OrchestratorProcess
    if ($fallbackProcess) {
        Set-Content -Path $PidFile -Value $fallbackProcess.ProcessId -Encoding ASCII -ErrorAction SilentlyContinue
        return $true
    }

    return $false
}

function Test-RelayProcessRecordMatches {
    param(
        [AllowNull()]
        [object]$ProcessRecord
    )

    if (-not $ProcessRecord) {
        return $false
    }

    return Test-OrchestratorProcessCommandLine `
        -ProcessName ([string]$ProcessRecord.Name) `
        -CommandLine ([string]$ProcessRecord.CommandLine) `
        -AppPattern $Script:RelayCommandLinePattern
}

function Find-RelayProcess {
    try {
        return Get-CimInstance Win32_Process `
            -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" `
            -ErrorAction Stop |
            Where-Object { Test-RelayProcessRecordMatches $_ } |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function Test-RelayAlreadyRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile,
        [string]$HealthUrl = $Script:RelayHealthUrl
    )

    $process = Get-OrchestratorProcessFromPidFile -PidFile $PidFile
    if ($process) {
        if (Test-OrchestratorHealthy -Url $HealthUrl) {
            return $true
        }
    }

    if ((-not $process) -and (Test-Path $PidFile)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-OrchestratorHealthy -Url $HealthUrl)) {
        return $false
    }

    $fallbackProcess = Find-RelayProcess
    if ($fallbackProcess) {
        Set-Content -Path $PidFile -Value $fallbackProcess.ProcessId -Encoding ASCII -ErrorAction SilentlyContinue
        return $true
    }

    return $true
}
