param(
    [Parameter(Mandatory = $true)]
    [string]$TasksFile,
    [Parameter(Mandatory = $true)]
    [string]$ResultsFile,
    [int]$StartAt = 0,
    [int]$MaxTasks = 0,
    [int]$TimeoutSeconds = 900,
    [int]$MaxRetries = 2,
    [switch]$RetryErrors
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Worker = Join-Path $Root "scripts\analyze_oos_repair.py"
$TaskPathResolved = (Resolve-Path $TasksFile).Path
$ResultsDir = Split-Path -Parent $ResultsFile
if ($ResultsDir) {
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
}
$TmpDir = Join-Path $Root "tmp\oos_repair_worker_tasks"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

function Quote-Arg([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Write-JsonLine([string]$Path, [string]$Json) {
    [System.IO.File]::AppendAllText($Path, $Json + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
}

$Lines = [System.IO.File]::ReadAllLines($TaskPathResolved)
$CompletedSuccess = @{}
$SeenLines = 0
if (Test-Path $ResultsFile) {
    $ExistingLines = [System.IO.File]::ReadAllLines((Resolve-Path $ResultsFile).Path)
    $SeenLines = $ExistingLines.Count
    for ($LineIndex = 0; $LineIndex -lt $ExistingLines.Count; $LineIndex++) {
        if ([string]::IsNullOrWhiteSpace($ExistingLines[$LineIndex])) { continue }
        try {
            $ExistingObject = $ExistingLines[$LineIndex] | ConvertFrom-Json
            if ($null -ne $ExistingObject.task_index) {
                $ExistingIndex = [int]$ExistingObject.task_index
            } else {
                $ExistingIndex = $LineIndex
            }
            $ExistingError = ""
            if ($null -ne $ExistingObject.error) {
                $ExistingError = [string]$ExistingObject.error
            }
            if ($ExistingError.Length -eq 0 -or -not $RetryErrors) {
                $CompletedSuccess[$ExistingIndex] = $true
            }
        } catch {}
    }
}
$Start = [Math]::Max($StartAt, 0)
$End = $Lines.Count
if ($MaxTasks -gt 0) {
    $End = [Math]::Min($End, $Start + $MaxTasks)
}

Write-Host "running worker JSONL tasks $Start..$($End - 1) of $($Lines.Count); existing_lines=$SeenLines; successful_done=$($CompletedSuccess.Count); retry_errors=$RetryErrors"
for ($Index = $Start; $Index -lt $End; $Index++) {
    if ($CompletedSuccess.ContainsKey($Index)) {
        continue
    }
    $Token = [Guid]::NewGuid().ToString("N")
    $TaskFile = Join-Path $TmpDir "$Token.task.json"
    $ResultFile = Join-Path $TmpDir "$Token.result.json"
    $StdoutFile = Join-Path $TmpDir "$Token.stdout.txt"
    $StderrFile = Join-Path $TmpDir "$Token.stderr.txt"
    $Success = $false
    $LastError = ""
    for ($Attempt = 0; $Attempt -le $MaxRetries; $Attempt++) {
        [System.IO.File]::WriteAllText($TaskFile, $Lines[$Index], [System.Text.Encoding]::UTF8)

        $Psi = New-Object System.Diagnostics.ProcessStartInfo
        $Psi.FileName = "python"
        $Psi.WorkingDirectory = $Root
        $Psi.UseShellExecute = $false
        $Psi.CreateNoWindow = $true
        $Psi.RedirectStandardOutput = $true
        $Psi.RedirectStandardError = $true
        $Psi.Arguments = @(
            (Quote-Arg $Worker),
            "--worker-task-file",
            (Quote-Arg $TaskFile),
            "--worker-output-file",
            (Quote-Arg $ResultFile)
        ) -join " "

        $Process = [System.Diagnostics.Process]::Start($Psi)
        $Completed = $Process.WaitForExit($TimeoutSeconds * 1000)
        $Stdout = $Process.StandardOutput.ReadToEnd()
        $Stderr = $Process.StandardError.ReadToEnd()
        [System.IO.File]::WriteAllText($StdoutFile, $Stdout, [System.Text.Encoding]::UTF8)
        [System.IO.File]::WriteAllText($StderrFile, $Stderr, [System.Text.Encoding]::UTF8)

        if (-not $Completed) {
            try { $Process.Kill() } catch {}
            $LastError = "worker_timeout=$TimeoutSeconds; attempt=$Attempt"
        } elseif ($Process.ExitCode -eq 0 -and (Test-Path $ResultFile)) {
            $ResultObject = [System.IO.File]::ReadAllText($ResultFile, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
            $ResultObject | Add-Member -NotePropertyName task_index -NotePropertyValue $Index -Force
            $ResultObject | Add-Member -NotePropertyName worker_attempt -NotePropertyValue $Attempt -Force
            Write-JsonLine $ResultsFile ($ResultObject | ConvertTo-Json -Depth 100 -Compress)
            $Success = $true
            break
        } else {
            $LastError = "worker_exit=$($Process.ExitCode); attempt=$Attempt; stdout_tail=$($Stdout.Substring([Math]::Max(0, $Stdout.Length - 1000))); stderr_tail=$($Stderr.Substring([Math]::Max(0, $Stderr.Length - 1000)))"
        }
        Start-Sleep -Seconds ([Math]::Min(10, 2 + $Attempt * 3))
    }

    if (-not $Success) {
        $TaskObject = $Lines[$Index] | ConvertFrom-Json
        $TaskObject | Add-Member -NotePropertyName task_index -NotePropertyValue $Index -Force
        $TaskObject | Add-Member -NotePropertyName worker_attempt -NotePropertyValue $MaxRetries -Force
        $TaskObject | Add-Member -NotePropertyName evaluation -NotePropertyValue ([pscustomobject]@{}) -Force
        $TaskObject | Add-Member -NotePropertyName error -NotePropertyValue $LastError -Force
        Write-JsonLine $ResultsFile ($TaskObject | ConvertTo-Json -Depth 100 -Compress)
    }

    Remove-Item -LiteralPath $TaskFile, $ResultFile, $StdoutFile, $StderrFile -Force -ErrorAction SilentlyContinue
    $Done = $Index + 1
    if (($Done % 10) -eq 0 -or $Done -eq $End) {
        $Errors = 0
        if (Test-Path $ResultsFile) {
            $Errors = Select-String -Path $ResultsFile -Pattern '"error":"(?!")' -AllMatches | Measure-Object | Select-Object -ExpandProperty Count
        }
        Write-Host "  JSONL worker progress $Done/$($Lines.Count) errors=$Errors"
    }
}
