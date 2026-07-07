#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Creates a Windows Task Scheduler task that runs refresh_data.py every 2 days.
.DESCRIPTION
    If the PC is off at the scheduled time, it runs as soon as the PC comes online
    (StartWhenAvailable). Run this script once as Administrator:
        powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\sehyu\AppData\Local\Programs\Python\Python312\python.exe"
$RefreshScript = Join-Path $ScriptDir "refresh_data.py"
$WorkingDir = Split-Path -Parent $ScriptDir
$TaskName = "CryptoTraderDataRefresh"

# Validate paths
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python not found at: $PythonExe"
    exit 1
}
if (-not (Test-Path $RefreshScript)) {
    Write-Error "refresh_data.py not found at: $RefreshScript"
    exit 1
}

Write-Host "Creating scheduled task: $TaskName"
Write-Host "  Python:  $PythonExe"
Write-Host "  Script:  $RefreshScript"
Write-Host "  WorkDir: $WorkingDir"
Write-Host "  Schedule: Every 2 days at 12:00, with missed-run catch-up"
Write-Host ""

# Build task components
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$RefreshScript`"" `
    -WorkingDirectory $WorkingDir

$Trigger = New-ScheduledTaskTrigger -Daily -DaysInterval 2 -At "12:00"

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

# Register (overwrites if exists)
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Refreshes crypto market data every 2 days" `
        -Force | Out-Null

    Write-Host "Task created successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Verify: Get-ScheduledTaskInfo -TaskName '$TaskName'"
    Write-Host "  Run now: Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  Delete:  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Error "Failed to create task: $_"
    Read-Host "Press Enter to close"
    exit 1
}

Read-Host "Press Enter to close"
