# Register current-user Windows Task Scheduler tasks that start the
# local relay and trading assistant supervisor at logon.
#
# Run as: powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$StartScript = Join-Path $ProjectRoot "scripts\start.ps1"
$RelayStartScript = Join-Path $ProjectRoot "scripts\start-relay.ps1"
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function Register-TradingAssistantStartupTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName,
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path $ScriptPath)) {
        Write-Error "$ScriptPath not found"
        exit 1
    }

    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed existing task '$TaskName'"
    }

    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
        -WorkingDirectory $ProjectRoot

    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $Trigger `
        -Action $Action `
        -Settings $Settings `
        -Description $Description `
        -RunLevel Limited | Out-Null

    Write-Host "Task '$TaskName' registered for $CurrentUser."
}

Register-TradingAssistantStartupTask `
    -TaskName "TradingAssistantRelayAutoStart" `
    -ScriptPath $RelayStartScript `
    -Description "Start the local trading relay in the background at logon"

Register-TradingAssistantStartupTask `
    -TaskName "TradingAssistantAutoStart" `
    -ScriptPath $StartScript `
    -Description "Start the trading assistant supervisor in the background at logon"

Write-Host "The local relay and trading assistant supervisor will start in the background at logon."
Write-Host ""
Write-Host "To remove relay: Unregister-ScheduledTask -TaskName 'TradingAssistantRelayAutoStart'"
Write-Host "To remove assistant: Unregister-ScheduledTask -TaskName 'TradingAssistantAutoStart'"
