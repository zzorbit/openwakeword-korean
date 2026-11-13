param([string]$Workspace = (Join-Path $PSScriptRoot '..'))
$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).ProviderPath
$studioRoot = Join-Path $workspaceRoot '.koww-studio'
$lockPath = Join-Path $studioRoot 'server.lock'
if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    Write-Host 'The web server is already stopped.'
    exit 0
}

# Confirm this workspace is actually owned. A copied or stale lease alone
# must never be enough to stop a process.
$serverLock = [IO.File]::Open($lockPath, 'Open', 'ReadWrite', 'ReadWrite')
try {
    try {
        $serverLock.Lock(0, 1)
        $serverLock.Unlock(0, 1)
        Write-Host 'The web server is already stopped.'
        exit 0
    } catch [IO.IOException] {
        if (($_.Exception.HResult -band 0xffff) -ne 33) { throw }
    }
} finally {
    $serverLock.Dispose()
}

$leasePath = Join-Path $studioRoot 'server.json'
if (-not (Test-Path -LiteralPath $leasePath -PathType Leaf)) {
    throw 'The web server is still starting. Try again in a moment.'
}
$lease = Get-Content -LiteralPath $leasePath -Raw | ConvertFrom-Json
if ($lease.pid -isnot [int] -or $lease.pid -le 0 -or
    $lease.creation_identity -isnot [string] -or $lease.creation_identity -notmatch '^\d+$') {
    throw 'The web server record is invalid. No process was stopped.'
}

$serverProcess = Get-Process -Id $lease.pid -ErrorAction SilentlyContinue
if ($null -eq $serverProcess) {
    Write-Host 'The web server is already stopped.'
    exit 0
}
try {
    # Keep the process handle open across verification and termination.
    # A recycled PID must never target a different process.
    $null = $serverProcess.Handle
    $started = $serverProcess.StartTime.ToFileTimeUtc().ToString([Globalization.CultureInfo]::InvariantCulture)
    if ($started -ne $lease.creation_identity) {
        throw 'The process does not match the saved web server record. No process was stopped.'
    }
    try {
        $serverProcess.Kill()
    } catch [InvalidOperationException] {
        if (-not $serverProcess.HasExited) { throw }
    }
    if (-not $serverProcess.WaitForExit(10000)) {
        throw 'The web server has not exited. Wait a moment before restarting it.'
    }
    Write-Host 'The web server has stopped.'
} finally {
    $serverProcess.Dispose()
}
