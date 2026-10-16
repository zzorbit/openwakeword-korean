param([string]$Workspace = (Join-Path $PSScriptRoot '..'), [int]$Port = 8793)
$ErrorActionPreference = 'Stop'
$kowwSourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$kowwWorkspaceRoot = (Resolve-Path -LiteralPath $Workspace).ProviderPath
. (Join-Path $PSScriptRoot 'Use-ProjectStorage.ps1') -ProjectRoot $kowwWorkspaceRoot
$kowwPythonPath = Join-Path $kowwSourceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $kowwPythonPath)) {
    throw 'No .venv found. Complete the README installation once, then run Start-Web again.'
}
& $kowwPythonPath -m korean_openwakeword web --workspace $kowwWorkspaceRoot --port $Port
exit $LASTEXITCODE
