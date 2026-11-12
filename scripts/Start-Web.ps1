param(
    [string]$Workspace = (Join-Path $PSScriptRoot '..'),
    [ValidateRange(1, 65535)][int]$Port = 8793,
    [string]$WheelPath,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$kowwInstallRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$kowwWorkspaceRoot = (Resolve-Path -LiteralPath $Workspace).ProviderPath
. (Join-Path $PSScriptRoot 'Use-ProjectStorage.ps1') -ProjectRoot $kowwWorkspaceRoot

function Test-PythonCommand {
    param([string]$Executable, [string[]]$Arguments)
    $ErrorActionPreference = 'SilentlyContinue'
    & $Executable @Arguments *> $null
    return ($? -and $LASTEXITCODE -eq 0)
}

$kowwSupportedPython = 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14) and sys.maxsize > 2**32))'
$kowwReadyProbe = @'
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

expected = sys.argv[1]
try:
    installed = version('korean-openwakeword')
    ready = (expected == '*' or installed == expected)
    ready = ready and version('openwakeword') == '0.6.0'
    modules = ('korean_openwakeword', 'openwakeword', 'numpy', 'scipy',
               'soundfile', 'onnxruntime', 'requests', 'tqdm', 'sklearn')
    ready = ready and all(find_spec(name) is not None for name in modules)
except PackageNotFoundError:
    ready = False
raise SystemExit(not ready)
'@

function Find-BasePython {
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($launcher) {
        foreach ($selector in @('-3.12', '-3.13', '-3.11', '-3.10')) {
            if (Test-PythonCommand $launcher.Path @($selector, '-I', '-c', $kowwSupportedPython)) {
                return @{ Path = $launcher.Path; Arguments = @($selector) }
            }
        }
    }
    foreach ($candidate in (Get-Command python.exe, python3.exe -CommandType Application -All -ErrorAction SilentlyContinue)) {
        if ($candidate.Path -notlike '*\WindowsApps\*' -and
            (Test-PythonCommand $candidate.Path @('-I', '-c', $kowwSupportedPython))) {
            return @{ Path = $candidate.Path; Arguments = @() }
        }
    }
    throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ and run Start-Web.cmd again.'
}

# In a source checkout install the local project. A portable copy can use a
# wheel next to the root launcher, or one explicitly selected by -WheelPath.
$kowwInstallTarget = $null
$kowwExpectedVersion = '*'
$kowwProjectFile = Join-Path $kowwInstallRoot 'pyproject.toml'
$kowwWheelName = '^korean_openwakeword-(?<version>[^-]+)-py3-none-any\.whl$'
$kowwWheels = @()
if ($WheelPath) {
    $kowwInstallTarget = (Resolve-Path -LiteralPath $WheelPath).ProviderPath
    if (-not (Test-Path -LiteralPath $kowwInstallTarget -PathType Leaf) -or
        (Split-Path -Leaf $kowwInstallTarget) -notmatch $kowwWheelName) {
        throw 'WheelPath must point to a korean_openwakeword release wheel.'
    }
    $kowwExpectedVersion = $Matches.version
} elseif ((Test-Path -LiteralPath $kowwProjectFile -PathType Leaf) -and
          (Test-Path -LiteralPath (Join-Path $kowwInstallRoot 'korean_openwakeword\__init__.py') -PathType Leaf)) {
    if ((Get-Content -LiteralPath $kowwProjectFile -Raw) -notmatch '(?m)^\s*version\s*=\s*"(?<version>[^"]+)"') {
        throw 'Could not read the library version from pyproject.toml.'
    }
    $kowwInstallTarget = $kowwInstallRoot
    $kowwExpectedVersion = $Matches.version
} else {
    $kowwWheels = @(Get-ChildItem -LiteralPath $kowwInstallRoot -Filter 'korean_openwakeword-*-py3-none-any.whl' -File)
    if ($kowwWheels.Count -eq 1 -and $kowwWheels[0].Name -match $kowwWheelName) {
        $kowwInstallTarget = $kowwWheels[0].FullName
        $kowwExpectedVersion = $Matches.version
    }
}

$kowwVenv = Join-Path $kowwInstallRoot '.venv'
$kowwPythonPath = Join-Path $kowwVenv 'Scripts\python.exe'
$kowwVersionArgs = @('-I', '-c', $kowwSupportedPython)
$kowwReadyArgs = @('-I', '-c', $kowwReadyProbe, $kowwExpectedVersion)
New-Item -ItemType Directory -Path $kowwVenv -Force | Out-Null
try {
    $kowwSetupLock = [IO.File]::Open((Join-Path $kowwVenv '.start-web.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch [IO.IOException] {
    throw 'Could not lock the local setup folder. If another Start-Web window is installing, let it finish and try again.'
}
try {
    $kowwVenvReady = (Test-Path -LiteralPath $kowwPythonPath -PathType Leaf) -and
        (Test-PythonCommand $kowwPythonPath $kowwVersionArgs)
    if (-not $kowwVenvReady -or -not (Test-PythonCommand $kowwPythonPath $kowwReadyArgs)) {
        if (-not $kowwInstallTarget) {
            if ($kowwWheels.Count -gt 1) {
                throw 'More than one release wheel was found. Use -WheelPath to select the wheel to install.'
            }
            throw 'No library source or release wheel was found. Use the extracted source folder or put a release wheel in the folder that contains scripts.'
        }
        if (-not $kowwVenvReady) {
            $kowwBase = Find-BasePython
            Write-Host 'Preparing the local Python environment...'
            $kowwBaseArgs = $kowwBase.Arguments
            & $kowwBase.Path @kowwBaseArgs -I -m venv $kowwVenv
            if ($LASTEXITCODE -ne 0 -or -not (Test-PythonCommand $kowwPythonPath $kowwVersionArgs)) {
                throw 'Could not prepare .venv. See the error above and run Start-Web.cmd again.'
            }
        }
        if (-not (Test-PythonCommand $kowwPythonPath @('-I', '-m', 'pip', '--version'))) {
            & $kowwPythonPath -I -m ensurepip --upgrade
            if ($LASTEXITCODE -ne 0) { throw 'Could not install pip in .venv.' }
        }
        Write-Host 'Installing the library and web dependencies...'
        & $kowwPythonPath -I -m pip install --disable-pip-version-check $kowwInstallTarget
        if ($LASTEXITCODE -ne 0) { throw 'Library installation failed. See the error above and run Start-Web.cmd again.' }
        & $kowwPythonPath -I -m pip install --disable-pip-version-check --no-deps 'openwakeword==0.6.0'
        if ($LASTEXITCODE -ne 0 -or -not (Test-PythonCommand $kowwPythonPath $kowwReadyArgs)) {
            throw 'The web dependencies are not ready. See the error above and run Start-Web.cmd again.'
        }
    }
} finally {
    $kowwSetupLock.Dispose()
}

$kowwWebArgs = @('-I', '-m', 'korean_openwakeword', 'web', '--workspace', $kowwWorkspaceRoot, '--port', $Port)
if ($NoBrowser) { $kowwWebArgs += '--no-browser' }
& $kowwPythonPath @kowwWebArgs
exit $LASTEXITCODE
