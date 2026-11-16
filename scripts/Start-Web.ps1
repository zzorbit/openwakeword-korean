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
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec

expected, identity_arg = sys.argv[1:]
identity = identity_arg.removeprefix('artifact=')
try:
    installed = version('korean-openwakeword')
    ready = (expected == '*' or installed == expected)
    ready = ready and version('openwakeword') == '0.6.0'
    modules = ('korean_openwakeword', 'openwakeword', 'numpy', 'scipy',
               'soundfile', 'onnxruntime', 'requests', 'tqdm', 'sklearn')
    ready = ready and all(find_spec(name) is not None for name in modules)
    if identity.startswith('wheel:'):
        direct = json.loads(distribution('korean-openwakeword').read_text('direct_url.json') or '{}')
        archive = direct.get('archive_info', {})
        sha = archive.get('hashes', {}).get('sha256')
        if sha is None and archive.get('hash', '').startswith('sha256='):
            sha = archive['hash'][7:]
        ready = ready and (sha or '').upper() == identity[6:]
except (PackageNotFoundError, ValueError, TypeError, AttributeError):
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

function Get-InstallFileHash {
    param([string]$Path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::OpenRead($Path)
        try {
            return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '')
        } finally {
            $stream.Dispose()
        }
    } finally {
        $hasher.Dispose()
    }
}

function Get-InstallIdentity {
    param([string]$Target)
    if (-not $Target) { return '' }
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        return 'wheel:' + (Get-InstallFileHash $Target)
    }
    $files = @(
        Get-ChildItem -LiteralPath $Target -File | Where-Object {
            $_.Extension -eq '.py' -or $_.Name -in @('pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md')
        }
        Get-ChildItem -LiteralPath (Join-Path $Target 'korean_openwakeword') -File -Recurse | Where-Object {
            $_.Extension -in @('.py', '.json', '.html', '.js', '.css', '.svg', '.txt', '.md', '.onnx', '.wav')
        }
    )
    # Use the same order in Windows PowerShell and PowerShell 7.
    $paths = [string[]]@($files.FullName)
    [Array]::Sort($paths, [StringComparer]::Ordinal)
    $entries = $paths | ForEach-Object {
        $relative = $_.Substring($Target.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/')
        (Get-InstallFileHash $_) + ' ' + $relative
    }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes(($entries -join "`n"))
        return 'source:' + [BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-', '')
    } finally {
        $hasher.Dispose()
    }
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
New-Item -ItemType Directory -Path $kowwVenv -Force | Out-Null
try {
    $kowwSetupLock = [IO.File]::Open((Join-Path $kowwVenv '.start-web.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch [IO.IOException] {
    throw 'Could not lock the local setup folder. If another Start-Web window is installing, let it finish and try again.'
}
try {
    $kowwVenvReady = (Test-Path -LiteralPath $kowwPythonPath -PathType Leaf) -and
        (Test-PythonCommand $kowwPythonPath $kowwVersionArgs)
    $kowwIdentity = Get-InstallIdentity $kowwInstallTarget
    $kowwReadyArgs = @('-I', '-c', $kowwReadyProbe, $kowwExpectedVersion, "artifact=$kowwIdentity")
    $kowwReady = $kowwVenvReady -and (Test-PythonCommand $kowwPythonPath $kowwReadyArgs)
    $kowwIdentityPath = Join-Path $kowwVenv '.koww-install-identity'
    $kowwMatches = -not $kowwIdentity -or ((Test-Path -LiteralPath $kowwIdentityPath -PathType Leaf) -and
        [IO.File]::ReadAllText($kowwIdentityPath) -ceq $kowwIdentity)
    if (-not $kowwReady -or -not $kowwMatches) {
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
        $kowwHasSameVersion = Test-PythonCommand $kowwPythonPath @('-I', '-c',
            "import sys; from importlib.metadata import version; raise SystemExit(version('korean-openwakeword') != sys.argv[1])",
            $kowwExpectedVersion)
        # Invalidate the old receipt before attempting any package changes.
        if (Test-Path -LiteralPath $kowwIdentityPath -PathType Leaf) {
            Remove-Item -LiteralPath $kowwIdentityPath
        }
        Write-Host 'Installing the library and web dependencies...'
        & $kowwPythonPath -I -m pip install --disable-pip-version-check $kowwInstallTarget
        if ($LASTEXITCODE -ne 0) { throw 'Library installation failed. See the error above and run Start-Web.cmd again.' }
        if ($kowwHasSameVersion) {
            # pip can keep the installed package when only its contents changed.
            & $kowwPythonPath -I -m pip install --disable-pip-version-check --force-reinstall --no-deps $kowwInstallTarget
            if ($LASTEXITCODE -ne 0) { throw 'Library replacement failed. See the error above and run Start-Web.cmd again.' }
        }
        & $kowwPythonPath -I -m pip install --disable-pip-version-check --no-deps 'openwakeword==0.6.0'
        if ($LASTEXITCODE -ne 0 -or -not (Test-PythonCommand $kowwPythonPath $kowwReadyArgs)) {
            throw 'The web dependencies are not ready. See the error above and run Start-Web.cmd again.'
        }
        if ((Get-InstallIdentity $kowwInstallTarget) -cne $kowwIdentity) {
            throw 'The library source changed during setup. Run Start-Web.cmd again after the update finishes.'
        }
        [IO.File]::WriteAllText($kowwIdentityPath, $kowwIdentity)
    }
} finally {
    $kowwSetupLock.Dispose()
}

$kowwWebArgs = @('-I', '-m', 'korean_openwakeword', 'web', '--workspace', $kowwWorkspaceRoot, '--port', $Port)
if ($NoBrowser) { $kowwWebArgs += '--no-browser' }
Write-Host 'Close this window or press Ctrl+C to stop the web server.'
# Run in the launcher's console so Windows delivers its close event to Python.
& $kowwPythonPath @kowwWebArgs
exit $LASTEXITCODE
