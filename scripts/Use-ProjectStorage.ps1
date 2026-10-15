param([string]$ProjectRoot = (Join-Path $PSScriptRoot '..'))
$ErrorActionPreference = 'Stop'
$kowwProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).ProviderPath
$kowwCacheRoot = Join-Path $kowwProjectRoot 'storage\runtime_cache'
if ($kowwCacheRoot -match '(^|[\\/])\.codex([\\/]|$)') { throw '프로젝트 캐시는 .codex 안에 둘 수 없습니다.' }
$kowwCachePaths = @{
    KOWW_CACHE_ROOT = $kowwCacheRoot
    HF_HOME = (Join-Path $kowwCacheRoot 'huggingface')
    HF_HUB_CACHE = (Join-Path $kowwCacheRoot 'huggingface\hub')
    HUGGINGFACE_HUB_CACHE = (Join-Path $kowwCacheRoot 'huggingface\hub')
    HF_XET_CACHE = (Join-Path $kowwCacheRoot 'huggingface\xet')
    TORCH_HOME = (Join-Path $kowwCacheRoot 'torch')
    XDG_CACHE_HOME = (Join-Path $kowwCacheRoot 'xdg')
    TORCHINDUCTOR_CACHE_DIR = (Join-Path $kowwCacheRoot 'torchinductor')
    TRITON_CACHE_DIR = (Join-Path $kowwCacheRoot 'triton')
    CUDA_CACHE_PATH = (Join-Path $kowwCacheRoot 'cuda')
    TORCH_EXTENSIONS_DIR = (Join-Path $kowwCacheRoot 'torch_extensions')
    NUMBA_CACHE_DIR = (Join-Path $kowwCacheRoot 'numba')
    PIP_CACHE_DIR = (Join-Path $kowwCacheRoot 'pip')
    TMP = (Join-Path $kowwCacheRoot 'tmp')
    TEMP = (Join-Path $kowwCacheRoot 'tmp')
    TMPDIR = (Join-Path $kowwCacheRoot 'tmp')
}
foreach ($kowwCacheEntry in $kowwCachePaths.GetEnumerator()) {
    New-Item -ItemType Directory -Path $kowwCacheEntry.Value -Force | Out-Null
    [Environment]::SetEnvironmentVariable($kowwCacheEntry.Key, $kowwCacheEntry.Value, 'Process')
}
Write-Output "프로젝트 캐시: $kowwCacheRoot (현재 PowerShell과 자식 프로세스에만 적용)"
