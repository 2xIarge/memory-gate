# Start a local OpenAI-compatible endpoint for memory-gate's real-model check.
#
#   pwsh -NoProfile -File scripts/start_local_server.ps1 -ModelPath <path\to\model.gguf> -ServerExe <path\to\llama-server>
#   pwsh -NoProfile -File scripts/start_local_server.ps1 -ModelPath <...> -ServerExe <...> -GpuLayers 20
#
# Then in another shell:
#   .\.venv\Scripts\python.exe scripts\verify_real_model.py
#
# In the foreground, stop the server with Ctrl+C in the window it runs in.

param(
    # No baked-in defaults on purpose: point these at your own GGUF and
    # llama-server build. The placeholders fail the Test-Path checks below with
    # a clear message rather than silently assuming anyone's disk layout.
    [string]$ModelPath = '<path\to\model.gguf>',
    [string]$ServerExe = '<path\to\llama-server>',
    [int]$Context = 8192,
    [int]$GpuLayers = 99,
    [int]$Port = 8080,
    [switch]$NewWindow
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ServerExe)) {
    throw "llama-server not found: $ServerExe"
}
if (-not (Test-Path $ModelPath)) {
    throw "model not found: $ModelPath"
}

$logDir = Join-Path (Split-Path -Parent $PSScriptRoot) '.scratch'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'llama-server.log'

$args = @('-m', $ModelPath, '-ngl', "$GpuLayers", '-c', "$Context",
          '--host', '127.0.0.1', '--port', "$Port", '--alias', 'local')

Write-Host "model : $ModelPath"
Write-Host "server: $ServerExe"
Write-Host "args  : $($args -join ' ')"
Write-Host "log   : $log"

if ($NewWindow) {
    Start-Process -FilePath $ServerExe -ArgumentList $args -WindowStyle Normal
    Write-Host "started in a new window"
}
else {
    # foreground: Ctrl+C stops it, and stderr is visible
    & $ServerExe @args 2>&1 | Tee-Object -FilePath $log
}
