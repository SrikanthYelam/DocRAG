# Downloads the official sqlite-vec loadable extension for Windows into native/win-x64/.
# (The Dockerfile fetches the Linux build itself.)
$ErrorActionPreference = 'Stop'
$version = '0.1.9'
$dest = Join-Path $PSScriptRoot '..\native\win-x64'
New-Item -ItemType Directory -Force $dest | Out-Null
$archive = Join-Path $env:TEMP "sqlite-vec-$version.tar.gz"
Invoke-WebRequest "https://github.com/asg017/sqlite-vec/releases/download/v$version/sqlite-vec-$version-loadable-windows-x86_64.tar.gz" -OutFile $archive
tar -xzf $archive -C $dest
Write-Host "sqlite-vec $version extracted to $dest"
