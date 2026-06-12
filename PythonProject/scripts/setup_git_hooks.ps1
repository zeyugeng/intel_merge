# One-time setup: use project githooks/ for this repo.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

git config core.hooksPath githooks
Write-Host "Git hooks enabled: $root\githooks"
Write-Host "Future commits will auto-remove Cursor co-author lines."
