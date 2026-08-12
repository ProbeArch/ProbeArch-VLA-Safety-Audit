# ship.ps1 - publish ProbeArch-VLA-Safety-Audit to GitHub (org remote)
# Repo: https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit (already exists)
# Run:  powershell -ExecutionPolicy Bypass -File scripts/ship.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git remote remove origin 2>$null
git remote add origin https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit.git
git push -u origin HEAD:main
git push origin v0.1 --force
Write-Host "Pushed. Verify: https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit"