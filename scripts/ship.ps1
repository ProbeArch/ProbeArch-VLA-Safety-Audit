# ship.ps1 - publish ProbeArch-VLA-Safety-Audit to GitHub
# Repo must exist (public) FIRST: https://github.com/new  ->  ProbeArch-VLA-Safety-Audit
# Then run:  powershell -ExecutionPolicy Bypass -File scripts/ship.ps1
#
# (Repo creation was attempted via MCP token but its scope lacks `repo` creation.
#  If you have a PAT/GH CLI with repo scope, `gh repo create ProbeArch-VLA-Safety-Audit --public --source . --push` also works.)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git remote remove origin 2>$null
git remote add origin https://github.com/dunli/ProbeArch-VLA-Safety-Audit.git
git push -u origin HEAD:main
git push origin v0.1 --force
Write-Host "Pushed. Verify: https://github.com/dunli/ProbeArch-VLA-Safety-Audit"