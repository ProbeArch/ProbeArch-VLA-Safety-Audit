#!/usr/bin/env bash
# ship.sh - publish the validated ProbeArch VLA safety-audit branch to GitHub
# Repo: https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit
# Run:  scripts/ship.sh --i-am-sure
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_URL="https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit.git"
EXPECTED_BRANCH="fix/harness-v0.2"

if [[ "${1:-}" != "--i-am-sure" && "${SHIP:-0}" != "1" ]]; then
  echo "Refusing to ship without --i-am-sure or SHIP=1." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to ship from a dirty working tree." >&2
  exit 1
fi

if grep -q "RETRACTED" README.md docs/REPORT.md; then
  echo "Refusing to ship while audit results are marked RETRACTED." >&2
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "Refusing to ship branch '$BRANCH'; expected '$EXPECTED_BRANCH'." >&2
  exit 1
fi

EXISTING_ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
if [[ -n "$EXISTING_ORIGIN" && "$EXISTING_ORIGIN" != "$REMOTE_URL" ]]; then
  echo "Refusing to ship: origin already points to '$EXISTING_ORIGIN'," >&2
  echo "not '$REMOTE_URL'. Fix the remote manually if this repo's home is" >&2
  echo "$REMOTE_URL." >&2
  exit 1
fi
if [[ -z "$EXISTING_ORIGIN" ]]; then
  git remote add origin "$REMOTE_URL"
fi
# (no-op when origin already points at the expected URL)
git push -u origin "$BRANCH"
echo "Pushed $BRANCH. Verify: https://github.com/ProbeArch/ProbeArch-VLA-Safety-Audit"
