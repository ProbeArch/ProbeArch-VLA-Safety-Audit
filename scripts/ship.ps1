# one-click ship: create public repo + push v0.1 (run from repo root)
# Step 1 (USER, one click): https://github.com/new -> repo name "ProbeArch-VLA-Safety-Audit", Public
# Step 2 (then run this):
git remote add origin https://github.com/Ehdunhackme/ProbeArch-VLA-Safety-Audit.git
git push -u origin main
git push origin v0.1
# Alternative: if gh CLI available:
# gh repo create Ehdunhackme/ProbeArch-VLA-Safety-Audit --public --source . --push