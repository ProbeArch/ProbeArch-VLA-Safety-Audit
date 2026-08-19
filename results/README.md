# results/ — v0.1 audit artifacts (RETRACTED)

**DO NOT CITE. DO NOT use as current results. Forensics only.**

These are the outputs of the **v0.1 audit run** (n=160, 2026-08-12), whose headline
numbers — **0/160 success** and **"0 external safety events"** — were shown to be
**instrumentation artifacts**, not policy findings (see `docs/HANDOFF.md` and
`docs/amendments.md` for the defect catalog C1–C6 and the post-handoff fix campaign).

Why the numbers are invalid:
- success was read once after the whole rollout (gymnasium auto-reset had already
  cleared the terminal `info`) → every success silently False;
- τ1 was derived from arm-on-table reset contacts (~893 N solver saturation) → R1
  structurally incapable of firing;
- the "drop" positive control was a no-op (`bodies.get(None)`);
- R4 compared z against a hardcoded `TABLE_Z = 0.0` (real table ≈ 0.9 m) → never fired;
- R3 fires here include an init-state artifact (bowl spawned pre-tilted, t=0).

Contents (all from the invalid v0.1 run; thresholds tau1=1786.9 N / tau2=0.2856 m are stale):
- `calibration.json` — v0.1 calibration (dead thresholds; drop trials were no-ops)
- `events_forensics.json` — event forensics used for amendment A6 (R5 reclassification)
- `safety_summary.json`, `stats.json` — v0.1 aggregate tables
- `figures/` — v0.1 plots (note: `eef_z.png` is superseded by the corrected
  `object_fall.png` view in the v0.2 pipeline; kept here for forensics only)

Current outputs land in `$AUDIT_DIR` (default `~/audit`) — see `docs/HANDOFF.md`
for the corrected pipeline and the mandatory validation run.
