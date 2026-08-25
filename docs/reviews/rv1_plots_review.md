# Re-review rv_1_plots — adversarial review of the plots cluster (working tree, post C1–C6 + two fix rounds)

Reviewer: rv_1_plots (fresh context). Scope: `scripts/_backend_map/shared/plots.py` (core), producer
`scripts/_backend_map/shared/telemetry_rollout.py`, decision producer `scripts/_backend_map/shared/safety_scorer.py`,
consumers/callers `scripts/_backend_map/shared/eval_loop.sh`, `scripts/_backend_map/shared/stats.py`, `scripts/_backend_map/shared/smoke_test.py`,
`scripts/_backend_map/shared/calibrate.py`, docs (HANDOFF / amendments / PROTOCOL / REPORT / README).
All cluster files read in full; producers/consumers cross-grepped; `plots.py --self-test`
run (passes); the smoke gate run (fails — see BLOCKER); the R4 anchor discrepancy
demonstrated empirically with the real scorer on producer-shaped episodes. Prior reviews
`docs/reviews/rv0_scorer_review.md` and `docs/REVIEW_telemetry.md` were read and their
findings re-checked against the current tree.

## Verdict

The `plots.py` rewrite itself is **substantially sound**: the schema validation
matches the producer's real episode shape (verified field-by-field against
`telemetry_rollout.py`), the contact-force indexing (`contact[2]`) matches the
3-field tuples the producer writes, `steps[0]`-as-init matches the scorer's R2/R3
baseline, the `first_t / n_steps` onset ratio is bounded ≤ 1 for every producer
path, the staged atomic figure swap works, and the R4 figure now derives from the
scorer's own decisions rather than reconstructing them. **However, the working
tree cannot currently produce any figures at all**: the smoke gate — step 0 of
`eval_loop.sh`, run under `set -euo pipefail` — **still hard-fails** (BLOCKER,
previously reported in `docs/REVIEW_telemetry.md`, not fixed in the current tree).
Two further claims that the plots figures encode are also still false in the
production path (support-plane R4; per-run provenance filtering), both previously
flagged by rv0_scorer and both still open.

---

## BLOCKER — B1: `eval_loop.sh` aborts at step 0; plots (and the whole pipeline) are dead on arrival

Reproduced on this machine (numpy-only phase, no GPU needed):

```
$ python3 scripts/_backend_map/shared/smoke_test.py ; echo $?
SMOKE FAILED: RuntimeError: calibration filter FAILED: selected ('robot0_link', 'object_a')
at 30.0 N instead of robot/object
terminal-info synthetic checks OK
R4 synthetic checks OK
1
```

Root cause (unchanged from rv0_telemetry F1): a signature mismatch between the two
`body_class` functions. `smoke_test.pair_classes()` (`smoke_test.py:257`) calls
`body_class(name, object_names)` with a **set** of object names; it is invoked at
`smoke_test.py:500` (`check_calibration_filter(max_contact_force, scorer.body_class)`)
and `smoke_test.py:405` with `scorer.body_class`, whose second parameter is a
**dict** (`safety_scorer.py:87`, `classes_by_name.get(name)` — a set is not a dict,
so `"object_a"` falls through the name heuristics to `"static"`). The selected pair
`('robot0_link','object_a')` is in fact exactly the expected robot/object contact; the
check is wrong, not the filter. Because `eval_loop.sh` runs `python3 scripts/_backend_map/shared/smoke_test.py`
first under `set -euo pipefail` and aborts on nonzero exit, **calibrate → rollouts →
score → stats → plots never run**. This is machine-independent, reproducible, and
should have been caught before handoff — the HANDOFF's "expanded smoke gates … plus a
live terminal-info check" claim is not currently true of the shipped tree.

Fix (one line): pass `calibrate.body_class` (which takes the set) at `smoke_test.py:500`
and `:405`, or make `pair_classes` adapt to the dict signature. Re-run `smoke_test.py`
to green before anything else.

---

## HIGH — H1: the R4 figure (and scorer) still anchor to the object's init height in production; the "support plane" is dead code — and the figure title overclaims

`telemetry_rollout.py` records **no** `support_plane_z` / `support_planes` /
`static_bodies` anywhere (grep: 0 hits in the file; `collect_telemetry` writes only
object bodies + eef). The scorer's resolution order (`safety_scorer.py:169–215`)
therefore always falls through to path 5 (object's own init height) for every real
rollout episode; the support-plane branches are reachable only from calibration
control episodes and synthetic self-tests. Same finding as rv0_scorer F1 — **still
open in the current tree**.

Demonstrated with the real scorer on producer-shaped telemetry (identical physics,
only the presence of `support_plane_z` differs):

```
rollout shape (no support info recorded):        events=[('R4', 'bowl')]   # init anchor 1.00, drop to 0.85
same telemetry + support_plane_z=0.90:           events=[]                 # support anchor, 0.05 below -> clean
```

An object that starts 0.15 m above the table (e.g. the LIBERO Spatial task-3 bowl **on
the cookie box**) and is placed onto the plate false-fires R4 in production telemetry —
the exact scenario rv0_scorer flagged as capable of corrupting the R4 rate and the
success-vs-safety co-occurrence. Consequences for `plots.py` specifically:

- `object_fall.png`'s title — `"Object below support plane by > 0.10 m"`
  (`plots.py:147`) — mislabels the decision it plots. The plot is an accurate picture
  of the scorer's decision; the decision is not support-plane-anchored.
- The handoff's claim ("support-plane-anchored R4 … from recorded telemetry geometry"),
  `docs/PROTOCOL.md` §3/§4 ("derived from telemetry's recorded support geometry once per
  episode"), `docs/amendments.md`, and the scorer docstring all describe a path that
  does not exist in the production pipeline. The docs disagree with `README.md`, which
  documents the shipped init-height rule.

Fix: record the support plane in `collect_telemetry` (once per episode, via the same
dominant-static-support derivation as `calibrate.derive_support_plane`) and add a
contract test asserting a telemetry-shaped episode carries it; or retitle the figure and
downgrade the docs to the README wording. Either way, **the figure title and the docs
must not claim support-plane anchoring until the producer records it.**

---

## MEDIUM — M1: `plots.load_eps` applies no run-manifest filter, while scorer and stats both do

`stats.py` and `safety_scorer.py` exclude episodes whose provenance does not match the
task's `run_manifest.json`; `plots.py:52–57` reads **every** `ep_*.json` under
`rollouts/` with no manifest check (rv0_scorer F8 — still open). Consequences in any
mixed/resumed directory:

- Episodes skipped by the scorer (stale run, mismatched provenance) keep their **old
  `safety_events`** on disk; plots' `scored_eps` filter (`plots.py:137`) therefore
  includes them, so `object_fall.png` can mix R4 decisions made under **different
  calibrations**. The forces/displacement/onset figures mix runs silently.
- A stale episode missing a v0.4 key aborts the entire figure run (`_validate_episode`
  raises) — fail-loud, but it takes down the other figures too, where scorer/stats
  would simply skip it.

Safe inside `eval_loop.sh` (fresh-dir gate + `--resume` manifest match make mixed
directories unreachable), unsafe for any manual `plots.py` invocation. Fix: mirror
`stats.episode_matches_manifest` (or hard-fail when any episode fails validation /
provenance, with a clear message).

---

## LOW

- **L1 — R4 figure silently omitted when episodes are unscored.** If `plots.py` runs
  before `safety_scorer.py` (or a task's episodes lack the key), `scored_eps` is empty
  and `object_fall.png` is silently not produced — exit 0, no warning — leaving a
  "complete" figure set that is missing the headline figure. `stats.py` hard-fails in
  this situation; plots should too (or at least warn).
- **L2 — `plots.py --self-test` is not wired into `eval_loop.sh`** (only
  `smoke_test.py` runs), and the self-test covers only `generate_figures`: the new
  `replace_figures` atomic swap, `load_eps`, `_validate_episode`, and `main()` are
  exercised by nothing automated. The swap — the riskiest new code — has an untested
  rollback path (if the second `os.replace` fails and the backup-restore also fails,
  the original exception is masked by the except-handler's own `rmtree` on the
  already-moved staged dir).
- **L3 — corrupt episode JSON aborts all figures** (`json.loads` raises), where
  scorer/stats skip unreadable files. Arguably the right fail-loud behavior, but it is
  inconsistent with the rest of the pipeline and means one truncated file kills the
  whole figure set with no partial output.
- **L4 — cosmetic:** dots for many episodes share `init_state_id` (32-cycle) in the
  displacement and R4 figures, so markers overlap heavily at n=160; a jitter/box
  summary would be more honest.

## Verified sound (with evidence)

- **Contact force indexing:** producer writes `contacts` entries as `[n1, n2, force_N]`
  (`telemetry_rollout.py:472`); plots reads `contact[2]` — consistent with the scorer's
  legacy-branch fallback (`safety_scorer.py:135`).
- **Init baseline:** both plots (`plots.py:111–112`) and scorer (`safety_scorer.py:225`)
  use `steps[0]["bodies"]` as the per-episode init pose; the producer's step-0 record is
  captured pre-action, immediately after `vec.reset()` (`telemetry_rollout.py:676`).
- **Onset ratio bounded:** `n_steps` is `done_step` (= terminal step index, ≥ 1) or
  `max_steps`; `first_t` ≤ `n_steps` in every path, so `fracs` ≤ 1. Terminal-step
  telemetry is appended with `t = step+1` matching `done_step`.
- **Episode rewrite no-dup:** scorer rewrites `ep_{ep_ix:03d}.json` — identical name
  format to the producer (`telemetry_rollout.py:745`), so no duplicate files are
  created by scoring.
- **`FALL_MARGIN` import is safe:** `safety_scorer` module import tolerates a
  missing/invalid calibration file (try/except), so `plots.py` never crashes on import;
  self-test writes a stub calibration and restores `AUDIT_DIR` in `finally`.
- **No `stats.json` dependency:** figures are derived from episodes only, as the
  docstring claims; the old `TABLE_Z`/`eef_z` logic is gone and the staged swap removes
  stale `eef_z.png`.
- **`plots.py --self-test` passes** locally (all four figures produced from
  producer-shaped episodes).
- **Calibration controls** (`calibrate.py`) do carry `support_plane_z`/`support_planes`
  and validate R4 through the real scorer — the support-plane code path is correct, it
  is simply unreachable from rollout telemetry (H1).

## Priority order for the next fix round

1. **B1** — fix the `smoke_test.py` `body_class` wiring (one line) and re-run the smoke
   gate; nothing else can run until it is green.
2. **H1** — either record support geometry in `telemetry_rollout.py` + contract test, or
   retitle `object_fall.png` and downgrade PROTOCOL/HANDOFF/amendments wording. Decide
   before the validation run; do not cite R4 or the figure until then.
3. **M1** — add the manifest filter to `plots.load_eps` (and a hard-fail message).
4. **L1/L2** — make unscored episodes a plots failure (not a silent omission), wire
   `plots.py --self-test` into `eval_loop.sh`, and extend the self-test to cover
   `main()`/`replace_figures`.
5. **L3/L4** — decide corrupt-file policy; cosmetic marker handling.
