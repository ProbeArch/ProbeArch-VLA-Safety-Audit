# Independent labels

`labels_template.csv` is a blank, versioned input template. Human labels are
deliberately not fabricated by the repository. Follow
[`docs/ANNOTATION_GUIDE.md`](../docs/ANNOTATION_GUIDE.md), keep one row per
annotator per episode, and store completed label files outside the raw rollout
tree until they are reviewed.

The blinded sampling manifest is generated reproducibly from local frozen
rollouts with:

```bash
python scripts/analysis/prepare_annotation_sample.py \
  configs/annotation_sample.json \
  --output annotations/sample_manifest.csv
```

It allocates 25 episodes to each of the four recorded-success/task-aware
strata, oversamples destination-related borderline cases, and marks a held-out
split. The generated manifest contains no ProbeArch candidate label; the
sampling summary is for the study owner only.

After labels are adjudicated, run `label_agreement.py` with the independent
labels and `adjudicated_template.csv`. A second `episode_path,label` CSV with
ProbeArch’s candidate statuses can be supplied through `--candidate` to
compute unsafe-event precision, recall, false-positive rate, and coverage.
