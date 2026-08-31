# Versioned schemas

The schemas describe the interchange boundary between a policy/environment
adapter and the audit core:

- `telemetry.schema.json` — raw episode observations and provenance.
- `task_spec.schema.json` — task targets, destinations, and distractors.
- `measurement.schema.json` — task-aware outcome and evidence quality.
- `rule.schema.json` — an individual detector’s candidate evidence.

Every task-aware measurement and detector result carries both a semantics
version and a measurement-contract version. `hazard_assessment` is explicitly
`not_assessed` until operational limits and independent labels justify a
separate claim. Schema validation belongs in CI and in downstream adapters;
schema changes require a version bump or a documented migration.
