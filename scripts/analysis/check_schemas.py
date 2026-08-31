#!/usr/bin/env python3
"""Run dependency-free structural checks over versioned JSON schemas."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"


def check_schema(path: Path) -> list[str]:
    errors = []
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: {exc}"]
    for key in ("$schema", "$id", "title", "type", "properties", "required"):
        if key not in schema:
            errors.append(f"{path.name}: missing top-level {key}")
    if not str(schema.get("$id", "")).startswith("probearch://schemas/"):
        errors.append(f"{path.name}: $id is not a ProbeArch schema id")
    properties = schema.get("properties") or {}
    for required in schema.get("required") or []:
        if required not in properties:
            errors.append(f"{path.name}: required field {required!r} has no property definition")
    return errors


def main() -> int:
    paths = sorted(SCHEMAS.glob("*.schema.json"))
    errors = [error for path in paths for error in check_schema(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"schemas OK: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
