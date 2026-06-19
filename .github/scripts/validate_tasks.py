#!/usr/bin/env python3
"""Validate every curated task line against ``schemas/task.schema.json``.

Used by ``.github/workflows/schema-validate.yml``. Self-contained: depends only
on ``jsonschema`` so it runs without installing the package or its extras. Each
non-empty line of every ``tasks/curated/*.jsonl`` file must be one valid Task
object. Exits non-zero (and prints the first errors per file) on any violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "task.schema.json"
CURATED_GLOB = "tasks/curated/*.jsonl"
MAX_ERRORS_PER_FILE = 10


def main() -> int:
    """Validate all curated task files; return process exit code."""
    if not SCHEMA_PATH.is_file():
        print(f"::error::schema not found: {SCHEMA_PATH}", file=sys.stderr)
        return 2

    schema_doc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema_doc)
    validator = jsonschema.Draft202012Validator(schema_doc)

    files = sorted(REPO_ROOT.glob(CURATED_GLOB))
    if not files:
        # No curated tasks yet is not a failure — the gate just has nothing to do.
        print(f"no files matched {CURATED_GLOB}; nothing to validate.")
        return 0

    total_lines = 0
    total_invalid = 0

    for path in files:
        rel = path.relative_to(REPO_ROOT)
        file_invalid = 0
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    file_invalid += 1
                    print(f"::error file={rel},line={lineno}::invalid JSON: {exc}")
                    continue
                errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
                if errors:
                    file_invalid += 1
                    for err in errors[:MAX_ERRORS_PER_FILE]:
                        loc = "/".join(str(p) for p in err.path) or "<root>"
                        print(f"::error file={rel},line={lineno}::{loc}: {err.message}")
        status = "OK" if file_invalid == 0 else f"{file_invalid} INVALID"
        print(f"{rel}: {status}")
        total_invalid += file_invalid

    print(f"\nvalidated {total_lines} task line(s) across {len(files)} file(s); "
          f"{total_invalid} invalid.")
    return 1 if total_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
