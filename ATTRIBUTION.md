# Attribution policy

`usability-benchmark` is built around *real* open-source repositories, but it
**does not vendor, fork, or redistribute any third-party source code**. Reference
repositories are used only as the seed for task design and as a provenance signal
for contamination labelling. This document states the policy precisely so that
every task in the benchmark is auditable and license-clean.

## What we reference, and how

Each task may cite one or more `reference_repos` in its frozen task gold
(`schemas/task.schema.json` → `reference_repos[]`). For every referenced repo we
record, and require:

| Field     | Meaning                                                              |
| --------- | ------------------------------------------------------------------- |
| `url`     | Canonical repository URL.                                            |
| `commit`  | A **pinned full SHA** (≥ 7 chars, no floating refs like `main`/tags).|
| `license` | The repository's declared license as an **SPDX identifier**.        |
| `role`    | Why it is referenced (e.g. `inspiration`, `api-shape`, `fixture`).  |
| `why`     | One-line human rationale.                                            |

The pinned SHA makes every reference reproducible and lets us label a model's
training-data contamination risk against a fixed snapshot in time, rather than a
moving branch head.

## What we DO NOT do

- **No vendored code.** No file from a referenced repository is copied into this
  repository. There is no `third_party/`, no embedded submodule of task sources,
  and no copied snippets in task fixtures.
- **No redistribution.** We distribute task *specifications* (goals, hidden
  acceptance criteria, ambiguity gold) authored for this benchmark — not the
  upstream source.
- **No license laundering.** A repository is only eligible to be referenced if it
  carries an OSI-approved, SPDX-identifiable license. Repos with no license, or a
  non-redistributable / source-available-only license, may still be *referenced*
  by URL+SHA but never have any content reproduced.

## SPDX requirement

The `license` field MUST be a valid SPDX identifier (e.g. `MIT`,
`Apache-2.0`, `BSD-3-Clause`, `GPL-3.0-only`). This is enforced at task QC time.
If a repository's license cannot be resolved to an SPDX id, the task is rejected
during curation rather than shipped with an ambiguous attribution.

## This repository's own license

The benchmark code, schemas, configs, and authored task specifications in this
repository are released under the MIT License — see [`LICENSE`](./LICENSE).
That license covers *our* work only; it makes no claim over the referenced
upstream projects, which remain under their own respective licenses.

## How to audit attributions

```bash
# List every (repo, commit, license) reference across curated tasks.
python -m collect.cli validate tasks/curated/*.jsonl --schema schemas/task.schema.json
```

Curated task files live in `tasks/curated/`; raw harvested candidates
(`tasks/raw/`) are git-ignored and never shipped.
