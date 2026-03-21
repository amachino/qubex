---
name: final-release-review
description: Compare the current Qubex release candidate against the previous tag and review release readiness. Use before beta, rc, or GA cuts to check version drift, release notes, compatibility messaging, build health, and validation status; use the bundled script to resolve the comparison range and collect the release diff mechanically.
---

# Final Release Review

## Overview

Review the release candidate from repository state, version files, docs, and
the previous tag. Produce blockers, risks, and missing release tasks while
letting the script resolve the comparison range and diff summary.

## Workflow

1. Resolve candidate context.
   - Read `VERSION`, `pyproject.toml`, and release-note files under
     `docs/release-notes/`.
   - Identify the intended release version from the working tree or target tag.
2. Select comparison tags.
   - Prefer the bundled script:
     `uv run python .agents/skills/final-release-review/scripts/collect_release_diff.py`
   - Compare against the immediate predecessor tag first.
   - If the candidate starts a new stable line, also compare against the latest
     stable predecessor.
3. Gather the diff summary.
   - Use the script output for the base tag, compare range, `git log`, and
     `git diff --stat` summary.
   - Note doc, test, and packaging deltas separately.
4. Check release gates.
   - `make check-release-version`
   - `make check`
   - For release packaging work, `make build-all`
   - For docs-heavy release work, `uv run mkdocs build`
5. Check release artifacts.
   - Release notes and migration guides reflect the actual code.
   - Version strings are synchronized.
   - Compatibility claims match released behavior.
   - Known limitations and non-blockers are documented.
6. Report readiness.
   - Separate blockers, risks, and nice-to-have follow-ups.
   - Call out missing release-note entries, version drift, failing checks, and
     undocumented breaking behavior.

## Bundled script

### `scripts/collect_release_diff.py`

Use this script for deterministic release-review mechanics:

- resolve the previous release tag automatically
- print the compare range
- collect `git log --oneline` and `git diff --stat`

Keep model work focused on interpreting compatibility risk, migration needs, and
release readiness.
