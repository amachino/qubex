---
name: code-change-verification
description: Run the mandatory Qubex verification stack when changes affect runtime code, tests, examples, packaging, or build/test behavior. Use after substantive code changes, and do not mark the work complete until the required checks pass in the `uv` environment.
---

# Code Change Verification

## Overview

Run the repository-required verification stack in the project environment.
Use targeted tests first when the scope is narrow, then run the full gate before
handoff unless the user explicitly asks for a smaller pass.

## Workflow

1. Decide whether the full code gate is needed.
   - If only docs or prose changed, prefer `docs-sync` and `uv run mkdocs build`.
   - Otherwise run the normal code gate.
2. Run normalization checks first.
   - `uv run ruff check --fix`
   - `uv run ruff format`
   - If either command changes files, review the diff instead of silently
     overwriting it.
3. Run static verification.
   - `uv run pyright`
   - If packaging or release metadata changed, also run `make check-release-version`.
4. Run tests.
   - Prefer focused `uv run pytest <paths>` on the touched area first.
   - Run `uv run pytest` before handoff for code or runtime changes unless the
     user explicitly limits scope.
5. Escalate build checks when needed.
   - For release or packaging work, add `make build` or `make build-all`.
   - For docs changes that affect generated API pages or navigation, add
     `uv run mkdocs build`.
6. Report results clearly.
   - List every command you ran.
   - Separate auto-fixed changes from failing checks.
   - Include the exact rerun command for each failure.
   - If required checks are still failing, do not treat the task as complete.

## Repo rules

- Run Python commands with `uv run ...`.
- Treat `uv run ruff check --fix`, `uv run ruff format`, `uv run pyright`,
  and `uv run pytest` as the default verification stack in this repository.
- Do not revert unrelated user changes while cleaning up verification failures.
