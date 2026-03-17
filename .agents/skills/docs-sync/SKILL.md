---
name: docs-sync
description: Audit Qubex documentation against code changes and report missing, incorrect, or outdated docs. Use when public APIs, behavior, setup, dependencies, examples, release notes, or developer guidance change; by default, inspect and prioritize the required updates first, then edit only after the user asks for docs changes.
---

# Docs Sync

## Overview

Compare changed code against `README.md`, `docs/user-guide/**`,
`docs/release-notes/**`, and `docs/developer-guide/**`.
By default this is a report-first workflow: inspect, prioritize, and explain
the required documentation changes before editing.

## Workflow

1. Classify the change surface.
   - public API or runtime behavior
   - setup, dependency, or workflow changes
   - examples or notebooks
   - developer-process documentation
   - release or migration surface
2. Read only the high-signal docs.
   - Public workflow: `README.md` and relevant `docs/user-guide/**`
   - Developer workflow: `docs/developer-guide/*.md`
   - Release and compatibility: `docs/release-notes/**`
3. Check for drift.
   - renamed arguments, defaults, return shapes, warnings, and deprecations
   - install or runtime prerequisites
   - examples that no longer match the code
   - stale paths and links
   - generated reference pages that should instead be fixed from source docstrings
     or source comments
4. Apply repository writing rules.
   - Use sentence case headings from `docs/developer-guide/document-guidelines.md`.
   - Use NumPy-style docstrings from `docs/developer-guide/docstring-guidelines.md`.
   - Keep documentation concise and imperative.
   - Treat source docstrings and source comments as the source of truth for any
     generated API reference output.
5. Handle compatibility deliberately.
   - Decide whether migration notes or deprecation docs are required based on
     whether the changed behavior is already released.
   - If released behavior changes, update release notes or migration guidance in
     the same change.
6. Report before editing by default.
   - Produce a focused list of required doc updates, grouped by severity or user impact.
   - Ask before patching docs unless the user explicitly requested doc edits in
     the current task.
7. Validate.
   - Run `uv run mkdocs build` when docs changed or generated API pages could be
     affected.
   - Mention any docs you intentionally left untouched and why.
