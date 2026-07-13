---
name: test-coverage-improver
description: Run Qubex coverage, identify the highest-value gaps, and propose focused tests. Use when improving test depth after code changes, debugging weakly covered behavior, or preparing for release hardening; by default, inspect coverage artifacts and recommend the best test additions before editing tests.
---

# Test Coverage Improver

## Overview

Use coverage to prioritize missing tests, but optimize for contract risk instead
of raw percentage. Favor released compatibility surfaces, public APIs, and
recent regressions. By default this is a report-first workflow.

## Workflow

1. Run coverage in the project environment.
   - `make coverage`
   - or `uv run pytest --cov=qubex --cov-report=term-missing`
2. Interpret gaps by risk.
   - Start with changed files and public APIs.
   - Then inspect low-coverage runtime modules with user-visible behavior.
   - Give extra weight to released compatibility paths and regression-prone branches.
3. Design tests using repository rules.
   - Follow `docs/developer-guide/test-guidelines.md`.
   - Write tests first when implementing.
   - Use one-line spec docstrings.
   - Prefer observable behavior over private implementation checks.
4. Pick the highest-value additions.
   - boundary cases
   - error and warning paths
   - compatibility shims that are already released
   - config resolution and result-shape regressions
5. Report recommendations clearly.
   - name the file or behavior gap
   - explain why it is high value
   - give the narrowest test file location to add
6. Ask before editing by default.
   - Recommend the top additions first unless the user explicitly asked you to
     write the tests in the current task.
7. Re-run targeted validation after adding tests.
   - `uv run pytest <target>`
   - then the broader gate as needed
