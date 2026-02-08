# Pull request guidelines

## Scope

- Keep each pull request focused on one concern.
- Separate behavior changes from broad refactors when possible.

## What to include

- Background and why the change is needed.
- Summary of changes.
- User-facing/API impact.
- Risks and compatibility notes.
- Validation results (commands and outcomes).

## Required checks

Run these in the project environment before opening or updating a PR:

```bash
uv run ruff check
uv run ruff format
uv run pyright
uv run pytest
```

## Documentation updates

Update docs when you change:

- Public APIs.
- Behavioral semantics.
- Setup, dependency, or workflow instructions.

## Review readiness

- Ensure tests are deterministic.
- Avoid unrelated formatting-only noise.
- Resolve reviewer comments with follow-up commits.
