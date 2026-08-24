---
name: pr-draft-summary
description: Create a branch suggestion, PR title, and draft description after substantive Qubex changes are finished. Trigger when wrapping up a moderate-or-larger change that touched runtime code, tests, build config, or docs with behavior impact and you need a PR-ready handoff block with summary, compatibility notes, and verification results.
---

# PR Draft Summary

## Overview

Create a review-ready handoff summary from the current diff and repository
conventions. Prefer one focused PR and make verification explicit.

## Workflow

1. Inspect current context.
   - Read `git status --short`, the current branch name, and a concise diff summary.
   - Check the current branch against
     `docs/developer-guide/development-flow.md#branching`.
   - If an existing branch violates the repository convention, flag it instead
     of silently carrying the invalid name into a pull request.
2. Suggest branch naming.
   - Follow the semantic prefixes defined in the repository development flow:
     `feature`, `fix`, `docs`, `refactor`, `test`, or `chore`.
   - Use `<type>/<short-slug>`, such as
     `docs/experiment-task-api-design`.
   - Do not prepend `codex/` or another agent or tool namespace.
3. Draft the PR title.
   - Use a concise prefix such as `fix:`, `docs:`, `refactor:`, or `test:`.
   - Keep the title aligned with the smallest coherent user-visible change.
4. Draft the PR body.
   - background and reason
   - summary of changes
   - user or API impact
   - compatibility or migration notes
   - validation commands and outcomes
   - open risks or follow-ups
5. Use repository guidance.
   - Keep the PR focused on one concern.
   - Mention docs updates when public behavior changed.
   - Include exact verification commands run in the `uv` environment.
6. Keep handoff honest.
   - Separate work completed from follow-up ideas.
   - State if tests, docs build, or release checks were not run.
