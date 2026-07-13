---
name: implementation-strategy
description: Choose the Qubex compatibility boundary and implementation approach before editing runtime code, public APIs, imports, config formats, or behavior. Use before making API or runtime changes, especially when you must decide between direct cleanup and backward-compatibility handling.
---

# Implementation Strategy

## Overview

Use this skill before editing behavior that users may observe. The first
decision is whether the touched surface is already released, because
compatibility handling in this repository depends on release status.

## Workflow

1. Decide release status first.
   - Inspect tags, release notes, migration guides, and public docs to see
     whether the current surface has shipped.
   - If the surface is not released yet, prefer the clean implementation over
     compatibility shims.
   - If the surface is already released, preserve compatibility unless the user
     explicitly wants a breaking change with migration work.
2. Classify the change.
   - public API signature or import path
   - runtime behavior or result shape
   - config file format or lookup rules
   - internal refactor with no user-visible change
3. Choose the owner.
   - For `Experiment` APIs, follow
     `docs/developer-guide/experiment-api-delegation-guidelines.md`.
   - Keep `Experiment` thin and move logic into the owning context or service.
   - Do not couple `Experiment` directly to measurement internals.
4. Plan supporting work before edits.
   - Add or update tests first.
   - Decide required docs, migration notes, deprecation warnings, or
     compatibility aliases.
   - Identify the narrowest safe edit set.
5. Write the implementation plan.
   - state the release-status decision
   - state the compatibility boundary
   - state the owner or layer for each code change
   - state the required tests and docs

## Qubex rule

Decide whether to add backward-compatibility handling based on whether the
affected code has already been released.
