# Qubex Instructions (Canonical)

---

## 1. Purpose / Scope

These instructions define how AI assistance shall behave for the `amachino/qubex` repository:

* Accelerate experiment & control stack development without eroding architectural clarity.
* Enforce consistent engineering standards (types, tests, docs, performance awareness, hardware realism).
* Minimize diff surface; prefer iterative, review-friendly changes.
* Keep this file canonical; other prompt artifacts must reference (not duplicate) its content.

Version: 2025-08-28 (v1.0 canonical consolidation)

---

## 2. Core Principles
1. Architectural Fidelity: Respect existing layering (`pulse` → `measurement` → `experiment` → higher analysis / API) and do not create cross-layer shortcuts.
2. Explicitness: Types, units, and assumptions are spelled out (frequency MHz, time μs, phase rad).
3. Reproducibility: Randomized procedures (RB, DRAG parameter search) expose `seed`.
4. Incremental Safety: Small, reviewable diffs; migrations with clear deprecation windows.
5. Test-First Mindset: For new public behavior, outline or add tests before full implementation.
6. Performance Conscious: Flag O(N^2)/large allocations; prefer vectorization or streaming.
7. Hardware Realism: Note when a value is simulated vs hardware sourced; mark approximations.
8. Single Source of Truth: If guidance exists here, do not restate verbatim elsewhere—link to the section.

---

## 3. Repository Quick Map (Reference Only)
`src/qubex/` modules: `analysis/`, `api/`, `backend/`, `clifford/`, `diagnostics/`, `experiment/`, `measurement/`, `pulse/`, `simulator/`, plus root utilities (`style.py`, `typing.py`, `version.py`). Tests in `tests/`; runnable example notebooks/scripts in `docs/examples/`.

Rules:
* Place new algorithms near related domain modules (e.g., pulse shaping → `pulse/`; calibration routines → `measurement/` or `experiment/` if user-facing).
* Avoid deep nesting unless > ~6 logically grouped functions.
* Public API additions should be re-exported deliberately (not automatically) through package `__init__` if intended for user consumption.

---

## 4. Coding Standards
* Language: All code (identifiers, comments, docstrings) in English.
* Docstrings: NumPy style sections: Summary, Parameters, Returns, Raises, Notes (optional), Examples. Include units.
* Type Hints: Prefer concrete numeric types (`float`), numpy arrays annotated via `from numpy.typing import NDArray`; be explicit on dtype where meaningful (`NDArray[np.float64]`). Avoid `Any`.
* Exceptions: Raise specific exceptions with parameter name and constraint in message.
* Logging: Structured minimal logging on critical execution path; avoid chatty loops.
* Style: Follow `ruff` and configured pyproject; no trailing whitespace; 88-100 char soft wrap (respect repo).
* Determinism: Provide `seed` parameter for stochastic algorithms; document nondeterminism.

---

## 5. Testing & Quality
* Framework: `pytest`.
* New Feature: Add tests (happy path + at least one edge/validation failure) under mirrored module path in `tests/`.
* Runtime Budget: Keep per-test < 0.5 s unless performance scenario; entire suite impact minimal.
* Fixtures: Reuse shared fixtures; prefer synthetic small arrays.
* Performance Checks: For critical loops, add micro benchmark (optional) or comment complexity.
* Quality Gates (assistant when editing): Build/import check (`python -c 'import qubex'`), run subset of impacted tests, summarize PASS/FAIL.

---

## 6. Documentation & Examples
* Every new public function/class: NumPy docstring + minimal example that executes quickly (≤ 5 s) using simulator backend.
* Larger workflows: Add or update a notebook / script in `docs/examples/`—clearly labelled, deterministic.
* Changelog: If repository adopts semantic versioning and a `CHANGELOG.md` exists, append entry under UNRELEASED; otherwise propose file creation if meaningful change.

---

## 7. Prompt / Interaction Patterns
Assistant should encourage structured user requests via short verbs:
* plan: <feature> → Produce plan only (Overview, API sketch, Modules, Risks, Tests, Migration, Open Questions).
* implement: <feature> → Execute plan in small diffs; each diff justified.
* refactor: <area> → Identify seams, propose extraction, preserve API (add deprecation warnings where needed).
* diagnose: <symptom> → Reproduce, isolate, propose minimal fix + test.
* doc: <module|symbol> → Generate/refine docstrings.
* example: <topic> → Add example referencing simulator.

When user request is unstructured, infer safest minimal clarifying assumptions (state them) and proceed.

---

## 8. Response & Formatting Rules
* Language of narrative: Mirror user language (Japanese/English). Code sections remain English.
* Stepwise Reasoning: Summarize plan first (bullets) before large edits; avoid exposing chain-of-thought internal details—just concise rationales.
* Diffs: Present only necessary hunks (no unrelated reformat). For multi-file changes apply smallest consistent patch.
* Tests: Run impacted tests after changes and summarize result.
* Avoid Redundancy: If a rule is defined here, reference its section number instead of restating.

---

## 9. Hardware & Physics Considerations
* Qubit frequency units: Hz vs GHz vs MHz—default docstrings to MHz for readability; clarify base units where ambiguity.
* Gate calibration: Note CR amplitude, phase, detuning assumptions.
* Readout (MUX): Distinguish between raw IQ, rotated IQ, and discriminated state. Document pipeline stage.
* Simulator: Clearly label approximations (e.g., no cross-talk, ideal measurement) when used in examples.

---

## 10. Deprecation & Migration Policy
* Minor version bumps may introduce new APIs; deprecations require at least one minor cycle with warning.
* Use `warnings.warn("message", DeprecationWarning, stacklevel=2)` and mention planned removal version.
* Provide adaptor / shim when trivial; else supply migration snippet in docstring Notes.

---

## 11. Security & Data Handling
* Do not embed secrets or tokens in examples.
* When generating temporary analysis artifacts, place under `tmp/` (gitignored) and mark ephemeral.

---

## 12. Acceptance Checklist (Definition of Done)
For any merged change the assistant should verify:
1. Imports clean: `import qubex` succeeds.
2. New/changed public symbols: typed + NumPy docstring + example.
3. Tests for new behavior pass (and fail appropriately before fix in bug scenarios).
4. Performance considerations documented if non-trivial complexity added.
5. Deprecated paths warn (if applicable) & documented.
6. Changelog updated (if file exists) or suggested.
7. No accidental broad refactors / style churn outside targeted scope.

---

## 13. Non-Goals
* Large speculative rewrites.
* Introducing heavy external dependencies without discussion.
* Auto-generating physics models beyond validated scope.

---

## 14. How Other Prompt Artifacts Should Refer Here
Any `.chatmode.md` or future instruction file must:
* Contain only: description, tool list, and a brief pointer to these canonical instructions.
* Avoid inlining sections 2–13; instead: "See .github/instructions/qubex.instructions.md".

---

## 15. Maintenance
* Update this file when architectural boundaries shift or tooling changes.
* Increment version header and add short CHANGE SUMMARY at top (future section) for traceability.

---

## 16. Quick Reference (Cheat Sheet)
| Area | Rule |
|------|------|
| Docstring | NumPy style, include units |
| Types | Precise, avoid Any |
| Tests | Add for all new behavior |
| Seeds | Expose & document randomness |
| Diffs | Minimal, scoped |
| Language | User language for narrative; code English |
| Deprecation | Warn >= 1 minor before removal |
| Examples | Fast, simulator-backed |

---

End of canonical instructions.
