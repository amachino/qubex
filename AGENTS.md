- Python commands must run in project env: use `uv run ...`, or `source .venv/bin/activate`
- Use `uv run ruff check --fix`, `uv run ruff format`, `uv run pyright` and `uv run pytest` as the default local quality gate.
- Write docstrings following the guideline in docs/developer-guide/docstring-guidelines.md. Do not use reST syntax.
- Follow the test guideline in docs/developer-guide/test-guidelines.md: write tests first and use 1-line spec docstrings.
- Decide whether to add backward-compatibility handling based on whether the affected code has already been released.

## Mandatory skill usage

- Repo-local skills live under `.agents/skills/`.
- Before editing runtime behavior or public APIs that may affect compatibility boundaries, use `implementation-strategy`.
- If runtime code, tests, examples, or build/test behavior changes, run `code-change-verification` and do not mark the work complete until it passes.
- If the work touches OpenAI API or platform integrations, use `openai-knowledge`.
- When wrapping up a moderate-or-larger change that touched runtime code, tests, build config, or docs with behavior impact, use `pr-draft-summary`.

## Report-first skills

- `docs-sync` and `test-coverage-improver` are report-first workflows by default: inspect the current diff or coverage artifacts, prioritize what matters, and ask before editing unless the user explicitly requested those edits.
- For generated API reference or docs pages, treat source docstrings and source comments as the source of truth. Do not patch generated output by hand.

## Other skills

- Use `docs-sync` when public behavior, setup, workflows, or examples may have drifted from the code.
- Use `examples-auto-run` for non-interactive example validation. Do not auto-run hardware-backed examples by default.
- Use `final-release-review` before beta, rc, or GA release cuts.
- Use `test-coverage-improver` when hardening tests or prioritizing coverage work.
