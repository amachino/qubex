# Development flow

This page defines the standard contribution flow for Qubex.

## Branching

- Create a topic branch from the latest default branch.
- Keep branch scope focused on one feature/fix.
- Prefer descriptive names such as:
  - `feature/<short-name>`
  - `fix/<short-name>`
  - `docs/<short-name>`
  - `refactor/<short-name>`

## Recommended steps

1. Sync your local default branch.
2. Create a topic branch.
3. Add or update tests first when possible.
4. Implement the change.
5. Run local quality checks:
   - `uv run ruff check`
   - `uv run ruff format`
   - `uv run pyright`
   - `uv run pytest`
6. Open a pull request with context, impact, and verification results.

## Keep changes reviewable

- Keep pull requests small and focused.
- Split unrelated refactors from behavior changes.
- Update user/developer documentation when behavior or APIs change.
