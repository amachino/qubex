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

Development commands in this repository assume a `uv`-managed environment.

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

## Release version updates

When you need to update the shared release version for `qubex` and the
publishable companion packages:

1. Set the new version and synchronize all package metadata:
   - `make sync-release-version VERSION=1.5.0b5`
2. Verify that no version drift remains:
   - `make check-release-version`
3. Re-run the normal checks after the version update:
   - `make check`
   - `make build-all`

The shared source of truth is the repository-root `VERSION` file.

## Keep changes reviewable

- Keep pull requests small and focused.
- Split unrelated refactors from behavior changes.
- Update user/developer documentation when behavior or APIs change.

## Naming conventions

See [Naming guidelines](naming-guidelines.md) for naming rules.
