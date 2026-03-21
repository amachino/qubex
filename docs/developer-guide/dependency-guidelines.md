# Dependency guidelines

This project aims to keep dependencies minimal, explicit, and stable for library users. Use these guidelines when adding or modifying dependencies in `pyproject.toml`.

## Where to declare dependencies

- **Runtime dependencies**: `dependencies` in the `[project]` section.
- **Optional features**: `[project.optional-dependencies]` with clear extra names.
- **Tooling / development**: `[dependency-groups].dev`.
- **Documentation**: `[dependency-groups].docs`.

## Adding a dependency

Only add a dependency when at least one of the following is true:

- The functionality is required at runtime for the core library.
- It is needed to parse/serialize supported data formats.
- It is required to keep public APIs stable or compatible.
- There is no reasonable standard library alternative.

Avoid adding dependencies for:

- Convenience only (prefer local utilities).
- Usage limited to notebooks, demos, or experiments.
- Optional visualization or UI features.

## Version pinning policy

We prefer **loose constraints** for libraries while staying safe:

1. **Set a minimum version**: use `>=` for the lowest tested version.
2. **Avoid upper bounds by default**: add upper bounds only when a dependency is known to break compatibility.
3. **Avoid `~=` unless necessary**: it implicitly adds an upper bound.
4. **Pin exact versions only for tools with fragile output** (e.g., image export).

Examples:

- `numpy >= 1.23.5`
- `plotly >= 5.23, <6` (only when needed to avoid a known break)
- `kaleido == 0.2.1` (exact pin for stable exports)

## Workspace companion package versioning

This repository contains companion packages such as `qxcore`, `qxpulse`,
`qxschema`, `qxsimulator`, `qxvisualizer`, `qxfitting`, and
`qxdriver-quel1`. Treat them as one tested bundle with `qubex`, not as
independently supported products by default.

Rules:

1. For public PyPI releases, companion packages that remain runtime
   dependencies of `qubex` must use the same release line as `qubex`
   (for example `1.5.0b4`, `1.5.0`, `1.5.1`).
2. Publish the companion packages first, then publish `qubex`.
3. "Not intended for standalone use" must be stated in docs and support
   policy.

In short:

- public PyPI bundle: companion packages match the `qubex` release line

Operational commands:

- Set and synchronize the shared release version:
  - `make sync-release-version VERSION=1.5.0b5`
- Check that package versions, exact pins, and `uv.lock` are aligned:
  - `make check-release-version`

The shared source of truth is the repository-root `VERSION` file.

## Documentation updates

Whenever you add, remove, or move a dependency:

- Update `pyproject.toml`.
- Note the reason in the PR description.
- If it changes user-visible installation steps, update docs.
