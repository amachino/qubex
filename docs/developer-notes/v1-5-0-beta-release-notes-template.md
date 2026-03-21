# v1.5.0 Internal Beta Release Notes Draft

## Release summary

- Version: current workspace snapshot `v1.5.0b4`
- Date: `2026-03-15` (last updated)
- Target audience: internal reviewers and existing QuEL-1 users
- Beta-scope baseline: internal compatibility contract frozen for `v1.5.0b1`

## Highlights

- QuEL-1 backward compatibility is the primary sign-off target.
- `Measurement` and `Experiment` compatibility contracts are preserved for core QuEL-1 flows.
- Sampling-period handling shifted from fixed `2 ns` assumptions toward backend-derived `dt`.
- Internal artifact bundle is validated with companion packages aligned to the same release line as `qubex`.
- QuEL-3 work remains in progress and is explicitly out of scope for internal `v1.5.0b1` sign-off.

## Compatibility contract (beta scope)

- Primary contract surface: `Measurement` on QuEL-1
- Delegation smoke scope: `Experiment` core lifecycle and measurement calls on QuEL-1
- Required compatibility mode: `mock_mode=True`

## Versioning and packaging

- `qubex` is versioned as `1.5.0b1`.
- Internal companion packages share the current `qubex` release line in this workspace snapshot:
  - `qxcore`
  - `qxfitting`
  - `qxpulse`
  - `qxschema`
  - `qxsimulator`
  - `qxvisualizer`
  - `qxdriver-quel1`
- Treat the companion packages above as one tested internal bundle for this beta; they are not independently versioned compatibility promises.

## Migration notes

- Configuration:
  - Backend selection precedence: explicit argument > `system.yaml` > `quel1` default.
  - Existing QuEL-1 deployments should be validated without changing companion package versions from the tested internal bundle.
- Measurement:
  - Backend kind may change adapter/executor path under `Measurement`.
  - Result timing metadata should be consumed from `sampling_period_ns`.
  - Existing QuEL-1 notebooks/scripts should be checked for unchanged behavior in `execute`, `measure`, and delegation-heavy `Experiment` flows.
  - On QuEL-3, legacy `reset_awg_and_capunits()` and `SystemManager.modified_backend_settings(...)` requests are treated as compatibility no-op when the backend does not expose the corresponding QuEL-1-only capability.
  - QuEL-3 spectroscopy and far-detuned retune behavior still require explicit validation because the coarse/fine sweep contract is not frozen.

## Known limitations (beta)

- QuEL-3 compatibility and hardware validation are tracked separately and are not part of internal `v1.5.0b1` sign-off.
- QuEL-3 compatibility fallback currently prefers continuing execution over raising for unsupported reset/backend-settings override requests; this does not by itself guarantee that a requested sweep range is valid on hardware.
- Internal companion packages are released as one aligned bundle, so bundle-level validation is required whenever artifacts are rebuilt.
- Some contrib/visualization paths may still rely on legacy sampling-period assumptions.

## Validation status

| Gate | Status | Evidence |
| --- | --- | --- |
| `uv run ruff check` | PASS | `2026-03-15` |
| `uv run ruff format --check` | PASS | `2026-03-15` |
| `uv run pyright` | PASS | `2026-03-15` |
| `uv run pytest` | PASS | `2026-03-15` (`882 passed`) |
| `make build-all` | PASS | `2026-03-15` |
| Hardware validation (QuEL-1) | TBD |  |
| Existing user backward compatibility smoke | TBD |  |
| Hardware validation (QuEL-3) | N/A | Out of scope for internal `v1.5.0b1` |

## Breaking changes

- `None` or list explicitly.

## Upgrade guidance

1. Confirm backend selection and wiring file policy in environment config.
2. Validate existing QuEL-1 notebooks/scripts against the current internal beta bundle without changing companion package versions.
3. Run compatibility tests and at least one existing-user smoke scenario before wider internal rollout.
4. Validate downstream tools consume `sampling_period_ns` metadata where timing-sensitive processing exists.

## Rollback plan

1. Revert to latest v1.4.x tag in deployment environment.
2. Re-apply the known-good companion package bundle used before the current beta snapshot.
3. Restore previous experiment notebooks/scripts and rerun smoke checks.
