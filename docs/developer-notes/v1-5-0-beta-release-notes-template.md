# v1.5.0 Internal Beta Release Notes Template

## Release summary

- Version: `v1.5.0b1`
- Date: `YYYY-MM-DD`
- Target audience: internal reviewers and existing QuEL-1 users

## Highlights

- QuEL-1 backward compatibility is the primary sign-off target.
- `Measurement` and `Experiment` compatibility contracts are preserved for core QuEL-1 flows.
- Sampling-period handling shifted from fixed `2 ns` assumptions toward backend-derived `dt`.
- Internal artifact bundle is validated with companion packages kept at `0.0.0.dev0`.
- QuEL-3 work remains in progress and is explicitly out of scope for internal `v1.5.0b1` sign-off.

## Compatibility contract (beta scope)

- Primary contract surface: `Measurement` on QuEL-1
- Delegation smoke scope: `Experiment` core lifecycle and measurement calls on QuEL-1
- Required compatibility mode: `mock_mode=True`

## Versioning and packaging

- `qubex` is versioned as `1.5.0b1`.
- Internal companion packages remain at `0.0.0.dev0`:
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

## Known limitations (beta)

- QuEL-3 compatibility and hardware validation are tracked separately and are not part of internal `v1.5.0b1` sign-off.
- Internal companion packages remain `0.0.0.dev0`, so bundle-level validation is required whenever artifacts are rebuilt.
- Some contrib/visualization paths may still rely on legacy sampling-period assumptions.

## Validation status

| Gate | Status | Evidence |
| --- | --- | --- |
| `uv run ruff check` | PASS | `2026-03-11` |
| `uv run ruff format --check` | PASS | `2026-03-11` |
| `uv run pyright` | PASS | `2026-03-11` |
| `uv run pytest` | PASS | `2026-03-11` (`794 passed`) |
| `make build-all` | PASS | `2026-03-11` |
| Hardware validation (QuEL-1) | TBD |  |
| Existing user backward compatibility smoke | TBD |  |
| Hardware validation (QuEL-3) | N/A | Out of scope for internal `v1.5.0b1` |

## Breaking changes

- `None` or list explicitly.

## Upgrade guidance

1. Confirm backend selection and wiring file policy in environment config.
2. Validate existing QuEL-1 notebooks/scripts against the internal `v1.5.0b1` bundle without changing companion package versions.
3. Run compatibility tests and at least one existing-user smoke scenario before wider internal rollout.
4. Validate downstream tools consume `sampling_period_ns` metadata where timing-sensitive processing exists.

## Rollback plan

1. Revert to latest v1.4.x tag in deployment environment.
2. Re-apply the known-good companion package bundle used before `v1.5.0b1`.
3. Restore previous experiment notebooks/scripts and rerun smoke checks.
