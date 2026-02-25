# v1.5.0 Beta Release Notes Template

## Release summary

- Version: `v1.5.0-betaX`
- Date: `YYYY-MM-DD`
- Target audience: experiment users and backend developers

## Highlights

- QuEL-3 controller path added via `quelware-client` integration.
- `Measurement` compatibility contract preserved for core flows.
- Sampling-period handling shifted from fixed `2 ns` assumptions toward backend-derived `dt`.
- Synchronized execution supports multiple instruments including cross-unit trigger paths.
- QuEL-3 result-mode behavior follows quelware capture-mode contract.

## Compatibility contract (beta scope)

- Primary contract surface: `Measurement`
- Delegation smoke scope: `Experiment` core lifecycle and measurement calls
- Required compatibility mode: `mock_mode=True`

## Migration notes

- Configuration:
  - Backend selection precedence: explicit argument > `system.yaml` > `chip.yaml` > `quel1` default.
  - QuEL-3 prefers `wiring.v2.yaml` when present.
  - Target-to-instrument mapping is auto-resolved from wiring/port consistency; unresolved or ambiguous mapping fails fast.
  - Runtime defaults are provisionally aligned to `quelware-client` defaults (`endpoint=localhost`, `port=50051`, `trigger_wait=1000000`, `ttl_ms=4000`, `tentative_ttl_ms=1000`).
- Measurement:
  - Backend kind may change adapter/executor path under `Measurement`.
  - Result timing metadata should be consumed from `sampling_period_ns`.
  - Capture mode contract for QuEL-3:
    - `avg` -> `CaptureMode.AVERAGED_VALUE`
    - `single` -> `CaptureMode.VALUES_PER_LOOP`
    - waveform inspection paths -> `CaptureMode.AVERAGED_WAVEFORM`

## Known limitations (beta)

- QuEL-3 orchestration scope is single-session, single backend-family per experiment session.
- Async task primitives and sweep API unification are still in progress.
- Some contrib/visualization paths may still rely on legacy sampling-period assumptions.

## Validation status

| Gate | Status | Evidence |
| --- | --- | --- |
| `uv run ruff check` | TBD |  |
| `uv run ruff format` | TBD |  |
| `uv run pyright` | TBD |  |
| `uv run pytest` | TBD |  |
| Hardware validation (QuEL-1) | TBD |  |
| Hardware validation (QuEL-3) | TBD |  |
| Auto target-to-alias resolution (`wiring/port` aligned, fail-fast on unresolved/ambiguous) | TBD |  |
| Multi-instrument synchronized trigger (cross-unit) | TBD |  |
| Capture-mode contract (`avg`=`AVERAGED_VALUE`, `single`=`VALUES_PER_LOOP`) | TBD |  |
| Synchronized scenario (`SP-BETA-001`) | TBD |  |

## Breaking changes

- `None` or list explicitly.

## Upgrade guidance

1. Confirm backend selection and wiring file policy in environment config.
2. Confirm target-to-instrument auto-resolution works in your wiring layout and fails fast on unresolved/ambiguous mappings.
3. Run compatibility tests and at least one cross-unit synchronized measurement smoke run before full migration.
4. Validate downstream tools consume `sampling_period_ns` metadata and mode-specific result semantics.

## Rollback plan

1. Revert to latest v1.4.x tag in deployment environment.
2. Re-apply known-good configuration set for previous backend path.
3. Restore previous experiment notebooks/scripts and rerun smoke checks.
