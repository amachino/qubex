# v1.5.0 Beta Contract Test Scope

## Purpose

Freeze the minimum compatibility test scope required for v1.5.0 beta sign-off.
Last status update: `2026-02-27`.

## Scope policy

- Primary contract surface: `Measurement`
- Delegation smoke surface: `Experiment` core lifecycle and measurement delegation
- Required execution modes: regular path and `mock_mode=True` compatibility path

## Measurement contract matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Primary facade import stability (`Measurement`) | covered | `tests/measurement/test_measurement_alias.py` |
| Backend-kind forwarding (`load(..., backend_kind=...)`) | covered | `tests/measurement/test_backend_kind_selection.py` |
| Legacy API delegation (`execute`, `measure` -> schedule execution) | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Schedule execution custom-factory hook path | covered | `tests/measurement/test_measurement_api_delegation.py` |
| Default executor backend-kind selection (`quel1` / `quel3`) | covered | `tests/measurement/test_measurement_schedule_runner.py` |
| QuEL-3 adapter payload conversion and alias resolution | covered | `tests/measurement/test_quel3_measurement_backend_adapter.py` |
| QuEL-3 controller-to-manager execution contract | covered | `tests/backend/test_quel3_backend_controller.py` |
| Sampling-period source contract (`dt`) | covered | `tests/measurement/test_sampling_period_source.py` |
| QuEL-3 integration compatibility with current `quelware-client` resolver API (`InstrumentResolver`) | partial | `tests/backend/test_quel3_backend_controller.py` (resolver-path payload resolution with resolver-compatible doubles); TODO: add direct import/usage regression against local quelware package layout. |
| Target-to-alias auto-resolution by wiring/port with fail-fast errors on unresolved/ambiguous cases | covered | `tests/backend/test_quel3_backend_controller.py`, `tests/measurement/test_quel3_measurement_backend_adapter.py` |
| Multi-instrument synchronized trigger including cross-unit execution | gap | TODO (new backend execution tests + hardware gate evidence) |
| Capture-mode contract (`avg`=`AVERAGED_VALUE`, `single`=`VALUES_PER_ITER` with legacy `VALUES_PER_LOOP` fallback, waveform inspection=`AVERAGED_WAVEFORM`) | partial | Source implementation in `src/qubex/backend/quel3/managers/execution_manager.py`; TODO: add explicit execution/result contract tests including waveform inspection path. |

## Experiment delegation smoke matrix

| Contract item | Status | Evidence |
| --- | --- | --- |
| Delegation of selected facade APIs (calibration/benchmarking/context methods) | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| `mock_mode=True` forwarding at Experiment creation | covered | `tests/experiment/test_experiment_mock_mode_compat.py` |
| Registry-based resolution path for execute/measure/measure_state | covered | `tests/experiment/test_measurement_service_registry_resolution.py` |
| Core lifecycle smoke: `connect`, `reload`, `run` | covered | `tests/experiment/test_experiment_facade_delegation.py` |
| Core delegation smoke: `measure_idle_states` | covered | `tests/experiment/test_experiment_facade_delegation.py` |

## Beta gate requirement

1. All currently covered items stay green.
2. Any compatibility-surface change must include this scope update in the same PR.
3. All `gap` and `partial` items above are required to be moved to `covered` before beta sign-off.
