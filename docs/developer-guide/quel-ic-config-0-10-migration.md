# quel_ic_config 0.10 migration plan for qubex and qube-calib

## Scope and objective

This document identifies the required work to migrate `qubex` and `packages/qube-calib` from `quel_ic_config` 0.8.x to 0.10.x.

- Migration target: `quel_ic_config` 0.10.x (current upstream `main`: 0.10.7)
- Primary references:
  - <https://github.com/quel-inc/quelware/blob/main/quel_ic_config/docs/MIGRATION_TO_0_10_X.md>
  - <https://github.com/quel-inc/quelware/blob/main/quel_ic_config/docs/GETTING_STARTED.md>
  - <https://github.com/quel-inc/quelware/blob/main/quel_ic_config/src/quel_ic_config/__init__.py>

## Confirmed breaking changes relevant to this repository

- `Quel1BoxWithRawWss` is not exported in 0.10.x; use `Quel1Box`.
- WSS execution APIs changed:
  - `start_emission` / `capture_start` flow is replaced by:
    - `start_wavegen(...)`
    - `start_capture_now(...)`
    - `start_capture_by_awg_trigger(...)`
- AWG/capture parameter model changed:
  - old: `e7awgsw.WaveSequence`, `e7awgsw.CaptureParam`
  - new: `e7awghal.AwgParam`, `e7awghal.WaveChunk`, `e7awghal.CapParam`, `e7awghal.CapSection`
- `config_channel(..., wave_param=...)` is replaced by `config_channel(..., awg_param=...)`.
- Method names changed:
  - `initialize_all_awgs()` -> `initialize_all_awgunits()`
  - `read_current_and_latched_clock()` -> `get_current_timecounter()` + `get_latest_sysref_timecounter()`
- `quel_clock_master` package is no longer present in upstream `quelware/main`; clock sync path is integrated with `e7awghal`/`quel_ic_config` (`QuelClockMasterV1`).
- Top-level `quel_ic_config` exports differ; direct import of `CaptureReturnCode` from `quel_ic_config` is not valid as-is.

## Impacted files in this repository

### qube-calib

- `packages/qube-calib/pyproject.toml`
- `packages/qube-calib/src/qubecalib/qubecalib.py`
- `packages/qube-calib/src/qubecalib/sysconfdb.py`
- `packages/qube-calib/src/qubecalib/e7utils.py`
- `packages/qube-calib/src/qubecalib/instrument/quel/quel1/driver/common.py`
- `packages/qube-calib/src/qubecalib/instrument/quel/quel1/driver/single.py`
- `packages/qube-calib/src/qubecalib/instrument/quel/quel1/driver/multi.py`
- `packages/qube-calib/tests/unit/task/quelware/direct/conftest.py`
- `packages/qube-calib/tests/unit/task/quelware/direct/test_direct.py`
- `packages/qube-calib/tests/unit/task/quelware/direct/test_single.py`
- `packages/qube-calib/tests/unit/task/quelware/direct/test_multi.py`

### qubex

- `src/qubex/backend/quel1/quel1_backend_controller.py`
- `src/qubex/backend/quel1/execution/parallel_action_builder.py`

## Work breakdown structure

## Phase 0: dependency and compatibility baseline

- Update `packages/qube-calib/pyproject.toml`:
  - bump `quel_ic_config` to 0.10.x
  - remove `e7awgsw` dependency
  - remove `quel_clock_master` dependency
- Update version-assert test:
  - `packages/qube-calib/tests/unit/task/quelware/direct/test_direct.py`

## Phase 1: qube-calib direct driver migration (core)

- Replace box type usage:
  - `Quel1BoxWithRawWss` -> `Quel1Box`
- Replace execution orchestration:
  - remove `prepare_for_emission`/`start_emission`/`capture_start` sequence
  - implement new flow with:
    - `start_wavegen`
    - `start_capture_now` or `start_capture_by_awg_trigger`
  - ensure all returned task objects call `result()` and handle cancel/error semantics
- Replace parameter types:
  - `WaveSequence` -> `AwgParam` + `WaveChunk`
  - `CaptureParam` -> `CapParam` + `CapSection`
- Update `config_channel` call sites:
  - `wave_param=` -> `awg_param=`
- Update capture result extraction:
  - old path relied on `(CaptureReturnCode, raw ndarray dict)`
  - new path must read from `CapIqDataReader` (`as_wave_dict` / `as_wave_list`)

## Phase 2: qube-calib timing and clock sync migration

- Replace `QuBEMasterClient` and `SequencerClient` usage:
  - migrate to `QuelClockMasterV1` + box counter APIs where needed
- Replace time read methods:
  - `read_current_and_latched_clock()` calls in multi-box sync logic
  - use `get_current_timecounter()` and `get_latest_sysref_timecounter()`
- Revisit timed reservation assumptions:
  - migration note states timed reservation should be operated with one reservation policy
  - validate existing multi-reservation behavior in `multi.Action.emit_at`

## Phase 3: qubex integration adaptation

- `src/qubex/backend/quel1/quel1_backend_controller.py`
  - adapt lazy imports to migrated qube-calib interfaces
  - keep linkup patch compatibility (`LinkupFpgaMxfe`) under new version
- `src/qubex/backend/quel1/execution/parallel_action_builder.py`
  - protocol methods currently assume old APIs (`read_current_and_latched_clock`, `reserve_emission` path)
  - align protocol and logic with 0.10 task-based APIs

## Phase 4: tests and verification

- Unit tests:
  - rewrite direct-driver tests around `AwgParam`/`CapParam` and task results
- Static checks:
  - `uv run ruff check`
  - `uv run pyright`
- Test execution:
  - `uv run pytest`
- Formatting:
  - `uv run ruff format`

## Recommended implementation order

1. Phase 0 first (dependency switch).
2. Complete Phase 1 on `qube-calib` direct driver.
3. Complete Phase 2 for timing/clock synchronization.
4. Adapt `qubex` (Phase 3) after qube-calib API surface stabilizes.
5. Execute Phase 4 at each milestone and once at final integration.

## Risks and open decisions

- `CaptureReturnCode` handling strategy:
  - keep status at driver layer (if accessible from lower module), or
  - simplify to exception-based success/failure with `CapIqDataReader`.
- DSP pipeline mapping:
  - old `DspUnit`-based parsing in `parse_capture_result` must be revalidated against 0.10 `CapParam` semantics.
- Clock synchronization API contract:
  - decide whether to keep a qube-calib-local abstraction compatible with old naming, or fully expose 0.10 naming.

## Definition of done

- No remaining imports of `e7awgsw`, `quel_clock_master`, or `Quel1BoxWithRawWss` in `qube-calib` and `qubex`.
- All direct-driver code paths run via 0.10 task-based capture/wavegen APIs.
- `uv run ruff check`, `uv run pyright`, and `uv run pytest` pass for the migrated scope.
- This migration document remains aligned with the implemented code.
