# Measurement Backend Responsibility Migration Plan (v1.5.0)

## Status

- State: `DRAFT-PLAN`
- Created: 2026-02-22
- Updated: 2026-02-22
- Target policy: `docs/developer-notes/measurement-backend-responsibility-policy.md`

## Goal

Migrate the current implementation to the target architecture defined in `measurement-backend-responsibility-policy.md`, in behavior-preserving increments.

## Progress

- Completed
  - Step 1
  - Step 2
  - Step 2A
  - Step 3
  - Step 4
  - Step 5
  - Step 6
  - Step 7A
  - Step 7B
  - Step 8
- Remaining
  - Step 9
  - Step 10

## Migration Rules

- Complete each step while preserving API compatibility.
- Run existing tests at the end of each step to prevent regressions.
- Keep dependency direction as `measurement -> backend`; do not introduce reverse dependency.
- Avoid big-bang replacements; follow this order: add delegate -> switch call sites -> remove legacy path.

## Step Plan

### Step 1: Split `MeasurementBackendManager` into `MeasurementContext` and `MeasurementSessionService`, then remove `MeasurementBackendManager`

- Purpose
  - Separate measurement-layer responsibilities between context/query access and session/connectivity lifecycle.
- Main changes
  - Add `src/qubex/measurement/measurement_context.py` and move context/query responsibilities there.
  - Add `src/qubex/measurement/measurement_session_service.py` and move `load/connect/reload/disconnect/check_*/linkup/relinkup` there.
  - Remove `MeasurementBackendManager` references from `src/qubex/measurement/measurement_client.py` and delegate to the new services.
  - Remove `src/qubex/measurement/measurement_backend_manager.py`.
- Behavior-preserving guardrails
  - Keep `MeasurementClient` public method names/signatures/return types unchanged.
  - Preserve `SystemManager`-based load/connect/pull/sync sequencing.
- Verification
  - `uv run pytest tests/measurement/test_backend_kind_selection.py`
  - `uv run pytest tests/measurement/test_measurement_api_delegation.py`
  - `uv run pytest tests/measurement/test_sampling_period_source.py`

### Step 2: Delegate execution responsibilities from `MeasurementClient` to `MeasurementExecutionService`

- Purpose
  - Move execution use-case ownership out of facade and into a dedicated service.
- Main changes
  - Add `src/qubex/measurement/measurement_execution_service.py`.
  - Move `create_measurement_config`, `build_measurement_schedule`, `execute_measurement_schedule`, `execute`, `measure`, and `measure_noise` into the service.
  - Move `sampling_period` and `constraint_profile` resolution into the service.
  - Keep `MeasurementClient` as API facade only.
- Behavior-preserving guardrails
  - Port default-value initialization logic exactly.
  - Keep rawdata save behavior, classifier conversion behavior, and return value types unchanged.
- Verification
  - `uv run pytest tests/measurement/test_measurement_api_delegation.py`
  - `uv run pytest tests/measurement/test_measurement_ordering.py`

### Step 2A: Delegate classifier and DC amplification responsibilities from `MeasurementClient`

- Purpose
  - Remove classifier-state and temporary DC-operation ownership from facade.
- Main changes
  - Add `src/qubex/measurement/measurement_classification_service.py`.
  - Add `src/qubex/measurement/measurement_amplification_service.py`.
  - Delegate `classifiers`, `update_classifiers`,
    `get_confusion_matrix`, and `get_inverse_confusion_matrix` from
    `MeasurementClient` to `MeasurementClassificationService`.
  - Delegate `apply_dc_voltages` from `MeasurementClient` to
    `MeasurementAmplificationService`.
  - Keep `MeasurementExecutionService` classifier mapping shared from
    `MeasurementClassificationService`.
- Behavior-preserving guardrails
  - Keep `MeasurementClient` public method names/signatures/return types unchanged.
  - Preserve confusion-matrix normalization and Kronecker-product semantics.
  - Preserve target-to-mux-to-voltage resolution behavior for temporary DC operations.
- Verification
  - `uv run pytest tests/measurement/test_measurement_classification_service.py`
  - `uv run pytest tests/measurement/test_measurement_amplification_service.py`
  - `uv run pytest tests/measurement/test_measurement_api_delegation.py`

### Step 3: Internalize `MeasurementScheduleExecutor` as `MeasurementScheduleRunner`

- Purpose
  - Align naming and ownership with policy: orchestration belongs to `MeasurementExecutionService`.
- Main changes
  - Add `src/qubex/measurement/measurement_schedule_runner.py`.
  - Update `MeasurementExecutionService` to use runner composition.
  - Remove `MeasurementScheduleExecutor` and migrate all call sites/imports to `MeasurementScheduleRunner`.
- Behavior-preserving guardrails
  - Port request-build and result-conversion logic without semantic changes.
- Verification
  - `uv run pytest tests/measurement/test_measurement_schedule_runner.py`
  - Confirm no remaining `MeasurementScheduleExecutor` references in source/tests/docs.

### Step 4: Convert `BackendController` contract to Protocol and define minimal required methods plus capability protocols

- Purpose
  - Make measurement-facing backend contract explicit at type level.
- Main changes
  - Replace union alias in `src/qubex/backend/controller_types.py` with a Protocol-based contract.
  - Required contract: `hash`, `is_connected`, `execute(...)`, `connect(...)`, `disconnect()`.
  - Move backend-dependent operations to separate capability protocols
    (`BackendSkewYamlLoader`, `BackendLinkStatusReader`,
    `BackendClockStatusReader`, `BackendLinkupOperator`,
    `BackendRelinkupOperator`, `BackendClockSynchronizer`,
    `BackendClockResynchronizer`, `BackendBoxConfigProvider`).
  - Align type annotations in `SystemManager` and measurement services with the new contract.
- Behavior-preserving guardrails
  - Keep runtime method resolution unchanged; introduce type strengthening first.
- Verification
  - `uv run pyright`
  - Confirm there are no type errors in backend/measurement packages.

### Step 5: Unify measurement-to-backend execution boundary to `BackendController.execute(...)`

- Purpose
  - Match policy boundary and remove measurement-layer coupling to backend-internal executors.
- Main changes
  - Update `MeasurementScheduleRunner` to call `backend_controller.execute(request=...)`.
  - Remove direct measurement-layer references to `Quel1BackendExecutor` and `Quel3BackendExecutor`.
  - Implement `execute(...)` entrypoint in backend controllers.
- Behavior-preserving guardrails
  - Keep payload contract (`BackendExecutionRequest`) and result-conversion path unchanged.
- Verification
  - `uv run pytest tests/measurement/test_measurement_schedule_runner.py`
  - `uv run pytest tests/backend/test_quel1_backend_executor_mode.py`
  - `uv run pytest tests/measurement/test_quel3_backend_executor.py`

### Step 6: Split QuEL-1 controller into manager-delegation structure

- Purpose
  - Decompose `Quel1BackendController` into connection/clock/execution responsibilities.
- Main changes
  - Add `src/qubex/backend/quel1/managers/connection_manager.py`.
  - Add `src/qubex/backend/quel1/managers/clock_manager.py`.
  - Add `src/qubex/backend/quel1/managers/execution_manager.py`.
  - Add `src/qubex/backend/quel1/quel1_runtime_context.py` and make
    `Quel1ConnectionManager` the runtime-state writer.
  - Make `Quel1ClockManager` and `Quel1ExecutionManager` read runtime state
    through `Quel1RuntimeContext` read-only interface.
  - Add `src/qubex/backend/quel1/managers/configuration_manager.py` and
    delegate `dump_*`, `config_*`, and configuration-definition operations.
  - Move qubecalib-compatibility modules into
    `src/qubex/backend/quel1/compat/` and use concise module names
    (`box_adapter.py`, `driver_loader.py`, `qubecalib_protocols.py`,
    `sequencer.py`, `capture_result_parser.py`,
    `parallel_action_builder.py`, `sequencer_execution_engine.py`).
  - Make `Quel1ConnectionManager` the owner of connection runtime state
    (`boxpool`, `quel1system`, `cap_resource_map`, `gen_resource_map`) and
    remove tuple-based connected-state returns from `connect(...)`.
  - Reduce `Quel1BackendController` to a thin facade delegating to managers.
- Behavior-preserving guardrails
  - Preserve existing public methods and side-effect ordering (connect/reconnect/linkup/clock sync).
- Verification
  - `uv run pytest tests/backend/test_quel1_backend_controller_connect.py`
  - `uv run pytest tests/backend/test_quel1_backend_controller_clocks.py`
  - `uv run pytest tests/backend/test_quel1_backend_controller_parallel_execution.py`
  - `uv run pytest tests/backend/test_quel1_backend_controller_resource_cleanup.py`

### Step 7: Apply the same manager structure to QuEL-3 controller

- Purpose
  - Keep QuEL-1 and QuEL-3 controller architecture consistent.
- Main changes
  - Remove `Quel3BackendController` class inheritance from
    `Quel1BackendController` and switch to explicit manager delegation.
  - Remove the QuEL-1 `control_plane` concept from QuEL-3 path.
  - Implement QuEL-3 connection/execution natively through
    `quelware-client`.
  - Add `src/qubex/backend/quel3/managers/connection_manager.py`.
  - Add `src/qubex/backend/quel3/managers/execution_manager.py`.
  - Add `src/qubex/backend/quel3/managers/sequencer_builder.py`.
  - Add `src/qubex/backend/quel3/quel3_runtime_context.py`.
  - Update `Quel3BackendController` to delegate to managers and implement shared `execute(...)` contract.
  - Remove standalone QuEL-3 sequencer module and absorb builder into manager package.
- Behavior-preserving guardrails
  - Route measurement boundary only through `execute(...)`.
  - Keep QuEL-1-only helper APIs unsupported on QuEL-3.
  - Keep QuEL-3 without `box_config` capability.
- Verification
  - `uv run pytest tests/backend/test_quel3_backend_controller.py`
  - `uv run pytest tests/backend/test_quel3_sequencer_builder.py`
  - `uv run pytest tests/measurement/test_quel3_backend_executor.py`
  - `uv run pytest tests/measurement/test_quel3_measurement_backend_adapter.py`

### Step 8: Lock `MeasurementSessionService` and `SystemManager` collaboration boundary per policy

- Purpose
  - Make measurement-side session service the clear owner of lifecycle operations.
- Main changes
  - Formalize `load/reload` as session-service + system-manager coordinated flow.
  - Gate optional capability APIs based on backend support.
  - Move active controller/session resolution responsibility into session service.
  - Delegate backend-specific synchronization logic from `SystemManager` to
    `Quel1SystemSynchronizer` / `Quel3SystemSynchronizer`.
- Behavior-preserving guardrails
  - Preserve externally visible behavior from `ExperimentContext.connect/reload/configure`.
- Verification
  - `uv run pytest tests/experiment/test_experiment_context_sampling_period_sync.py`
  - `uv run pytest tests/backend/test_system_manager.py`

### Step 9: Align facade naming and exports to policy baseline

- Purpose
  - Make `Measurement` the canonical facade name while retaining `MeasurementClient` compatibility.
- Main changes
  - Make `src/qubex/measurement/measurement.py` the primary facade entrypoint.
  - Keep `MeasurementClient` as compatibility alias.
  - Adjust exports in `src/qubex/measurement/__init__.py` and `src/qubex/__init__.py`.
- Behavior-preserving guardrails
  - Preserve existing import compatibility for `MeasurementClient`.
- Verification
  - `uv run pytest tests/measurement/test_measurement_client_alias.py`
  - `uv run pytest tests/backend/test_backend_public_api.py`

### Step 10: Clean compatibility layer and run final quality gates

- Purpose
  - Remove temporary migration artifacts and converge to policy target structure.
- Main changes
  - Minimize temporary compatibility aliases/classes/modules.
  - Close remaining deltas against target directory and naming baseline.
  - Update cross-referenced development notes where needed.
- Behavior-preserving guardrails
  - Keep required public compatibility paths via aliases; avoid breaking changes.
- Verification
  - `uv run ruff format`
  - `uv run ruff check`
  - `uv run pyright`
  - `uv run pytest`

## Done Criteria

- `Measurement` is API-focused and delegates non-API responsibilities to `MeasurementContext`, `MeasurementSessionService`, and `MeasurementExecutionService`.
- `Measurement` is API-focused and delegates non-API responsibilities to `MeasurementContext`, `MeasurementSessionService`, `MeasurementExecutionService`, `MeasurementClassificationService`, and `MeasurementAmplificationService`.
- Measurement-layer execution boundary call is only `BackendController.execute(...)`.
- QuEL-1 and QuEL-3 controllers both follow manager-delegation structure.
- `SystemManager` remains focused on state synchronization and is not the owner of backend operation implementations.
- Compatibility APIs are preserved and full quality gates pass.
