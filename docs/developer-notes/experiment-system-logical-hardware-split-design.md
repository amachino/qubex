# Experiment system logical/hardware configuration split design

## Status

- State: `PROPOSED`
- Created: 2026-03-05
- Updated: 2026-03-05
- Related notes:
  - `quel3-configuration-design.md`
  - `quel3-control-system-model-design.md`
  - `system-package-quel1-quel3-boundary.md`

## Purpose

Define the final split between:

- logical model construction in `ExperimentSystem` (`TargetRegistry`)
- hardware configuration at `push()` time in backend-specific components

This note focuses on the QuEL-3 path and removes intermediate models that are not required in the current scope.

## Final design summary

1. `ExperimentSystem` keeps logical ownership only.
2. QuEL-3 hardware deployment is executed during `SystemManager.push(...)`.
3. QuEL-3 planning is implemented by a system-side planner that consumes
   `TargetRegistry`.
4. QuEL-3 deploy execution is implemented by a backend-side
   `Quel3ConfigurationManager`.
5. `InstrumentMode.FIXED_TIMELINE` is fixed in this phase.
6. `backend_settings_pull` remains unsupported on QuEL-3.
7. QuEL-3 does not persist `ExperimentSystem` inside the controller or
   synchronizer just to support `push()`.

## Scope

In scope:

- QuEL-3 push-time `deploy_instruments` flow from `TargetRegistry`
- role and frequency-range derivation from `Target`
- synchronizer and configuration-manager responsibility boundary

Out of scope:

- QuEL-1 retuning behavior redesign
- QuEL-3 LO/CNCO/FNCO readback standardization
- extended role variants (`RECEIVER`, `TRANSCEIVER_LOOPBACK`) in v1.5.0

## Current issue

Current `ExperimentSystem.configure()` still mixes:

- logical target registry construction
- low-level port/channel value mutation

For QuEL-3, low-level values should be selected by quelware at deploy time.  
Therefore configuration-time responsibility must move from `ExperimentSystem` to QuEL-3 backend configuration flow.

## Architecture boundary

### Common layer

Owner: `ExperimentSystem`

Responsibilities:

- maintain `TargetRegistry`
- provide logical target metadata (`Target.type`, `Target.channel.port`, `Target.frequency`)
- stay backend-neutral

### QuEL-3 configuration layer

Owners:

- `Quel3SystemSynchronizer` as push entrypoint
- system-side target deploy planner as logical-to-runtime converter
- backend-side `Quel3ConfigurationManager` as deploy owner

Responsibilities:

- planner:
  - convert selected targets to one-instrument-per-target deploy requests
  - derive role, port ID, alias, and frequency ranges from logical metadata
- backend configuration manager:
  - create `InstrumentDefinition` with fixed timeline profile
  - call `session.deploy_instruments(...)`
  - cache deployed instrument infos for execution path

### QuEL-3 execution layer

Owner: execution manager

Responsibilities:

- consume deployed instrument aliases/resources
- run fixed timeline execution and fetch measurement results
- no deployment ownership

## QuEL-3 deploy input model

No generic realization classes are introduced in this phase.

Use one minimal internal grouping model between planner and backend deploy code:

```python
@dataclass(frozen=True)
class InstrumentDeployRequest:
    port_id: str
    role: InstrumentRole
    frequency_range_min_hz: float
    frequency_range_max_hz: float
    alias: str
    target_labels: tuple[str, ...]
```

Notes:

- This is a QuEL-3-specific boundary model shared only between planner and
  backend deploy code.
- `ChannelRealization` and similar generic realization containers are not part of this phase.

## Mapping rules

### Role mapping

Derive role from `Target.type`:

- `CTRL_GE`, `CTRL_EF`, `CTRL_CR`, `PUMP` -> `InstrumentRole.TRANSMITTER`
- `READ` -> `InstrumentRole.TRANSCEIVER` (v1.5.0 policy)

### Port mapping

Derive `port_id` from target channel binding:

- `target.channel.port` identifies box and port in `ExperimentSystem`
- convert to QuEL-3 port ID format used by quelware (for example `"{unit_label}:trx_p00p01"`)

### Frequency-range mapping

For each selected target, compute:

- `frequency_margin = control_params.frequency_margin[target.type]`
- `frequency_range_min_hz = (target.frequency - frequency_margin) * 1e9`
- `frequency_range_max_hz = (target.frequency + frequency_margin) * 1e9`

Constraints:

- `frequency_margin` is a QuEL-3 deploy-time parameter stored in params data
- validate `frequency_margin < Fs / 2` before deploy to avoid fold-back noise

Use the computed range for:

- `FixedTimelineProfile(frequency_range_min=..., frequency_range_max=...)`

### Alias policy

- generate deterministic alias from `port_id`, role, and target label
- keep stable alias naming across repeated push

## QuEL-3 deployment API contract

Based on `quelware-client`:

```python
profile = FixedTimelineProfile(
    frequency_range_min=request.frequency_range_min_hz,
    frequency_range_max=request.frequency_range_max_hz,
)
definition = InstrumentDefinition(
    alias=request.alias,
    mode=InstrumentMode.FIXED_TIMELINE,
    role=request.role,
    profile=profile,
)
inst_infos = await session.deploy_instruments(
    request.port_id,
    definitions=[definition],
)
```

## Manager contract

```python
class Quel3TargetDeployPlanner(Protocol):
    def build_deploy_requests(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        target_labels: Sequence[str] | None = None,
    ) -> tuple[InstrumentDeployRequest, ...]:
        ...


class Quel3ConfigurationManager(Protocol):
    async def deploy_instruments(
        self,
        *,
        requests: Sequence[InstrumentDeployRequest],
    ) -> dict[str, tuple[InstrumentInfo, ...]]:
        ...
```

Optional delegation:

- `Quel3TargetDeployPlanner` for logical target to deploy-request conversion
- `Quel3ConfigurationManager` for client/session/deploy calls

## Push flow

1. `SystemManager.push(box_ids)` selects QuEL-3 synchronizer.
2. `Quel3SystemSynchronizer.sync_experiment_system_to_hardware(...)` is called
   with the current `experiment_system`.
3. Synchronizer asks the system-side planner for deploy requests.
4. Synchronizer delegates the requests to backend-side
   `Quel3ConfigurationManager`.
5. Configuration manager:
   - deploys instruments via quelware session
   - stores returned `inst_infos` in backend runtime state
6. Push completes.

Capability policy:

- `hardware_push_configure`: supported
- `backend_settings_pull`: unsupported

## Validation and fail-fast rules

- target frequency must be finite
- group range must satisfy `min <= max`
- unsupported `Target.type` to role mapping must fail
- unresolved `port_id` derivation must fail
- role/range incompatibility is validated by quelware deployment and surfaced as error
- duplicated alias within deploy batch must fail

## Migration plan

### Phase 1

- keep `ExperimentSystem` logical API stable
- extract deploy-request planning from deploy execution

### Phase 2

- wire planner plus backend configuration manager in synchronizer
- persist deployed instrument infos for execution lookup

### Phase 3

- remove QuEL-3 synchronizer-side cached `ExperimentSystem` state
- keep pull/snapshot unsupported with explicit error messaging

## Test plan

Unit tests:

- role mapping from `Target.type`
- grouping by `(port_id, role)`
- frequency-range calculation
- deterministic alias generation

Integration tests:

- `load -> push` triggers deploy calls on QuEL-3 path
- deployed `inst_infos` are available for execution manager resolution
- invalid target mapping/range fails with explicit errors

Regression tests:

- existing QuEL-1 push behavior remains unchanged
- existing QuEL-3 execution path remains intact after deployment-stage addition

## Open questions

1. Final canonical conversion from `target.channel.port` to quelware `port_id` string.
2. Final backend runtime-state location for deployed `inst_infos`
   (controller-owned cache vs dedicated runtime context).
