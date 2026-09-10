# Experiment system logical/hardware configuration split design

## Status

- State: `IMPLEMENTED`
- Created: 2026-03-05
- Updated: 2026-09-10
- Related notes:
  - `quel3-configuration-design.md`
  - `quel3-control-system-model-design.md`
  - `system-package-quel1-quel3-boundary.md`

## Purpose

Define the final split between:

- logical model construction in `ExperimentSystem` (`TargetRegistry`)
- hardware configuration at `push()` time in backend-specific components

This note defines QuEL-3 responsibilities from logical target planning to hardware deployment.

## Final design summary

1. `ExperimentSystem` keeps logical ownership only.
2. QuEL-3 hardware deployment is executed during `SystemManager.push(...)`.
3. QuEL-3 planning is implemented by a system-side planner that consumes
   `TargetRegistry`.
4. QuEL-3 deploy execution is implemented by a backend-side
   `Quel3ConfigurationManager`.
5. `InstrumentMode.FIXED_TIMELINE` is fixed in this phase.
6. `backend_settings_pull` collects independent instrument snapshots.
   Runtime instrument acquisition occurs on connect, deploy, and explicit refresh.
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
- `Quel3BackendController` as owner of the shared `InstrumentCache`

Responsibilities:

- planner:
  - convert selected targets to an `InstrumentConfiguration` with one specification per target
  - derive role, port ID, alias, and frequency ranges from logical metadata
- backend configuration manager:
  - create `InstrumentDefinition` with fixed timeline profile
  - call `session.deploy_instruments(...)`
  - send all selected definitions for a port together with `append=False`
  - acquire complete instrument information after deployment
  - replace touched ports in the supplied cache, preserving other ports
- backend controller:
  - own the shared cache and delegate instrument operations to managers

### QuEL-3 execution layer

Owner: execution manager

Responsibilities:

- consume deployed instrument aliases/resources
- run fixed timeline execution and fetch measurement results
- no deployment ownership

## QuEL-3 deploy input model

Use frozen Pydantic models between the planner and controller. Each
`InstrumentSpec` contains five fields:

```python
from qubex.backend.quel3 import InstrumentConfiguration, InstrumentSpec

configuration = InstrumentConfiguration(
    instruments=(
        InstrumentSpec(
            port_id="unit-a:tx_p01",
            alias="Q00",
            role="TRANSMITTER",
            frequency_range_min_hz=4.0e9,
            frequency_range_max_hz=4.2e9,
        ),
    ),
)
```

Notes:

- This is a QuEL-3-specific boundary model shared only between planner and
  backend deploy code.
- `ChannelRealization` and similar generic realization containers are not part of this phase.

## Mapping rules

### Role mapping

Derive role from `Target.type`:

- `CTRL_GE`, `CTRL_EF`, `CTRL_FH`, `CTRL_CR`, `CTRL_2Q`, `PUMP` -> `"TRANSMITTER"`
- `READ` -> `"TRANSCEIVER"`

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

- use the exact target label as instrument alias
- require aliases to be unique across the controller cache
- keep unit decoration in resource and port IDs

## QuEL-3 deployment API contract

Group selected targets by unit-qualified port ID. Build one
`InstrumentDefinition` per specification, converting its `InstrumentRoleName` to the quelware
role value, and deploy the entire port batch:

```python
await session.deploy_instruments(
    port_id,
    definitions=definitions,
    append=False,
)
```

A port is the replacement boundary. Include every target that must remain on a
shared port. The configuration manager reads complete hardware information after
all deploy calls succeed and updates only touched ports in the controller's cache.
It does not cache the
deploy response. Other ports are preserved; an empty configuration is a no-op.

## Manager contract

```python
class Quel3TargetDeployPlanner(Protocol):
    def build_configuration(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        target_labels: Sequence[str] | None = None,
    ) -> InstrumentConfiguration:
        ...


class Quel3BackendController(Protocol):
    def deploy_instruments(
        self,
        *,
        configuration: InstrumentConfiguration,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        ...
```

Optional delegation:

- `Quel3TargetDeployPlanner` for logical targets to instrument configuration
- `Quel3ConfigurationManager` for client/session/deploy calls

## Push flow

1. `SystemManager.push(box_ids)` selects QuEL-3 synchronizer.
2. `Quel3SystemSynchronizer.sync_experiment_system_to_hardware(...)` is called
   with the current `experiment_system`.
3. Synchronizer asks the planner for an `InstrumentConfiguration`.
4. Synchronizer delegates that configuration to `Quel3BackendController`.
5. Controller delegates to the configuration manager, passing its shared cache
   and hardware-state reader. The manager invalidates touched ports and deploys
   the selected definitions.
6. The configuration manager reads complete hardware information through the
   state reader and replaces the touched ports in the supplied `InstrumentCache`.
7. Push reads a pure backend-settings snapshot for manager state. Execution
   subsequently uses the cached instruments without resolver discovery.

Reload/runtime note:

- Current runtime model still assumes one active session per process because
  `SystemManager` remains singleton-managed.
- `Experiment(..., backend_controller=...)` configures the active session at
  startup. Subsequent `exp.configure()` calls rely on the same ambient
  `SystemManager.backend_controller` as long as no other session reconfigures
  that singleton.
- Future refactor target:
  - make `SystemManager` session/experiment-owned
  - remove process-global controller/runtime coupling

Capability policy:

- `hardware_push_configure`: supported
- `backend_settings_pull`: instrument snapshots supported

## Validation and fail-fast rules

- target frequency must be finite
- group range must satisfy `min <= max`
- unsupported `Target.type` to role mapping must fail
- unresolved `port_id` derivation must fail
- role/range incompatibility is validated by quelware deployment and surfaced as error
- duplicated alias within deploy batch must fail

## Configuration persistence

The controller owns the public instrument configuration workflow:

- `refresh_instrument_cache()` acquires complete runtime information from hardware.
- `get_instrument_configuration()` exports specifications from the current cache.
- `save_instrument_configuration(path)` writes those specifications to YAML.
- `load_instrument_configuration(path)` returns specifications without hardware
  access or runtime state changes.
- `deploy_instruments(configuration=...)` explicitly applies specifications and
  acquires complete hardware information for the touched ports.

YAML contains the five specification fields only. Resource IDs and driver
configuration remain in the private runtime cache.

## Test plan

Unit tests:

- role mapping from `Target.type`
- grouping by unit-qualified `port_id`
- frequency-range calculation
- target name equals instrument alias

Integration tests:

- `load -> push` triggers deploy calls on QuEL-3 path
- hardware-read instrument information is available for execution
- invalid target mapping/range fails with explicit errors

Regression tests:

- existing QuEL-1 push behavior remains unchanged
- existing QuEL-3 execution path remains intact after deployment-stage addition

## Cache ownership decision

`Quel3BackendController` privately owns one `InstrumentCache` shared with managers.
The cache retains actual hardware information and resource IDs. Connect,
deployment, and explicit refresh are the acquisition boundaries. Pull and inspection collect
independent diagnostic snapshots, which can be partial and never populate the
runtime cache. Applications access configuration through the controller.
