# Skew adjustment

This page documents the current skew-adjustment mechanism in Qubex, including how `skew.yaml` is loaded, which fields are actually consumed, and how to run skew checks.

## Overview

Skew settings are used in two different paths:

1. Runtime timing application path
2. Interactive skew measurement path

### Runtime timing application path

- `skew.yaml` is loaded during session setup (`ExperimentContext` / `MeasurementSessionService`).
- Qubex calls `backend_controller.load_skew_yaml(...)` only if the active backend supports it.
- For QuEL-1, loading is delegated to `qubecalib.sysdb.load_skew_yaml(...)`.
- During execution, sequencer padding is adjusted by:
  - box skew: `sysdb.skew[box_name]`
  - per-port skew: `sysdb.port_skew[box_name][port]`
- `slot` and `time_to_start` are applied to hardware timing fields (`timing_shift`, `displacement`).

### Interactive skew measurement path

- `exp.tool.check_skew(...)` runs a measurement workflow on selected boxes.
- For QuEL-1, it delegates to `run_skew_measurement(...)`, which does:
  1. `Skew.from_yaml(...)`
  2. `skew.system.resync()`
  3. `skew.measure()`
  4. optional `skew.estimate()`
  5. `skew.plot()`
- The helper returns in-memory results (`skew`, `fig`) and shows a plot.
- It does not write updated values back to `skew.yaml`.

## `skew.yaml` fields and current behavior

Two loaders consume different subsets of `skew.yaml`.

### Fields consumed by runtime timing application (`SystemConfigDatabase.load_skew_yaml`)

- `box_setting.<box>.slot` (required): converted to `timing_shift = slot * 16`
- `box_setting.<box>.wait` (required): stored as per-box skew
- `box_setting.<box>.port_wait` (optional): per-port skew map; each value must be non-negative
- `time_to_start` (required): stored as system displacement

### Fields consumed by interactive measurement (`SkewSetting.from_yaml_dict`)

- `reference_port` (required): `<BOX>-<PORT>`
- `monitor_port` (required): `<BOX>-<PORT>`
- `trigger_nport` (required): monitor-box trigger port number
- `target_port` (required): list/tuple/set of `<BOX>-<PORT>` strings
- `scale` (required): map of `<BOX>-<PORT>` to pulse scale
- `repeats` (optional): per-target repeat override
- `rf_switches` (optional): RF switch state map by box/port
- `clockmaster_ip` (required unless passed as function argument)

## Known limitation in current implementation

`qubex init-config` currently generates `target_port` as a YAML mapping (dictionary style), while `SkewSetting.from_yaml_dict` requires list/tuple/set semantics.

Practical impact:

- Runtime timing application can still load `box_setting` and `time_to_start`.
- `check_skew(...)` may fail with `TypeError: target_port must be a list/tuple/set of strings` unless `target_port` is written in set/list/tuple form.

## Recommended `skew.yaml` example for skew check workflow

```yaml
clockmaster_ip: 10.0.0.10

box_setting:
  Q2A:
    slot: 3
    wait: 0
    port_wait:
      4: 0

reference_port: Q2A-4
monitor_port: Q2A-5
trigger_nport: 3
time_to_start: 4

scale:
  Q2A-4: 0.125

target_port: !!set
  Q2A-4: null
```

## How to use

1. Place `skew.yaml` in `<chip_id>/config/skew.yaml`.
2. Initialize an experiment/measurement session normally. Skew is loaded automatically if the backend supports it.
3. Run skew check from an `Experiment` instance:

```python
from qubex import Experiment

exp = Experiment(chip_id="64Qv2")
result = exp.tool.check_skew(["Q2A"], estimate=True)
```

4. Inspect the returned plot and in-memory `result["skew"]`.
5. Manually update `box_setting.wait` / `box_setting.port_wait` in `skew.yaml` if calibration needs to be persisted.

## Backend support

- QuEL-1: supports `load_skew_yaml` and `run_skew_measurement`.
- Other backends: skew loading/check calls are skipped or unavailable unless those optional methods are implemented.
