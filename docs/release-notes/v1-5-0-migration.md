# v1.5.0 migration guide

Use this guide when upgrading from `v1.4.8` to `v1.5.0`.
For the high-level summary of what changed, see the
[v1.5.0 release notes](v1-5-0.md).

## Who should read this guide

Read this guide if any of the following apply:

- You run Qubex on real hardware through `Experiment` or `Measurement`
- You maintain configuration files under `box.yaml`, `chip.yaml`, or
  `wiring.yaml`
- You import low-level types from `qubex.backend`
- You use contrib-heavy `Experiment` helpers such as RZX, multipartite
  entanglement, purity benchmarking, or Stark workflows
- You wrote timing-sensitive code that assumed a fixed `2 ns` sampling period

If you only use the basic high-level QuEL-1 workflow through top-level
`qubex` imports, and you do not depend on moved helper APIs or backend-side
imports, the upgrade is usually straightforward.

## At a glance checklist

- Use Python `3.10+`
- Prefer `system_id` over `chip_id`
- Add or validate `system.yaml`
- Move system-side imports from `qubex.backend` to `qubex.system`
- Rename `shots` to `n_shots` and `interval` to `shot_interval`
- Replace moved `Experiment` helper methods with `qubex.contrib` functions
- Remove hardcoded `2 ns` assumptions from sweeps, plots, and timing utilities

## Installation and environment changes

The `v1.5.0` repository workflow assumes a `uv`-managed environment.
Follow the current [installation guide](../user-guide/getting-started/installation.md)
for the exact supported commands.

At minimum, update these assumptions:

- Python `3.9` is no longer supported. Use Python `3.10` or newer.
- Backend-enabled installs use the `backend` extra.
- In-repository development now assumes `make sync` in a `uv` environment.

## Configuration changes

### Move from `chip_id`-first to `system_id`-first loading

In `v1.4.8`, many workflows were effectively single-chip oriented. In
`v1.5.0`, the public configuration model is one `system_id` per runnable
hardware setup.

Old style:

```python
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/64Q/config",
    params_dir="/path/to/64Q/params",
)
```

New style:

```python
import qubex as qx

exp = qx.Experiment(
    system_id="64Q-HF-Q1",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/64Q-HF-Q1",
)
```

`chip_id` still works as a compatibility input in `v1.5.0`, but it is now
deprecated and should not be the long-term selector in updated notebooks.

### Add `system.yaml`

`system.yaml` is now the canonical place to define one runnable system and its
backend family.

```yaml
64Q-HF-Q1:
  chip_id: 64Q
  backend: quel1

144Q-LF-Q3:
  chip_id: 144Q
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
```

Update your configuration with these rules:

- Key `wiring.yaml` by `system_id`, not by chip name alone
- Put backend selection in `system.yaml`
- Treat `config_dir` as the shared config directory and `params_dir` as the
  selected per-system parameter directory

Backend selection is resolved in this order:

1. Explicit `backend_kind` argument
2. `backend` field in `system.yaml`
3. Default `quel1`

If you previously stored backend selection in `chip.yaml`, move that setting to
`system.yaml`. In `v1.5.0`, `chip.yaml` is no longer the authoritative backend
source when `system.yaml` exists.

### Prefer structured parameter files

`v1.5.0` prefers one structured YAML file per parameter family.

```yaml
meta:
  unit: GHz
  description: Example control frequencies
data:
  0: 5.000
  1: 5.125
```

Recommended layout:

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
  params/
    64Q-HF-Q1/
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      measurement_defaults.yaml
```

Legacy `params.yaml` and `props.yaml` are still supported as fallback inputs in
`v1.5.0`. You do not need to migrate every parameter file at once, but new
work should use the structured per-file format.

Use `measurement_defaults.yaml` under `params/<system_id>/` when you want one
system to carry different default values for `n_shots`, `shot_interval`, or
readout timing.

### Recheck `configuration_mode` against control-port channel counts

`configuration_mode` is now interpreted as a priority-ordered channel layout.

- `ge-ef-cr` means `ge`, then `ef`, then `cr`
- `ge-cr-cr` means `ge`, then `cr`, then `cr`
- control ports with fewer channels keep only the leftmost roles

If your hardware profile changes control-port channel counts, the realized
targets change with it. For example, QuEL-1 SE R8 `se8_mxfe1_awg2222` gives
`2-2-2-2` on the four profile-controlled ports, so
`configuration_mode="ge-ef-cr"` now builds `ge-ef` targets there. If you need
CR targets on those ports, use `configuration_mode="ge-cr-cr"` instead.

## API and import changes

### Move system-side imports out of `qubex.backend`

The biggest low-level import change is that system/configuration objects no
longer live in `qubex.backend`.

Update imports like this:

```python
# v1.4.8
from qubex.backend import ConfigLoader, ControlSystem, ExperimentSystem, SystemManager

# v1.5.0
from qubex.system import ConfigLoader, ControlSystem, ExperimentSystem, SystemManager
```

The `qubex.backend` namespace now focuses on backend controller contracts and
concrete backend implementations such as `qubex.backend.quel1` and
`qubex.backend.quel3`.

### Rename common kwargs and properties

These changes are not hard breaks in `v1.5.0`, but they should be migrated now:

| Old usage | New usage |
| --- | --- |
| `shots=` | `n_shots=` |
| `interval=` | `shot_interval=` |
| `exp.linkup()` | `exp.connect()` |
| `exp.device_controller` | `exp.backend_controller` |
| `measurement.qubits` | `measurement.qubit_labels` |

Example:

```python
# v1.4.8
result = exp.measure(sequence=sequence, shots=1024, interval=150 * 1024)

# v1.5.0
result = exp.measure(
    sequence=sequence,
    n_shots=1024,
    shot_interval=150 * 1024,
)
```

### Move contrib-style helper APIs out of `Experiment`

Several specialized helper APIs were removed from `Experiment` as direct
methods and moved to `qubex.contrib`. The old methods now warn and raise
`NotImplementedError`, so you must update direct call sites.

Representative mappings:

| Old usage | New usage |
| --- | --- |
| `exp.rzx(...)` | `qx.contrib.rzx(exp, ...)` |
| `exp.rzx_gate_property(...)` | `qx.contrib.rzx_gate_property(exp, ...)` |
| `exp.measure_cr_crosstalk(...)` | `qx.contrib.measure_cr_crosstalk(exp, ...)` |
| `exp.cr_crosstalk_hamiltonian_tomography(...)` | `qx.contrib.cr_crosstalk_hamiltonian_tomography(exp, ...)` |
| `exp.measure_ghz_state(...)` | `qx.contrib.measure_ghz_state(exp, ...)` |
| `exp.measure_graph_state(...)` | `qx.contrib.measure_graph_state(exp, ...)` |
| `exp.measure_bell_states(...)` | `qx.contrib.measure_bell_states(exp, ...)` |
| `exp.purity_benchmarking(...)` | `qx.contrib.purity_benchmarking(exp, ...)` |
| `exp.interleaved_purity_benchmarking(...)` | `qx.contrib.interleaved_purity_benchmarking(exp, ...)` |
| `exp._stark_t1_experiment(...)` | `qx.contrib.stark_t1_experiment(exp, ...)` |
| `exp._stark_ramsey_experiment(...)` | `qx.contrib.stark_ramsey_experiment(exp, ...)` |
| `exp._simultaneous_measurement_coherence(...)` | `qx.contrib.simultaneous_coherence_measurement(exp, ...)` |

Example:

```python
import qubex as qx

schedule = qx.contrib.rzx(
    exp,
    control_qubit="Q00",
    target_qubit="Q01",
    angle=0.78539816339,
)
```

### Update visualization and result access

`v1.5.0` introduces canonical figure accessors on result models.

Update code like this:

```python
# legacy payload access
fig = result["fig"]
figures = result["figures"]

# v1.5.0 canonical access
fig = result.figure
figures = result.figures
detail = result.get_figure("detail")
```

Also move visualization imports to the new module:

```python
# legacy
from qubex.analysis import visualization as viz

# v1.5.0 canonical
import qubex.visualization as viz
```

Legacy import shims still exist for many model modules, but new code should
prefer `qubex.measurement.models` and `qubex.experiment.models`.

### Avoid deep imports into removed internal modules

Top-level package exports such as `qubex.pulse` and `qubex.simulator` still
work, but many old internal module paths were removed as part of the package
split onto companion packages.

Update imports like this:

```python
# v1.4.8 deep import
from qubex.pulse.library import Rect
from qubex.simulator.quantum_system import QuantumSystem

# v1.5.0 stable import
from qubex.pulse import Rect
from qubex.simulator import QuantumSystem
```

If you are building reusable libraries on top of Qubex internals, consider
importing from the companion packages directly (`qxpulse`, `qxsimulator`,
`qxcore`, `qxvisualizer`) instead of relying on removed internal file layouts.

## Timing and result-model updates

### Stop assuming a fixed `2 ns` sampling period

Key execution paths in `v1.5.0` now resolve timing from the active backend.
Replace hardcoded `2` or `2.0` sampling-period values with backend-derived
values where possible.

Recommended pattern:

```python
import numpy as np

wait_range = exp.util.discretize_time_range(
    np.geomspace(100, 100e3, 51),
    sampling_period=exp.measurement.sampling_period,
)
```

For low-level measurement results, use per-capture sampling metadata instead of
assuming one global constant. This is especially important if you are adapting
scripts for QuEL-3.

### Use canonical measurement models when you adopt async or low-level flows

Synchronous compatibility flows such as `measure()` and `execute()` still
return legacy `MeasureResult` and `MultipleMeasureResult` objects where
expected. New async-first and low-level flows return canonical measurement
models such as `MeasurementResult`, `CaptureData`, and
`SweepMeasurementResult`.

These canonical models support structured persistence:

```python
result = await exp.run_measurement(schedule=schedule, n_shots=1024)
path = result.save("result.nc")
restored = type(result).load(path)
```

## Validation steps

After migrating code and configuration, run a small but real validation set:

1. Create a fresh Python `3.10+` environment and install Qubex with the
   required extras.
2. Load one real system through `Experiment(system_id=..., config_dir=..., params_dir=...)`.
3. Run `exp.connect()` and, if your workflow requires it, `exp.configure()`.
4. Execute one smoke measurement with `measure()` or `execute()`.
5. Execute one timing-sensitive sweep or notebook that previously relied on a
   fixed `2 ns` assumption.
6. Run one contrib workflow if your project uses moved helper APIs.
7. Confirm your project no longer emits migration warnings for `chip_id`,
   `shots`, `interval`, legacy figure payload keys, or old import paths.

## Rollback notes

If you need to roll back:

1. Restore the previous `v1.4.8` environment or reinstall from the
   `v1.4.8` tag.
2. Restore the previous configuration snapshot if you changed file layout or
   introduced `system.yaml`.
3. Revert notebook and script updates that depend on `qubex.system`,
   `qubex.contrib`, or backend-derived timing.

Because `v1.5.0` still accepts several legacy inputs as compatibility paths,
you can often stage the migration gradually: update imports and runtime
selection first, then move parameter files and warning-producing call sites.
