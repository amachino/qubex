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
- You rely on simulator `Control` interpolation or mutate control segment data
  in place

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
- Replace simulator `Control` interpolation with explicitly sampled waveforms
- Remove hardcoded `2 ns` assumptions from sweeps, plots, and timing utilities

## Installation and environment changes

The `v1.5.0` repository workflow assumes a `uv`-managed environment.
Follow the current [installation guide](../user-guide/getting-started/installation.md)
for the exact supported commands.

At minimum, update these assumptions:

- Python `3.9` is no longer supported. Use Python `3.10` or newer.
- Backend-enabled installs use the `backend` extra.
- In-repository development now assumes `make sync` in a `uv` environment.
- `qxsimulator` no longer installs JAX, Optax, or IPython. JAX and Optax were
  used only by the deprecated `PulseOptimizer`; existing users who temporarily
  retain that API must install those two packages separately. Its IPython
  display integration has been removed.

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
- `ge-ef-fh` means `ge`, then `ef`, then `fh`
- `ge-cr-cr` means `ge`, then `cr`, then `cr`
- control ports with fewer channels keep only the leftmost roles

If your hardware profile changes control-port channel counts, the realized
targets change with it. For example, QuEL-1 SE R8 `se8_mxfe1_awg2222` gives
`2-2-2-2` on the four profile-controlled ports, so
`configuration_mode="ge-ef-cr"` now builds `ge-ef` targets there. If you need
CR targets on those ports, use `configuration_mode="ge-cr-cr"` instead. For
EF/FH workflows on two-channel ports, use `configuration_mode="ge-ef-fh"`; EF
and FH share the second channel.

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

### Update simulator `Control` sampling

Simulator `Control` objects now represent finite-duration, piecewise-constant
signals. The `interpolation` constructor argument and `interpolator` property
have been removed. Use `get_samples()` to evaluate the zero-order-held signal.

```python
# v1.4.x
control = Control(..., interpolation="linear")
samples = control.interpolator(times)

# v1.5.0
control = Control(...)
samples = control.get_samples(times)
```

At an internal segment boundary, `get_samples()` returns the segment beginning
at that boundary. It returns zero before the control starts and after its total
duration. If you relied on linear, cubic, or FIR-like reconstruction, generate
the desired sampled waveform before constructing `Control` and provide the
corresponding segment durations.

`Control` copies `waveform` and `durations` and exposes them as read-only
arrays. Construct a new `Control` instead of modifying these arrays in place.
Every segment duration must be finite and greater than zero; an empty control
may still use empty waveform and duration arrays.

### Update `simulate()` propagation settings

`QuantumSimulator.simulate()` now uses `dt` as its fixed propagation interval.
The final interval may be shorter so evolution ends exactly at the common
control duration. Control segment boundaries and requested output times are not
inserted into this integration grid, so discontinuities that do not coincide
with the fixed grid are resolved only as `dt` is reduced.

The `TIME_STEP` constant has been removed. `simulate()` now declares its
default directly as `dt=0.1`; pass `dt` explicitly when a different fixed
propagation interval is required.

Within each interval, the zero-order-held control amplitude is selected at the
left endpoint. Continuously time-dependent carrier and coupling terms are
evaluated at the interval midpoint. Results for detuned drives or rotating
couplings can therefore differ from the previous left-endpoint propagation.

`Control.frame_shifts` and `Control.final_frame_shift` are logical-frame
metadata and are not applied as physical rotations to states or propagators.
Intermediate shifts from a `PulseSchedule` are already reflected in the phases
of subsequent waveform samples. The per-segment metadata additionally lets
`SimulationResult` interpret the returned trajectory in the changing logical
frame. If `n_samples` is specified, it must be at least 2 so that the initial
and final physical evolution points are both retained. Downsampling occurs only
after the complete fixed-step evolution, so `n_samples` does not change the
simulated final state. Uniformly spaced trajectory indices are selected, which
need not produce exactly uniform physical times when the terminal interval is
shorter than `dt`. If the trajectory already contains at most `n_samples`
points, all points are returned. A zero-duration trajectory contains only its
initial point. If `n_samples` is omitted, every fixed-step integration point is
returned.

### Configure QuTiP solver integration with `options`

The `dt` argument no longer appears in the signatures of the QuTiP-based
`QuantumSimulator.sesolve()`, `mesolve()`, `propagator()`, `gate_fidelity()`,
`create_simulation_parameters()`, and `create_simulation_model()` methods.
Calls that still pass `dt` are accepted for compatibility, emit a
`DeprecationWarning`, and ignore its value. The model time list is now the union
of all `Control` segment boundaries, and control amplitudes use exact
zero-order hold between those boundaries. Continuous drive-frame and coupling
phases remain analytic QuTiP coefficients. This list is exposed as
`SimulationModel.boundary_times` and as the `boundary_times` entry returned by
`create_simulation_parameters()`; the previous generic `times` names are no
longer used.

For `sesolve()` and `mesolve()`, `n_samples` requests exactly that many
uniformly spaced public output times for a positive control duration. Qubex
passes the union of those output times and all control boundaries to QuTiP,
then retains only the requested output trajectory. Thus every zero-order-hold
discontinuity remains a solver checkpoint without forcing the public result
onto the irregular control grid. A zero-duration trajectory contains only its
initial point. If `n_samples` is omitted, all control boundaries are returned
as before.

QuTiP chooses adaptive internal integration steps. Pass solver settings such as
`method`, `rtol`, `atol`, and `max_step` through `options`. When `max_step` is
omitted, Qubex uses half the shortest control segment duration; an explicitly
provided value takes precedence. When `nsteps` is omitted, Qubex allows at
least 2500 internal steps and twice the number required by `max_step` over the
longest solver interval. Qubex otherwise uses QuTiP's defaults, including the
integration method and error tolerances. The `dt` argument remains meaningful
only for `QuantumSimulator.simulate()`.

`QuantumSimulator.propagator()` now returns cumulative propagators at the union
of all `Control` segment boundaries. Use the final list element when only the
complete evolution is needed. Advancing through every boundary also gives each
piecewise-constant discontinuity its own solver interval. For a closed system,
the list contains unitary operators computed in Hilbert space. For a system
with any positive decoherence rate, it contains superoperators computed in
Liouville space. Zero-rate relaxation and dephasing operators are no longer
added to the model. The fidelity methods use the final propagator and accept
either representation. They extract the computational-subspace map by default,
accept `levels="full"` for the complete physical space, and accept a per-object
level mapping for qudit or non-computational subspaces. This avoids the much
larger Liouville-space integration for closed systems.

`gate_fidelity()` is deprecated; use `average_gate_fidelity()` instead. The
deprecated name remains an alias during the compatibility period.
`process_fidelity()` returns the normalized Choi overlap of the extracted
computational-subspace map with the target unitary. Because extraction can
make the map trace-decreasing, `average_gate_fidelity()` counts leakage as
failure and uses
$F_\mathrm{avg}=(dF_\mathrm{pro}+p_\mathrm{surv})/(d+1)$, where
$p_\mathrm{surv}=\operatorname{Tr}[\mathcal{E}_\mathrm{sub}(I)]/d$. For a
trace-preserving map, $p_\mathrm{surv}=1$ and this reduces to QuTiP's standard
average-gate-fidelity relation.

Use `QuantumSystem.unitary()` to construct a target by object label and embed
it in the full physical Hilbert space:

```python
from qxsimulator import gates

target = system.unitary({"Q04-Q01": "CZ"})
fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

Named strings include the existing Qubex Clifford gate names and common static
gates. Build parameterized gates with
`gates.rotation(generator, angle)`, which evaluates
`exp(-1j * angle * generator / 2)`. The `X`, `Y`, `Z`, `XX`, `YY`, `ZZ`, and
`ZX` generators can be combined directly, for example
`gates.rotation((gates.XX + gates.YY) / 2, angle)`. Fidelity methods accept
these labeled gate mappings directly. A `Qobj` fidelity target may have the
selected subspace dimensions or the full physical-system dimensions.

`PulseOptimizer` is deprecated and will be removed in a future release. JAX
and Optax are no longer installed with `qxsimulator` and must be installed
separately to use this compatibility API. Its IPython display integration has
been removed; ordinary simulator imports and workflows load none of these
packages.

### Request propagator trajectories explicitly

`SimulationResult.states` and `SimulationResult.propagators` are lists of
QuTiP `Qobj` instances. `SimulationResult.unitaries` is deprecated; use
`propagators` instead. The deprecated attribute remains an alias for the same
list during the compatibility period.

`SimulationResult.control_frequencies` is also deprecated because one target
may have controls at multiple frequencies. Inspect `SimulationResult.controls`
directly instead. When `frame="drive"` is requested, the result infers the
analysis frame only if the target has exactly one distinct control frequency.
If the target has no controls or multiple tones, pass `frame_frequency`
explicitly in GHz.

`SimulationResult.get_substates()` now returns `list[Qobj]` instead of an
object-dtype NumPy array, matching the documented result model. Bloch-vector
and density-matrix helpers continue to return numeric NumPy arrays, with
`float64` and `complex128` dtypes respectively.
The `frame`, `frame_frequency`, and `apply_frame_shifts` arguments of the
substate extraction methods are keyword-only. Update positional calls to use
explicit argument names.

`SimulationResult` now validates trajectory alignment and system dimensions at
construction. It copies the supplied control, state, and propagator containers,
and stores times as a copied, read-only `float64` array. Times must be finite and
strictly increasing; invalid result objects now raise `ValueError` immediately.
Equality is identity-based, and `repr()` reports trajectory counts without
expanding large arrays or QuTiP objects.

`QuantumSimulator.simulate()` computes propagators by default. Pass
`compute_propagators=False` to retain only its state trajectory.
`QuantumSimulator.sesolve()` and `mesolve()` do not compute propagators by
default. Request them explicitly when both trajectories are required:

```python
result = simulator.sesolve(
    controls,
    compute_propagators=True,
)
```

For `sesolve()`, each propagator is an operator acting on a ket. For
`mesolve()`, each propagator is a superoperator acting on a vectorized density
matrix. Computing a full propagator is more expensive than evolving one state,
especially for `mesolve()`, where the superoperator contains `d ** 4` elements
for Hilbert-space dimension `d`. An empty `propagators` list indicates that the
trajectory was not computed.

States and propagators remain in the simulator's physical rotating frame.
Controls converted from a `PulseSchedule` retain both per-segment
`frame_shifts` and the terminal `final_frame_shift` as coordinate metadata.
`SimulationResult.get_substates()` and the density-matrix and Bloch-vector
helpers apply the accumulated frame shifts at every returned time by default.
Pass `apply_frame_shifts=False` to inspect the raw physical-frame trajectory.
At an internal boundary, the shift of the segment starting at that boundary is
used; from the final boundary onward, the terminal shift is used.

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
