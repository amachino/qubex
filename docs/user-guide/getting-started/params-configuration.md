# Parameter files

Use the `params/<system_id>/` directory for parameters that belong to one
concrete runnable system. These files are separate from the shared config files
under `config/`, such as `chip.yaml`, `system.yaml`, `wiring.yaml`, and
`skew.yaml`.

Qubex loads parameter files when `Experiment` or `ConfigLoader` loads the
system. If you edit a YAML file after that, create a new `Experiment` or call
`ConfigLoader.load()` again before expecting the change to affect runtime
objects.

## Directory layout

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
    skew.yaml
  params/
    SYSTEM_A/
      qubit_frequency.yaml
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      measurement_defaults.yaml
      capture_delay.yaml
      ...
```

Pass the selected system directory as `params_dir`.

```python
import qubex as qx

exp = qx.Experiment(
    system_id="SYSTEM_A",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)
```

Unknown files under `params/<system_id>/` are ignored unless user code opens
them directly.

## Structured parameter format

Most parameter files use the structured `meta` / `data` format.

```yaml
meta:
  description: Example GE control frequencies
  unit: GHz
  default: null
data:
  0: 4.912
  1: 4.846
  2: null
```

- `meta` is for annotations and loader options.
- `data` contains the values that Qubex loads.
- `meta.unit` converts top-level numeric values into Qubex internal units.
- `meta.default` replaces `null` and YAML NaN values in `data` before unit
  conversion.

The recognized unit conversion families are:

| Unit family | Accepted units | Internal unit |
| --- | --- | --- |
| Frequency | `Hz`, `kHz`, `MHz`, `GHz` | `GHz` |
| Time | `s`, `ms`, `us`, `ns` | `ns` |

Unrecognized units are kept with scale `1.0`, except for backend-specific
`capture_delay` validation. `meta.unit` conversion applies only to top-level
numeric values, so nested maps such as `jpa_params.yaml` should already use the
internal units documented below.

### Keys

Qubit-scoped files accept integer indices, numeric strings, or concrete qubit
labels.

```yaml
data:
  0: 4.912      # normalized to the chip's qubit label
  "1": 4.846    # also normalized
  Q002: 4.901   # accepted as-is
```

For numeric keys, Qubex derives the label width from `chip.yaml:n_qubits`. For
example, a 64-qubit chip uses labels like `Q00`, while a 144-qubit chip uses
labels like `Q000`. String labels are accepted as-is, so they must already match
the labels for the selected chip.

Mux-scoped files use mux indices. Pair-scoped files use the pair labels expected
by the consuming workflow, such as `Q000-Q001`.

## Source priority

The current preferred format is one YAML file per parameter family. Legacy
`params.yaml` and `props.yaml` are still supported as compatibility inputs.

For each recognized parameter:

1. If `<parameter>.yaml` exists and its `data` section is non-empty, Qubex
   loads it.
2. Legacy entries from `params.yaml` or `props.yaml` are used only for keys that
   are missing from the per-file YAML.
3. If the per-file YAML exists but has empty `data`, Qubex falls back to the
   legacy map for that parameter.
4. If the per-file YAML is absent, Qubex uses only the legacy map.

This merge is shallow and key-based. Per-file entries win over legacy entries
for the same key.

## Frequency priority

Some files describe related physical quantities but are not equivalent. Qubex
uses the following runtime priority when building targets.

| Runtime value | Primary source | Fallback source | Used by |
| --- | --- | --- | --- |
| Qubit GE frequency | `control_frequency.yaml` | `qubit_frequency.yaml` | GE targets and CR targets with a target qubit |
| Qubit bare frequency | `qubit_frequency.yaml` | none | `Qubit.bare_frequency` and GE fallback |
| Qubit EF frequency | `control_frequency_ef.yaml` | effective GE frequency + anharmonicity | EF targets |
| Qubit FH frequency | `control_frequency_fh.yaml` | `2 * EF frequency - GE frequency` | FH targets |
| Resonator readout frequency | `readout_frequency.yaml` | `resonator_frequency.yaml` | readout generator and capture targets |
| Resonator ground-state frequency | `resonator_frequency.yaml` | none | `Resonator.frequency_g` and readout fallback |

This means that editing `qubit_frequency.yaml` does not change the GE target
frequency while `control_frequency.yaml` contains a finite value for the same
qubit. Likewise, editing `resonator_frequency.yaml` does not change readout
targets while `readout_frequency.yaml` contains a finite value for the same
resonator.

## Recognized files

These are the parameter files recognized by `ConfigLoader`.

### Quantum and frequency parameters

| File | Scope | Unit | Effect | Legacy source |
| --- | --- | --- | --- | --- |
| `qubit_frequency.yaml` | qubit | frequency, usually `GHz` | Bare qubit frequency and fallback GE frequency. | `props.yaml:<chip_id>.qubit_frequency` |
| `qubit_anharmonicity.yaml` | qubit | frequency, usually `GHz` | Qubit anharmonicity. If absent, the model derives a fallback from the effective GE frequency. | `props.yaml:<chip_id>.anharmonicity` |
| `control_frequency.yaml` | qubit | frequency, usually `GHz` | Preferred GE control frequency. This is the frequency used by GE targets and CR targets with a target qubit. | `props.yaml:<chip_id>.control_frequency` |
| `control_frequency_ef.yaml` | qubit | frequency, usually `GHz` | Preferred EF control frequency. If absent, the model uses GE frequency plus anharmonicity. | `props.yaml:<chip_id>.control_frequency_ef` |
| `control_frequency_fh.yaml` | qubit | frequency, usually `GHz` | Preferred FH control frequency. If absent, the model estimates it as `2 * EF frequency - GE frequency`. | `props.yaml:<chip_id>.control_frequency_fh` |
| `resonator_frequency.yaml` | qubit or resonator-by-qubit | frequency, usually `GHz` | Resonator ground-state frequency and readout fallback. Keys are qubit labels or indices. | `props.yaml:<chip_id>.resonator_frequency` |
| `readout_frequency.yaml` | qubit or resonator-by-qubit | frequency, usually `GHz` | Preferred readout drive and capture frequency. Keys are qubit labels or indices. | `props.yaml:<chip_id>.readout_frequency` |

Example:

```yaml
meta:
  description: Tuned GE control frequencies
  unit: GHz
data:
  0: 4.9123
  1: 4.8461
```

### Control and hardware parameters

| File | Scope | Unit | Effect | Legacy source |
| --- | --- | --- | --- | --- |
| `frequency_margin.yaml` | target type | frequency, usually `GHz` | Overrides target-deployment frequency margins by target type. Supported keys are `READ`, `CTRL_GE`, `CTRL_EF`, `CTRL_FH`, `CTRL_CR`, and `PUMP`. | `params.yaml:<chip_id>.frequency_margin` |
| `control_amplitude.yaml` | qubit | dimensionless | Default control pulse amplitude. | `params.yaml:<chip_id>.control_amplitude` |
| `readout_amplitude.yaml` | qubit | dimensionless | Default readout pulse amplitude. | `params.yaml:<chip_id>.readout_amplitude` |
| `control_vatt.yaml` | qubit | integer or `null` | Control-line VATT setting for backends that support it. | `params.yaml:<chip_id>.control_vatt` |
| `readout_vatt.yaml` | mux | integer or `null` | Readout-line VATT setting for backends that support it. | `params.yaml:<chip_id>.readout_vatt` |
| `pump_vatt.yaml` | mux | integer or `null` | Pump-line VATT setting for backends that support it. | `params.yaml:<chip_id>.pump_vatt` |
| `control_fsc.yaml` | qubit | integer or `null` | Control-line full-scale current setting for backends that support it. | `params.yaml:<chip_id>.control_fsc` |
| `readout_fsc.yaml` | mux | integer or `null` | Readout-line full-scale current setting for backends that support it. | `params.yaml:<chip_id>.readout_fsc` |
| `pump_fsc.yaml` | mux | integer or `null` | Pump-line full-scale current setting for backends that support it. | `params.yaml:<chip_id>.pump_fsc` |
| `capture_delay.yaml` | mux | backend-specific | Capture timing offset. QuEL-1 requires `meta.unit: ndelay`; QuEL-3 requires `meta.unit: ns`. | `params.yaml:<chip_id>.capture_delay` |
| `capture_delay_word.yaml` | mux | `word` or `words` | QuEL-1 capture-delay word offset. This file is not supported for QuEL-3. | `params.yaml:<chip_id>.capture_delay_word` |
| `jpa_params.yaml` | mux | internal units | JPA parameters. Each mux entry may contain `pump_frequency` in `GHz`, `pump_amplitude`, the operating-point `bias_voltage`, and an optional resting-point `idle_voltage`. Missing pump fields are filled from backend defaults; `bias_voltage` has no default — a mux without it is skipped by DC voltage application — and a missing `idle_voltage` falls back to the configured reset voltage. | `params.yaml:<chip_id>.jpa_params` |

QuEL-1 materializes defaults for VATT, FSC, capture delay, and JPA fields.
QuEL-3 materializes `null` for unsupported VATT/FSC/capture-delay-word settings
and uses QuEL-3 defaults for frequency margins and JPA fields.

Examples:

```yaml
# capture_delay.yaml for QuEL-3
meta:
  description: Capture timing offsets
  unit: ns
data:
  0: 8.0
  1: 8.0
```

```yaml
# capture_delay.yaml for QuEL-1
meta:
  description: Capture timing offsets
  unit: ndelay
data:
  0: 7
  1: 8
```

```yaml
# jpa_params.yaml
meta:
  description: JPA pump and bias settings
data:
  0:
    pump_frequency: 6.000
    pump_amplitude: 0.10
    bias_voltage: 0.27
    idle_voltage: -0.05
```

This file holds calibrated values only. `bias_voltage` has no default:
`measurement.apply_dc_voltages()` skips muxes without one, so a system
without this file performs no DC voltage operations. `idle_voltage` is
optional; without it a mux idles at the configured reset voltage. The
control policy lives in `external_devices.yaml` (see the system
configuration guide).

### Measurement defaults

`measurement_defaults.yaml` does not use the `meta` / `data` format. It is a
system-scoped partial default file for measurement execution and readout timing.

```yaml
schema_version: 1

execution:
  n_shots: 2048
  shot_interval_ns: 200000.0

readout:
  duration_ns: 512.0
  ramp_time_ns: 24.0
  pre_margin_ns: 16.0
  post_margin_ns: 96.0
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `1` when present. |
| `execution.n_shots` | Default number of shots when a measurement API uses configured defaults and no explicit argument is provided. Must be positive. |
| `execution.shot_interval_ns` | Default shot interval in `ns` when a measurement API uses configured defaults and no explicit argument is provided. Must be positive. |
| `readout.duration_ns` | Default readout duration in `ns`. Must be non-negative. |
| `readout.ramp_time_ns` | Default readout ramp time in `ns`. Must be non-negative. |
| `readout.pre_margin_ns` | Default pre-readout margin in `ns`. Must be non-negative. |
| `readout.post_margin_ns` | Default post-readout margin in `ns`. Must be non-negative. |

Explicit API arguments take precedence over `measurement_defaults.yaml`. Some
legacy or specialized measurement paths may also define local defaults before
they reach the lower-level measurement configuration, so check the function
signature when a default does not appear to apply.

### Characterization and diagnostic properties

These files are recognized by `ConfigLoader.load_param_data(...)`. The base
system loader does not turn them into target frequencies or hardware settings,
but characterization and workflow code can use them as persisted system
properties.

| File | Scope | Unit | Meaning | Legacy source |
| --- | --- | --- | --- | --- |
| `t1.yaml` | qubit | time, usually `ns` or `us` | T1 estimate. | `props.yaml:<chip_id>.t1` |
| `t2_echo.yaml` | qubit | time, usually `ns` or `us` | Echo T2 estimate. | `props.yaml:<chip_id>.t2_echo` |
| `t2_star.yaml` | qubit | time, usually `ns` or `us` | Ramsey T2* estimate. | `props.yaml:<chip_id>.t2_star` |
| `t2_star_ef.yaml` | qubit | time, usually `ns` or `us` | EF Ramsey T2* estimate. | `props.yaml:<chip_id>.t2_star_ef` |
| `average_readout_fidelity.yaml` | qubit | dimensionless | Average readout fidelity estimate. | `props.yaml:<chip_id>.average_readout_fidelity` |
| `quantum_efficiency.yaml` | qubit | dimensionless | Quantum-efficiency estimate. | `props.yaml:<chip_id>.quantum_efficiency` |
| `x90_gate_fidelity.yaml` | qubit | dimensionless | X90 gate fidelity estimate. | `props.yaml:<chip_id>.x90_gate_fidelity` |
| `x180_gate_fidelity.yaml` | qubit | dimensionless | X180 gate fidelity estimate. | `props.yaml:<chip_id>.x180_gate_fidelity` |
| `zx90_gate_fidelity.yaml` | pair | dimensionless | ZX90 gate fidelity estimate. | `props.yaml:<chip_id>.zx90_gate_fidelity` |
| `static_zz_interaction.yaml` | pair | frequency, usually `GHz` or `MHz` | Static ZZ interaction estimate. | `props.yaml:<chip_id>.static_zz_interaction` |
| `qubit_qubit_coupling_strength.yaml` | pair | frequency, usually `GHz` or `MHz` | Qubit-qubit coupling estimate. | `props.yaml:<chip_id>.qubit_qubit_coupling_strength` |
| `resonator_external_linewidth.yaml` | qubit or resonator-by-qubit | frequency, usually `GHz` or `MHz` | Resonator external linewidth estimate. | `props.yaml:<chip_id>.external_loss_rate` |
| `resonator_internal_linewidth.yaml` | qubit or resonator-by-qubit | frequency, usually `GHz` or `MHz` | Resonator internal linewidth estimate. | `props.yaml:<chip_id>.internal_loss_rate` |

## Files outside `params/`

Do not put shared system files under `params/<system_id>/`.

- `skew.yaml` belongs in the shared `config/` directory. It controls QuEL-1 or
  QuBE inter-box timing and can affect waveform timing, but it is not a
  parameter-family file.
- `calibration/<system_id>/calib_note.json` is the default calibration-note
  location. It is separate from the parameter files described here.

## Common pitfalls

- Expecting `params.yaml` or `props.yaml` to override a per-file YAML. The
  per-file YAML wins for the same key.
- Editing `qubit_frequency.yaml` while `control_frequency.yaml` still defines
  the qubit. GE and CR targets use `control_frequency.yaml` first.
- Editing `resonator_frequency.yaml` while `readout_frequency.yaml` still
  defines the resonator. Readout targets use `readout_frequency.yaml` first.
- Omitting `meta.unit` for `capture_delay.yaml` or using a QuEL-1 unit on a
  QuEL-3 system.
- Editing YAML after an `Experiment` has already loaded and expecting the
  existing object to update automatically.
