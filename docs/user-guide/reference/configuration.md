# Configuration

Qubex uses YAML files to describe chip topology, wiring, and control parameters. These files are loaded by `ConfigLoader` and cached in `SystemManager`.

## Directory layout

```
<base>/<chip_id>/
  config/
    chip.yaml
    system.yaml  # backend selection / runtime config
    box.yaml
    wiring.yaml
    skew.yaml
  params/
    <section>.yaml
```

By default, `<base>` is `/home/shared/qubex-config`. You can override paths using `config_dir` and `params_dir`.

## Config files

- **chip.yaml**: Chip metadata (name, qubit count, topology).
- **system.yaml**: Backend selection and runtime settings.
- **box.yaml**: Control hardware inventory and addresses.
- **wiring.yaml**: Mux-level wiring and port mapping.
- **skew.yaml**: Timing skew calibration used by the backend controller.

### Minimal examples

```yaml
# chip.yaml
64Q:
  name: "Example Chip"
  n_qubits: 64
  topology:
    type: square_lattice
    mux_size: 4
```

Backend/runtime settings are managed in `system.yaml`.

```yaml
# system.yaml
schema_version: 1
chip_id: 64Q
backend: quel3

quel1:
  clock_master: 10.0.0.10
```

### Backend selection precedence

When `SystemManager.load(..., backend_kind=...)` omits `backend_kind`, Qubex resolves backend family in this order:

1. explicit `backend_kind` argument
2. `system.yaml` top-level `backend`
3. `chip.yaml` chip entry `backend` (legacy fallback)
4. default `quel1`

`backend` must be either `quel1` or `quel3`.

### Current `system.yaml` runtime behavior (v1.5.0 pre-release)

- `backend` and `chip_id` are used by loader/runtime selection.
- `quel1.clock_master` is used by `ConfigLoader` and overrides legacy `chip.yaml` `clock_master`.
- QuEL-3 endpoint/port/trigger runtime values currently use controller defaults:
  - endpoint: `localhost`
  - port: `50051`
  - trigger wait: `1000000`
  (`quel3:` section in YAML remains reserved for future runtime binding.)

```yaml
# box.yaml
Q2A:
  name: "Controller"
  type: "quel1-a"
  address: "10.0.x.x"
  adapter: "adapter-name"

S159A:
  name: "QuEL-1 SE R8"
  type: "quel1se-riken8"
  address: "10.0.y.y"
  adapter: "adapter-name"
  options:  # optional
    - "se8_mxfe1_awg1331"
```

### Box options (`box.yaml`)

`box.yaml` entries support an optional `options` field:

```yaml
<box_id>:
  ...
  options:
    - "<quel1_config_option>"
```

- `options` is optional. If omitted, behavior is backward-compatible.
- Each value must match a valid `Quel1ConfigOption` string supported by your backend stack.
- For `type: "quel1se-riken8"`, AWG layout options can be selected explicitly:
  - `se8_mxfe1_awg1331`
  - `se8_mxfe1_awg2222`
  - `se8_mxfe1_awg3113`
- For `quel1se-riken8`, if no AWG option is specified, `se8_mxfe1_awg2222` is used by default.
- For `quel1se-riken8`, specifying multiple AWG options at once is invalid and raises an error.

```yaml
# wiring.yaml
64Q:
  - mux: 0
    ctrl: [Q2A-1, Q2A-2, Q2A-3, Q2A-4]
    read_out: Q2A-5
    read_in: Q2A-6
```

## Parameter files

Qubex uses structured per-section files: `params/<section>.yaml` with `meta` and `data` keys.

### Structured format

```yaml
# params/qubit_frequency.yaml
meta:
  unit: GHz
  description: "Qubit transition frequencies"

data:
  Q00: 6.832214
  Q01: 7.532409
```

Supported units include `Hz`, `kHz`, `MHz`, `GHz`, `s`, `ms`, `us`, `ns`. Values are converted into internal base units (GHz for frequency-like quantities, ns for time-like quantities).

## Common parameter sections

- `control_amplitude`, `readout_amplitude`
- `qubit_frequency`, `resonator_frequency`
- `control_frequency`, `readout_frequency`
- `capture_delay`
- `control_vatt`, `readout_vatt`, `pump_vatt`
- `t1`, `t2_star`, `t2_echo`
- `average_readout_fidelity`, `x90_gate_fidelity`, `x180_gate_fidelity`

## Loading configuration programmatically

```python
from qubex.configuration import ConfigLoader

cfg = ConfigLoader(
    chip_id="64Q",
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

system = cfg.get_experiment_system()
```

`from qubex.backend import ConfigLoader` also remains available for compatibility.

If you want deferred loading:

```python
cfg = ConfigLoader(chip_id="64Q", autoload=False)
cfg.load()
system = cfg.get_experiment_system()
```

## Example files in repository

The repository includes ready-to-read examples under `docs/examples/configuration/`.

### Config examples (`docs/examples/configuration/config`)

- `chip.yaml`
  Defines chip metadata such as chip name, `n_qubits`, and topology.
- `system.yaml`
  Defines backend selection and runtime settings.
- `box.yaml`
  Defines available control boxes (type, IP address, adapter).
- `wiring.yaml`
  Defines mux-level port assignment (`ctrl`, `read_out`, `read_in`, optional `pump`).
- `wiring.v2.yaml`
  Defines QuEL-3 style physical wiring with zero-based `qubit_id`/`mux_id` mapped to `port_id`.
  When backend kind resolves to `quel3`, Qubex prefers `wiring.v2.yaml` if present and falls back to `wiring.yaml`.

### Parameter examples (`docs/examples/configuration/params`)

- `control_frequency.yaml`
  Per-qubit control frequencies (`Q00`-style keys, unit: `GHz`).
- `readout_frequency.yaml`
  Per-qubit readout frequencies (`Q00`-style keys, unit: `GHz`).
- `control_amplitude.yaml`
  Per-qubit control amplitudes (dimensionless, so `unit` is blank).
- `readout_amplitude.yaml`
  Per-qubit readout amplitudes (dimensionless, so `unit` is blank).
- `capture_delay.yaml`
  Per-mux capture delay using mux index keys (`0`, `1`, ...).
  Delay unit is 128 ns per count.

These samples are intended as templates. In your own setup, copy the structure
and replace values with your chip-calibrated parameters.
