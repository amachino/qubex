# Configuration

Qubex uses YAML files to describe chip topology, wiring, and control parameters. These files are loaded by `ConfigLoader` and cached in `SystemManager`.

## Directory layout

```
<base>/<chip_id>/
  config/
    chip.yaml
    box.yaml
    wiring.yaml
    skew.yaml
  params/
    <section>.yaml
```

By default, `<base>` is `/home/shared/qubex-config`. You can override paths using `config_dir` and `params_dir`.

## Config files

- **chip.yaml**: Chip metadata (name, qubit count).
- **box.yaml**: Control hardware inventory and addresses.
- **wiring.yaml**: Mux-level wiring and port mapping.
- **skew.yaml**: Timing skew calibration used by the backend controller.

### Minimal examples

```yaml
# chip.yaml
64Q:
  name: "Example Chip"
  n_qubits: 64
```

```yaml
# box.yaml
Q2A:
  name: "Controller"
  type: "quel1-a"
  address: "10.0.x.x"
  adapter: "adapter-name"
```

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
from qubex.backend import ConfigLoader

cfg = ConfigLoader(
    chip_id="64Q",
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

system = cfg.get_experiment_system()
```

## Example files in repository

The repository includes ready-to-read examples under `docs/examples/configuration/`.

### Config examples (`docs/examples/configuration/config`)

- `chip.yaml`
  Defines chip metadata such as chip name and `n_qubits`.
- `box.yaml`
  Defines available control boxes (type, IP address, adapter).
- `wiring.yaml`
  Defines mux-level port assignment (`ctrl`, `read_out`, `read_in`, optional `pump`).

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
