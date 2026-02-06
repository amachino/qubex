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
    params.yaml      # legacy
    props.yaml       # legacy
    <section>.yaml   # structured (recommended)
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

Qubex supports two formats:

1. **Legacy monolithic files**: `params.yaml` and `props.yaml`.
2. **Structured per-section files**: `params/<section>.yaml` with `meta` and `data` keys.

When a per-section file exists, its `data` overrides entries in the legacy files. If it is absent, Qubex falls back to legacy values.

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

- `qubit_frequency`, `resonator_frequency`
- `control_amplitude`, `readout_amplitude`
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
