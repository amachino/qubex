# System configuration

Use `system_id` as the canonical selector for one concrete instrument setup.
A system ties together chip metadata, the backend kind, backend-specific runtime
settings, and the wiring that maps muxes to physical ports.

`chip_id` and `system_id` are user-defined labels. Qubex does not require a
specific naming convention for either of them.

If you manage configuration files yourself, pass both `config_dir` and
`params_dir` explicitly. This keeps the file layout under your control and
avoids relying on legacy path conventions.

When you omit them, Qubex resolves the config root in this order:

1. `QUBEX_CONFIG_ROOT`
2. `~/qubex-config`
3. `/opt/qubex-config`
4. legacy `/home/shared/qubex-config`

If none of these paths exist, Qubex defaults to `~/qubex-config`.

## Recommended directory layout

Keep shared catalogs in one config directory, and keep parameter files in a
directory for the system you want to run.

```text
qubex-config/
  config/
    chip.yaml
    box.yaml
    system.yaml
    wiring.yaml
    skew.yaml  # for QuEL-1/QuBE
  params/
    SYSTEM_A/
      control_frequency.yaml
      readout_frequency.yaml
      control_amplitude.yaml
      readout_amplitude.yaml
      capture_delay.yaml
      jpa_params.yaml
  calibration/
    SYSTEM_A/
      calib_note.json
```

- `config/` stores the shared system catalogs.
- Each file under `params/<system_id>/` stores one parameter family.
- `calibration/<system_id>/calib_note.json` is the default calibration note location.
- `skew.yaml` is optional, but it is typically needed for QuEL-1 skew-calibration workflows.

## Define the shared config files

### `chip.yaml`

Define chip metadata once per chip.

```yaml
CHIP_A:
  name: "Example chip"
  n_qubits: 64
  topology:
    type: square_lattice
    mux_size: 4
```

### `box.yaml`

Register the hardware units that may appear in the wiring.

```yaml
BOX_A:
  name: "quel3-02-a01"
  type: "quel3"

BOX_B:
  name: "QuEL-1 #5-01"
  type: "quel1-a"
  address: "10.1.0.73"
  adapter: "500202A50TAAA"
```

For QuEL-3 entries, `address` and `adapter` are optional. For QuBE and QuEL-1
entries, they are required.

### `system.yaml`

Create one entry per runnable setup. Multiple systems may point to the same
`chip_id`.

```yaml
SYSTEM_A:
  chip_id: CHIP_A
  backend: quel3
  quel3:
    endpoint: localhost
    port: 50051
```

- The top-level key is the `system_id`.
- `backend` selects the backend family for this system.
- The backend-specific section uses the same name as `backend`.

### `wiring.yaml`

Key the wiring by the same `system_id` and define one row per mux.

```yaml
SYSTEM_A:
  - mux: 0
    ctrl: [BOX_A:4, BOX_A:2, BOX_A:11, BOX_A:9]
    read_out: BOX_A:1
    read_in: BOX_A:0
  - mux: 1
    ctrl: [BOX_A:16, BOX_A:14, BOX_A:17, BOX_A:15]
    read_out: BOX_A:8
    read_in: BOX_A:7
```

Qubex accepts both `BOX:PORT` and `BOX-PORT` forms in `wiring.yaml`, but using
one style consistently is easier to maintain.

## Define parameter files

The current format is one structured YAML file per parameter family. Each file
uses `meta` for annotations and `data` for the actual values.

```yaml
meta:
  description: Example low-frequency control frequencies
  unit: GHz
data:
  0: 3.000
  1: 3.031
  2: 3.062
```

- Put the file in the `params_dir` that you pass to Qubex.
- For qubit-scoped parameters, keys may be integer indices such as `0` and `1`,
  or labels such as `Q000` and `Q001`.
- When `meta.unit` is set, Qubex converts values into its internal base units:
  `GHz` for frequency-like values and `ns` for time-like values.
- When `meta.default` is set, `None` values in `data` fall back to that default.

Legacy `params.yaml` and `props.yaml` are still supported as compatibility
inputs. When both legacy maps and per-file YAMLs exist, Qubex loads the
per-file YAML first and uses the legacy files only as fallback for missing keys.

## Load the configuration in code

Pass the concrete `system_id`, the shared config directory, and the selected
parameter directory.

```python
import qubex as qx

exp = qx.Experiment(
    system_id="SYSTEM_A",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)
```

You can also load and inspect the same files directly through `ConfigLoader`.

```python
from qubex.system import ConfigLoader

loader = ConfigLoader(
    system_id="SYSTEM_A",
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/SYSTEM_A",
)

system = loader.get_experiment_system()
```

After the files are in place, continue with [Quickstart](quickstart.md).
