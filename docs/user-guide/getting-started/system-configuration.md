# System configuration

Use `system_id` as the canonical selector for one concrete instrument setup.
A system ties together chip metadata, the backend kind, backend-specific runtime
settings, and the mapping between readout MUXes and physical ports.

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

Keep shared configuration files in one config directory, and keep
system-specific parameter files in separate directories for the systems you
want to run.

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
      measurement_defaults.yaml
      capture_delay.yaml
      ...
  calibration/
    SYSTEM_A/
      calib_note.json
```

- `config/` stores the shared system configuration files.
- Each file under `params/<system_id>/` stores one parameter family.
- `calibration/<system_id>/calib_note.json` is the default calibration file location.
- `skew.yaml` is optional, but it is required for synchronized experiments that use multiple QuEL-1 control units.

## Define shared configuration files

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

BOX_C:
  name: "QuEL-1 SE R8 #1"
  type: "quel1se-riken8"
  address: "10.1.0.160"
  adapter: "500202A800RAA"
  options:
    - "se8_mxfe1_awg2222"
```

For QuEL-3 entries, `address` and `adapter` are optional. For QuBE and QuEL-1
entries, they are required.

`options` is optional and accepts a list of backend option labels for that box.
Use it when a box needs a non-default hardware profile.

For example, `quel1se-riken8` accepts an AWG profile label such as
`se8_mxfe1_awg1331`, `se8_mxfe1_awg2222`, or `se8_mxfe1_awg3113`. When no AWG
profile is specified, Qubex uses `se8_mxfe1_awg2222`.

### Control Layout Resolution

`configuration_mode` is a priority-ordered request, not a fixed channel-count
guarantee.

- `ge-ef-cr` resolves channels in the order `ge`, `ef`, `cr`.
- `ge-cr-cr` resolves channels in the order `ge`, `cr`, `cr`.
- A control port with fewer channels keeps only the leftmost roles.

For `quel1se-riken8`, the AWG profile controls the four profile-dependent
control ports.

- `se8_mxfe1_awg1331` resolves those ports as `1-3-3-1`. With
  `configuration_mode="ge-ef-cr"`, the resolved layouts are
  `ge`, `ge-ef-cr`, `ge-ef-cr`, `ge`.
- `se8_mxfe1_awg2222` resolves those ports as `2-2-2-2`. With
  `configuration_mode="ge-ef-cr"`, each port resolves to `ge-ef`. With
  `configuration_mode="ge-cr-cr"`, each port resolves to `ge-cr`.

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

For QuEL-1 systems that need skew measurement or clock synchronization, define
`quel1.clock_master`.

```yaml
SYSTEM_B:
  chip_id: CHIP_A
  backend: quel1
  quel1:
    clock_master: 10.0.0.10
```

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

### `skew.yaml`

Use `skew.yaml` for synchronized QuEL-1 or QuBE setups that require inter-box
timing adjustment.

```yaml
box_setting:
  BOX_A:
    slot: 0
    wait: 0
  BOX_B:
    slot: 1
    wait: 0
monitor_port: BOX_A-12
reference_port: BOX_A-1
scale:
  BOX_A-1: 0.125
target_port:
  BOX_A-1: null
  BOX_B-8: null
time_to_start: 0
trigger_nport: 10
```

- `box_setting.<box>.slot` defines the coarse timing slot for each box.
- `box_setting.<box>.wait` defines the fine skew wait value you tune during
  correction.
- `reference_port` selects the reference signal source.
- `monitor_port` and `trigger_nport` define the monitor capture path.
- `target_port` lists the ports included in the skew scan.

After loading the same config through `Experiment`, you can inspect and update
the file with the QuEL-1 skew helpers:

```python
result = exp.tool.check_skew(["BOX_A", "BOX_B"])
exp.tool.update_skew(250, ["BOX_A", "BOX_B"], backup=True)
result = exp.tool.check_skew(["BOX_A", "BOX_B"])
```

`exp.tool.update_skew(...)` overwrites `skew.yaml`. Set `backup=True` when you
want to save the previous file as `skew.yaml.bak`.

For a full walkthrough, see [QuEL-1 skew adjustment workflow](../../examples/system/quel1_skew_adjustment.md).

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
- When string labels are used, keep in mind that the zero-padding width depends
  on the number of qubits on the chip.
- When `meta.unit` is set, Qubex converts values into its internal base units
  (`GHz`, `ns`).
- When `meta.default` is set, `None` values in `data` fall back to that default.

Legacy `params.yaml` and `props.yaml` are still supported as compatibility
inputs. When both legacy maps and per-file YAMLs exist, Qubex loads the
per-file YAML first and uses the legacy files only as fallback for missing keys.

### `measurement_defaults.yaml`

Use `measurement_defaults.yaml` when one system should carry different default
measurement execution or readout timing values.

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

- Put the file directly under `params/<system_id>/`.
- The file is optional. If it is missing, Qubex keeps the built-in defaults.
- `execution.n_shots` and `execution.shot_interval_ns` become the defaults for
  measurement APIs when those arguments are omitted.
- `readout.*` becomes the default timing source for readout pulse generation
  and `ExperimentContext` readout timing when explicit overrides are omitted.
- Explicit API arguments still take precedence over `measurement_defaults.yaml`.
- All time values are in `ns`.

## Load configuration from code

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
