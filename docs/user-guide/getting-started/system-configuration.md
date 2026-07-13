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
- `ge-ef-fh` resolves three-channel ports as `ge`, `ef`, `fh`; two-channel
  ports use `ge` on channel 0 and share channel 1 between `ef` and `fh`.
- `ge-cr-cr` resolves channels in the order `ge`, `cr`, `cr`.
- Other control ports with fewer channels keep only the leftmost roles.

For `quel1se-riken8`, the AWG profile controls the four profile-dependent
control ports.

- `se8_mxfe1_awg1331` resolves those ports as `1-3-3-1`. With
  `configuration_mode="ge-ef-cr"`, the resolved layouts are
  `ge`, `ge-ef-cr`, `ge-ef-cr`, `ge`.
- `se8_mxfe1_awg2222` resolves those ports as `2-2-2-2`. With
  `configuration_mode="ge-ef-cr"`, each port resolves to `ge-ef`. With
  `configuration_mode="ge-ef-fh"`, each port resolves to shared `ge-ef/fh`.
  With `configuration_mode="ge-cr-cr"`, each port resolves to `ge-cr`.

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
    port_wait:
      1: 0
  BOX_B:
    slot: 1
    wait: 0
    port_wait:
      8: 0
monitor_port: BOX_A-12
reference_port: BOX_A-1
scale:
  BOX_A-1: 0.125
target_port: !!set
  BOX_A-1: null
  BOX_B-8: null
time_to_start: 0
trigger_nport: 10
```

- `box_setting.<box>.slot` defines the coarse timing slot for each box.
- `box_setting.<box>.wait` defines the box-common wait value.
- `box_setting.<box>.port_wait` defines the per-port residual wait value.
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

`exp.tool.update_skew(target, ...)` shifts each measured effective wait by
`target - measured_idx` based on the previous `check_skew(...)` result, then
writes the box-common part to `wait` and measured-port residuals to `port_wait`.
Set `backup=True` when you want to save the previous file as a timestamped
backup such as `skew.yaml.bak.20260520_124900`.

For a full walkthrough, see [QuEL-1 skew adjustment workflow](../../examples/system/quel1_skew_adjustment.md).

## Define parameter files

Put system-specific parameter files in the `params_dir` that you pass to Qubex.
The preferred format is one structured YAML file per parameter family, with
legacy `params.yaml` and `props.yaml` used only as compatibility fallbacks.

For the complete file catalog, source-priority rules, and frequency fallback
rules such as `control_frequency.yaml` taking precedence over
`qubit_frequency.yaml`, see [Parameter files](params-configuration.md).

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
