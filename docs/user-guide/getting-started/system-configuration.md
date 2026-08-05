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
    external_devices.yaml  # when using external instruments
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
- `external_devices.yaml` is optional and configures instruments outside the
  QuEL control system, such as a DC voltage source used for JPA bias control.
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

### `external_devices.yaml`

The file mirrors the main system configuration: `devices` declares each
external instrument like `box.yaml`, `wiring` connects muxes to device
outputs like `wiring.yaml`, and `settings` holds the control policy.

```yaml
devices:
  ONS1:
    driver: ons61797
    channels: [1, 2]
    params:
      port: /dev/ttyACM0

wiring:
  - mux: 6
    bias: ONS1-1
  - mux: 7
    bias: ONS1-2

settings:
  ramp:
    rate_v_per_s: 0.1
    step_size_v: 0.01
    wait_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  reset_voltage: 0.0
  overrides:
    - mux: 7
      ramp:
        rate_v_per_s: 0.05
```

- `devices` — `driver` selects the adapter and `channels` lists the device
  outputs. Everything under `params` is driver-specific and validated by
  the selected driver (ONS61797: `port` for serial or `ip_address` for
  network, not both). Qubex limits ONS61797 writes to 0 V through 4 V and
  channels 1 through 16, and requires its independent output mode (`OMD 0`);
  the Qblox driver accepts -4 V through 4 V.
- `wiring` — each entry names a role (`bias`) and references an output as
  `DEVICE-CHANNEL` (`ONS1-2` = channel 2 of device `ONS1`, one-based).
  Only wired outputs can be driven: an unwired mux or an unlisted channel
  raises an error instead of guessing.
- `settings` — the body sets the defaults for every wired mux and
  `overrides` adjusts them per mux. All outputs of one `settings` (role
  `bias` by default, changeable with `role`) must be on the same device.

The idle voltage — where each output remains while DC bias is not in use — is a
calibrated per-mux value: `idle_voltage` in `jpa_params.yaml`, falling back
to `reset_voltage` for uncalibrated muxes. A DC voltage context holds one
device connection across its voltage applications, sweeps, and readbacks,
then ramps back to the idle voltage and closes the connection on exit.

To control a Qblox SPI Rack through a server process that owns its USB
connection, use the `qblox_server` driver. Qubex connects to that server as a
TCP client. Because a single process keeps ownership of the serial device,
this is the recommended configuration when multiple systems use the same
instrument.

```yaml
devices:
  Qblox1:
    driver: qblox_server
    channels: [1, 2]
    params:
      host: "<qblox-backend-host>"
      port: <qblox-backend-port>
      timeout_s: 1200

wiring:
  - mux: 6
    bias: Qblox1-1

settings:
  ramp:
    rate_v_per_s: 0.1
    step_size_v: 0.01
    wait_s: 0.1
  readback:
    tolerance_v: 0.001
    max_attempts: 3
  reset_voltage: 0.0
```

The server names each output `<device name>-<channel>`, so name the device
exactly as the backend reports it (`Qblox1` → `Qblox1-15`); for irregular
names use `device_names: {channel: name}` in `params`. Ramps run as one
server-side sweep — no other client can interleave setpoints, but other
channels may wait until a sweep finishes. Do not expose the unauthenticated
socket outside a trusted network.

The D5a module has no per-channel output switch and its standard bipolar
span is -4 V to 4 V: `idle` and `shutdown` only ramp to their configured
voltages without electrically disconnecting the output, and the reported
voltage is the module's stored setting, not an independent measurement.

The `ramp` values use the backend sweep's own vocabulary and are passed to
it verbatim: `rate_v_per_s` sets the overall speed (duration ≈ |dV| / rate),
`step_size_v` the setpoint spacing, and `wait_s` the minimum dwell per
setpoint. A setpoint succeeds when its readback error is within
`readback.tolerance_v`, retried up to `readback.max_attempts` times.

`apply_voltage()` ramps an enabled output from its current voltage to the
target; an off output raises instead of being switched on implicitly. DC
voltage operations are scoped to a context; exiting it ramps back to the
idle voltage.

```python
with experiment.external_devices.dc_voltage(mux=6) as dc:
    dc.apply_voltage(0.27)
    state = dc.state
```

Applying a voltage requires the output to be on already — nothing switches
it implicitly. Use `reset_dc_voltages()` to initialize selected outputs at
`reset_voltage` (default 0 V), then ramp them to idle or an operating point.
For maintenance or a deliberate safe stop, use `shutdown_dc_voltages()` to
ramp selected outputs back to `reset_voltage` and switch them off when the
device supports it. Normal experiment contexts return to idle instead.
`get_dc_voltage_states()` reads every wired mux on one connection;
`reset_dc_voltages()` brings muxes to their reset voltages with the outputs
on, `bias_dc_voltages()` ramps calibrated muxes to their bias voltages,
`idle_dc_voltages()` ramps them back to idle, and
`shutdown_dc_voltages()` switches them off when supported. Like box operations,
each
takes an optional `muxes` selection (indices or labels; all wired muxes when
omitted), and all writes prompt for confirmation, like a box push.

To bias every wired mux with a calibrated `bias_voltage` outside a temporary
context, use the bulk operation.

```python
experiment.external_devices.bias_dc_voltages()
```

`sweep()` ramps through each supplied target using the same profile.

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
