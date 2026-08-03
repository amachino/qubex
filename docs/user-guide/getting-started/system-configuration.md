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

Configure each external instrument by its purpose. The following example names
the JPA bias controller, defines the default ramp policy, and overrides the ramp
rate for mux 7. Output channels are one-based.

```yaml
dc_voltage_controllers:
  jpa_bias:
    driver: ons61797

    connection:
      port: /dev/ttyACM0

    voltage_control:
      defaults:
        ramp:
          rate_v_per_s: 0.1
          step_interval_s: 0.1
        shutdown:
          voltage_v: 0.0
        readback:
          tolerance_v: 0.001
          max_attempts: 3

      muxes:
        6:
          channel: 1
        7:
          channel: 2
          ramp:
            rate_v_per_s: 0.05
```

The selected driver interprets `connection`. For an ONS61797 network
connection, use `connection.ip_address` instead of `connection.port`. Do not
specify both. A mux entry inherits omitted voltage-control values from
`defaults`. Every mux must map its channel explicitly: using a mux that is
not listed raises an error instead of guessing a channel, so a voltage can
never reach an unintended output.

To control a Qblox SPI Rack through a server process that owns its USB
connection, use the `qblox_server` driver. Qubex connects to that server as a
TCP client. Because a single process keeps ownership of the serial device,
this is the recommended configuration when multiple systems use the same
instrument.

```yaml
dc_voltage_controllers:
  jpa_bias:
    driver: qblox_server

    connection:
      host: "<qblox-backend-host>"
      port: <qblox-backend-port>
      timeout_s: 1200
      channels:
        1: "<backend-device-name-1>"
        2: "<backend-device-name-2>"

    voltage_control:
      defaults:
        ramp:
          rate_v_per_s: 0.1
          step_interval_s: 0.1
        shutdown:
          voltage_v: 0.0
        readback:
          tolerance_v: 0.001
          max_attempts: 3

      muxes:
        6:
          channel: 1
```

The channel values in `connection.channels` are the device identifiers managed
by the server. Qubex delegates each complete ramp to the server-side sweep
command so another client cannot interleave setpoints within that ramp. The
server processes a sweep synchronously, so operations on other channels may
wait until it finishes. Do not expose an unauthenticated socket outside a
trusted network.

The D5a module has no physical per-channel output switch, and the standard
bipolar span is limited to -4 V through 4 V. With this driver, shutdown means
ramping to `shutdown.voltage_v`; it does not electrically disconnect the
output, and direct `turn_on()` and `turn_off()` calls are unsupported. The
reported voltage is the module's stored output setting, not an independent
voltage measurement.

`ramp.rate_v_per_s` is the voltage change per second and
`ramp.step_interval_s` is the interval between setpoints. Their product is the
maximum voltage change per step. On context exit, Qubex ramps to
`shutdown.voltage_v` and turns the output off when the device supports physical
output switching. A setpoint succeeds when its readback error is within
`readback.tolerance_v`; otherwise Qubex retries up to `readback.max_attempts`
times. Both values can be overridden per mux.

`apply_voltage()` enables the output and ramps from the current voltage to the
target using the resolved mux profile. When the output is initially off, Qubex
sets the configured safe voltage before enabling it. DC voltage operations are
scoped to a context; exiting it ramps back to the safe voltage and, when
supported, turns the output off.

```python
with experiment.dc_voltage_control(mux=6) as dc:
    dc.apply_voltage(0.27)
    state = dc.state
```

`turn_on()` and `turn_off()` control the output for the selected mux without
changing its voltage. By default, the output is turned off when the context exits.

To keep a fixed bias enabled after leaving the context, explicitly disable the
automatic shutdown.

```python
with experiment.dc_voltage_control(mux=6, shutdown_on_exit=False) as dc:
    dc.apply_voltage(0.27)
```

`sweep()` ramps through each supplied target using the same profile. Use
`apply_voltage_immediately()` only when the voltage must be applied without a
ramp.

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
