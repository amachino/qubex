# QuEL-1 skew adjustment workflow

Use this walkthrough when you need to inspect and retune inter-box skew for a
QuEL-1 or QuBE setup with shared clock synchronization.

This flow assumes:

- your `system.yaml` entry uses the `quel1` backend and defines
  `quel1.clock_master`
- your config directory already contains `box.yaml` and `skew.yaml`
- `skew.yaml` lists the monitor path, reference port, and target boxes

## Minimal setup

Create an `Experiment` with the QuEL-1 system you want to operate.

```python
import qubex as qx

exp = qx.Experiment(
    system_id="64Q-HF-Q1",
    qubits=[0, 1],
    config_dir="/path/to/qubex-config/config",
    params_dir="/path/to/qubex-config/params/64Q-HF-Q1",
)
```

The active `config_dir` should contain a `skew.yaml` similar to this:

```yaml
box_setting:
  S87R:
    slot: 0
    wait: 0
  S89R:
    slot: 1
    wait: 0
monitor_port: S87R-12
reference_port: S87R-1
scale:
  S87R-1: 0.125
target_port:
  S87R-1: null
  S89R-8: null
time_to_start: 0
trigger_nport: 10
```

`box_setting.*.wait` is the value you tune during skew correction.

## Check the current skew

Run `exp.tool.check_skew(...)` for the boxes you want to inspect.

```python
result = exp.tool.check_skew(["S87R", "S89R"])
```

This helper:

- reads `skew.yaml` and `box.yaml` from the active config directory
- uses the reference box from `reference_port`
- runs the current skew measurement flow on the QuEL-1 backend
- returns a `Result` that keeps the measured skew object and plot

For example:

```python
skew_runtime = result["skew"]
figure = result.figure
```

## Update the target skew value

When you want to set one common wait value across a set of boxes, call
`exp.tool.update_skew(...)`.

```python
exp.tool.update_skew(250, ["S87R", "S89R"], backup=True)
```

This helper:

- updates `box_setting.<box>.wait` in `skew.yaml`
- writes a backup file as `skew.yaml.bak` when `backup=True`
- reloads the updated skew file into the active QuEL-1 backend

If `box_ids` is omitted, Qubex updates every box listed under `box_setting`.

```python
exp.tool.update_skew(250, backup=True)
```

## Re-check after the update

Measure again with the updated file.

```python
result = exp.tool.check_skew(["S87R", "S89R"])
```

Inspect the returned figure and repeat the adjustment if needed.

If the skew does not converge, one practical recovery path is to retry with
`wait=0`, check again, and then restore the intended value.

```python
exp.tool.update_skew(0, ["S87R", "S89R"], backup=True)
exp.tool.check_skew(["S87R", "S89R"])

exp.tool.update_skew(250, ["S87R", "S89R"], backup=True)
exp.tool.check_skew(["S87R", "S89R"])
```

## Related pages

- [System configuration](../../user-guide/getting-started/system-configuration.md)
- [`system` example workflows](../../user-guide/system/examples.md)
- [`backend` example workflows](../../user-guide/backend/examples.md)
