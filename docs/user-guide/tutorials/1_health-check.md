# Environment health check

This tutorial checks that the experiment environment is ready before running
calibration, characterization, or benchmarking procedures.

## 1. Setup

```python
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0, 1],
    # config_dir="/path/to/config",
    # params_dir="/path/to/params",
)
```

## 2. Connect and recover failed links

Start by connecting to the instruments. If `connect()` reports that a box
failed linkup, relink that box and then connect again.

```python
exp.connect()
```

```python
box_id = "BOX_ID"  # replace with the failed box ID reported by connect()
exp.tool.relinkup_box(box_id)
exp.connect()
```

## 3. Configure LO/NCO settings for the relinked box

If you relinked a box, re-apply the LO/NCO settings only for that box.

```python
exp.configure([box_id])
```

## 4. Check background capture data

Use `check_noise()` to inspect capture data when no intentional signal is
present.

```python
exp.check_noise()
```

## 5. Check reflected waveform placement

Use `check_waveform()` to confirm that the reflected readout waveform falls
inside the configured capture region.

```python
exp.check_waveform()
```

If the reflected waveform is outside the capture region, adjust
`capture_delay.yaml` and run the check again.
