# Basic measurement with `Experiment`

This tutorial shows the basic measurement flow with `Experiment`:

1. Run health checks for readout.
2. Execute `single` and `avg` measurements.
3. Run a custom measurement with `PulseSchedule`.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0, 1],
    # config_dir="/path/to/config",
    # params_dir="/path/to/params",
)

exp.connect()
```

## 2. Check readout health

```python
exp.check_noise()
exp.check_waveform()
```

## 3. Run `single` and `avg` measurements

```python
waveform = np.full(16, 0.01 + 0.01j)
sequence = {
    exp.qubit_labels[0]: waveform,
    exp.qubit_labels[1]: waveform,
}

# Per-shot result (useful for IQ cloud checks)
result_single = exp.measure(sequence=sequence, mode="single", shots=1024)
result_single.plot()

# Averaged result (higher SNR)
result_avg = exp.measure(sequence=sequence, mode="avg", shots=1024)
result_avg.plot()
```

## 4. Measure with a custom `PulseSchedule`

```python
target = exp.qubit_labels[0]

schedule = qx.PulseSchedule()
with schedule as s:
    s.add(target, exp.hpi_pulse[target])  # X90
    s.barrier()

custom_result = exp.execute(
    schedule=schedule,
    mode="avg",
    shots=1024,
    add_last_measurement=True,  # append readout automatically
)
custom_result.plot()
```

This example appends a final readout automatically, but `execute()` also lets
you design readout pulses explicitly and place measurements inside the
sequence. If one target is captured multiple times, `custom_result.data[target]`
returns one capture entry per measurement point.

## Related example

- `docs/examples/experiment/0_basic_usage.ipynb`
