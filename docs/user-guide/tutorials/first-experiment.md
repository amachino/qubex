# First experiment

This tutorial walks through an end-to-end experiment: setup, measurement, pulse construction, parameter sweep, and persistence.

## 1. Create an experiment

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)
```

## 2. Connect and inspect the system

```python
exp.connect()

exp.tool.print_chip_info()
exp.tool.print_wiring_info(exp.qubits)
exp.tool.print_target_frequencies(exp.qubits)
```

## 3. Validate readout

```python
exp.check_noise()
exp.check_waveform()
```

## 4. Run a basic measurement

```python
waveform = np.full(16, 0.01 + 0.01j)

result = exp.measure(
    sequence={
        exp.qubit_labels[0]: waveform,
        exp.qubit_labels[1]: waveform,
    },
    mode="avg",
    shots=1024,
)

result.plot()
```

`measure()` accepts a simple `numpy` array for each qubit waveform and appends
the readout automatically after the control sequence.

## 5. Build a pulse and sweep a parameter

```python
pulse = qx.pulse.FlatTop(duration=30, amplitude=0.02, tau=10)

sweep = exp.sweep_parameter(
    lambda amp: {exp.qubit_labels[0]: pulse.scaled(amp)},
    sweep_range=np.linspace(0.0, 2.0, 30),
)

sweep.plot(normalize=True)
```

## 6. Save results and metadata

```python
record = sweep.save("amplitude_sweep", "flat-top sweep on Q00")
exp.note.put("operator", "lab-user")
exp.note.save()
```

## Next steps

- Explore more experiments in [Examples](../../examples/index.md)
- Dive into the measurement pipeline: [Basic measurement](../how-to/basic-measurement.md)
- Review configuration requirements: [Configuration](../reference/configuration.md)
