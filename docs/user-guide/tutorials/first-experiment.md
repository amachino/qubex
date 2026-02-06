# First experiment

This tutorial walks through an end-to-end experiment: setup, measurement, pulse construction, parameter sweep, and persistence.

## 1. Create an experiment

```python
import numpy as np
import qubex as qx

ex = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
    # mock_mode=True,
)
```

## 2. Connect and inspect the system

```python
ex.connect()

ex.tool.print_chip_info()
ex.tool.print_wiring_info(ex.qubits)
ex.tool.print_target_frequencies(ex.qubits)
```

## 3. Validate readout

```python
ex.check_noise()
ex.check_waveform()
```

## 4. Run a basic measurement

```python
waveform = [0.01 + 0.01j] * 16

result = ex.measure(
    sequence={
        ex.qubit_labels[0]: waveform,
        ex.qubit_labels[1]: waveform,
    },
    mode="avg",
    shots=1024,
)

result.plot()
```

## 5. Build a pulse and sweep a parameter

```python
pulse = qx.pulse.FlatTop(duration=30, amplitude=0.02, tau=10)

sweep = ex.sweep_parameter(
    lambda amp: {ex.qubit_labels[0]: pulse.scaled(amp)},
    sweep_range=np.linspace(0.0, 2.0, 30),
)

sweep.plot(normalize=True)
```

## 6. Save results and metadata

```python
record = sweep.save("amplitude_sweep", "flat-top sweep on Q00")
ex.note.put("operator", "lab-user")
ex.note.save()
```

## Next steps

- Explore more experiments in [Examples](../../examples/index.md)
- Dive into the measurement pipeline: [Basic Measurement](../how-to/basic-measurement.md)
- Review configuration requirements: [Configuration](../reference/configuration.md)
