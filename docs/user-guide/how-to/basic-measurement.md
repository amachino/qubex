# Basic measurement

This guide walks through a simple measurement workflow using `MeasurementClient`.

## Goal

Run a single measurement, visualize the result, and inspect the underlying data.

## Steps

### 1. Create a measurement client

```python
import numpy as np
from qubex.measurement import MeasurementClient

cli = MeasurementClient(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)
```

### 2. Connect to devices (optional)

```python
# Required for real hardware. Skip in mock mode or offline tests.
cli.connect()
```

### 3. Execute a measurement

```python
control_waveforms = {
    "Q00": np.full(16, 0.03),
    "Q01": np.full(16, 0.03),
}

result = cli.measure(
    control_waveforms,
    mode="avg",
    shots=1024,
)

result.plot()
print(result.data)
```

### 4. Run a single-shot measurement

```python
result = cli.measure(
    control_waveforms,
    mode="single",
    shots=1024,
)

result.plot()
```

## Notes

- `mode="avg"` returns averaged readout traces.
- `mode="single"` returns single-shot data suitable for classification.
- Use `result.save()` to persist measurements for later analysis.
