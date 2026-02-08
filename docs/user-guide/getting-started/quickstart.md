# Quickstart

This quickstart shows a minimal measurement using the high-level `Experiment` API. It assumes you already have configuration and parameter files for your chip.

> [!NOTE]
> Qubex loads configuration and parameter files that describe your chip, wiring, and control settings.
>
> - Default location: `/home/shared/qubex-config/<chip_id>/`.
> - Custom locations can be passed via `config_dir` and `params_dir` when creating an `Experiment` object.
> See [Configuration](../reference/configuration.md) for details.

## 1. Create an experiment

```python
import qubex as qx

exp = qx.Experiment(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
    # mock_mode=True,  # enable for offline testing
)
```

## 2. Connect to hardware (optional)

```python
# Required for real hardware. Skip in mock mode.
exp.connect()
# exp.configure()  # optional, if you need to push settings to devices
```

## 3. Run a simple measurement

```python
waveform = [0.01 + 0.01j] * 16

result = exp.measure(
    sequence={
        exp.qubit_labels[0]: waveform,
        exp.qubit_labels[1]: waveform,
    },
    mode="avg",
    shots=1024,
)

result.plot()
print(result.data)
```

## 4. Save and reload results

```python
record = result.save()

from qubex.measurement.models.measurement_record import MeasurementRecord
loaded = MeasurementRecord.load(record.file_name)
print(loaded.data)
```

## Next steps

- Learn how configuration and targets are modeled: [Concepts](../concepts/overview.md)
- Execute measurements with lower-level control: [Basic measurement](../how-to/basic-measurement.md)
- Explore notebooks: [Examples](../../examples/index.md)
