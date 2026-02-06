# Save and load results

Qubex provides helpers to persist measurement results and experiment records for reproducibility.

## Save a measurement result

```python
result = cli.measure({"Q00": [0.01 + 0.01j] * 16}, mode="avg", shots=1024)
record = result.save()
print(record.file_name)
```

Measurement records are saved under `.rawdata` by default.

## Load a measurement record

```python
from qubex.measurement.models.measurement_record import MeasurementRecord

record = MeasurementRecord.load(record.file_name)
loaded_result = record.data
```

## Save an experiment result

```python
sweep = exp.sweep_parameter(
    lambda amp: {"Q00": [amp + 0.01j] * 16},
    sweep_range=[0.5, 1.0, 1.5],
)

record = sweep.save("demo_sweep", "basic sweep")
```

Experiment records are saved under `data/` by default.

## Load an experiment record

```python
loaded = exp.load_record(record.file_name)
print(loaded.data)
```

## Persist metadata

```python
exp.note.put("calibration_id", "cal-2025-01")
exp.note.save()
```
