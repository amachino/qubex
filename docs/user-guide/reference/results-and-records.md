# Results and records

Qubex returns structured result objects that support plotting and persistence.

## Measurement results

- **`MeasureResult`**: Primary result type returned by `Experiment.measure()`.
  - Contains per-target data with helpers like `plot()`, `plot_fft()`, and classification utilities.
  - Can be saved via `result.save()`.

- **`MeasurementResult`**: Canonical serializable result for lower-level `Measurement` workflows (advanced usage). Saved to NetCDF with `save()`.

### Save and load a measurement record

```python
record = result.save()
print(record.file_name)

from qubex.measurement.models.measurement_record import MeasurementRecord
loaded = MeasurementRecord.load(record.file_name)
loaded_result = loaded.data
```

Measurement records are stored under `.rawdata/` by default.

## Experiment records

Experiments can store higher-level results such as sweeps or calibrations using `ExperimentRecord`.

```python
record = sweep.save("amplitude_sweep", "flat-top sweep on Q00")
loaded = exp.load_record(record.file_name)
```

Experiment records are stored under `data/` by default.

## Notes and metadata

- **`ExperimentNote`**: Key-value store for metadata (`exp.note`).
- **`CalibrationNote`**: Stores calibration parameters and validity windows.

```python
exp.note.put("operator", "lab-user")
exp.note.save()
```
