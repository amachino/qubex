# Results and records

Qubex returns structured result objects that support plotting and persistence.
In v1.5.0, the experiment layer and measurement layer use different persistence
models.

## Experiment-layer results

- **`MeasureResult`** and **`MultipleMeasureResult`** are returned by
  `Experiment.measure()` and `Experiment.execute()`.
  They keep the existing record-based `save()` API for compatibility.
- **`ExperimentResult`** is returned by higher-level workflows such as
  `Experiment.sweep_parameter()`.
  Save it with `result.save(name, description)` and reload it with
  `exp.load_record(...)`.

```python
sweep = exp.sweep_parameter(
    sequence=lambda amplitude: {
        "Q00": qx.pulse.Rect(duration=64, amplitude=amplitude),
    },
    sweep_range=[0.01, 0.02, 0.03],
)

record = sweep.save("amplitude_sweep", "Rect pulse sweep on Q00")
loaded = exp.load_record(record.file_name)
print(loaded.data)
```

Experiment records are stored under `data/` by default.

## Measurement-layer results

The lower-level `Measurement` APIs return canonical `DataModel` objects:

- **`MeasurementResult`** from `run_measurement()`
- **`SweepMeasurementResult`** from `run_sweep_measurement()`
- **`NDSweepMeasurementResult`** from `run_ndsweep_measurement()`

These objects serialize directly to NetCDF.
`MeasurementResult` provides `save()` / `load()` convenience aliases, while the
sweep result models use `save_netcdf()` / `load_netcdf()`.

```python
single_path = measurement_result.save("data/measurement-result.nc")
restored_single = type(measurement_result).load(single_path)

sweep_path = sweep_result.save_netcdf("data/measurement-sweep.nc")
restored_sweep = type(sweep_result).load_netcdf(sweep_path)
```

Legacy `MeasurementRecord` wrappers remain for compatibility, but new
measurement-layer workflows should use direct NetCDF serialization.

## Notes and metadata

- **`ExperimentNote`**: Key-value store for metadata (`exp.note`).
- **`CalibrationNote`**: Stores calibration parameters and validity windows.

```python
exp.note.put("operator", "lab-user")
exp.note.save()
```
