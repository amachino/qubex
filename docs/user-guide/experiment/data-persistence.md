# Data persistence

Qubex has two result-persistence paths. Use `ExperimentResult.save()` for
analyzed experiment-level results. Use `SystemManager.save_rawdata()` or
`set_rawdata_dir()` when you need to keep the raw data and metadata produced
during measurement execution.

| Use case | API | Saved object | Default location | Format |
| --- | --- | --- | --- | --- |
| Save an analyzed experiment result | `ExperimentResult.save()` | `ExperimentRecord[ExperimentResult]` | `data/` | jsonpickle JSON |
| Save raw data and metadata for a specific experiment operation | `exp.system_manager.save_rawdata(...)` | `MeasurementResult` | `.rawdata/` | NetCDF4 `.nc` |
| Keep raw-data capture enabled for a session | `exp.system_manager.set_rawdata_dir(...)` | `MeasurementResult` | User-selected directory | NetCDF4 `.nc` |

## Save `ExperimentResult`

Many high-level `Experiment` methods return `ExperimentResult`. This object
contains experiment-oriented target data and helpers such as `plot()` and
`fit()`. Call `save()` on the result when you want to keep that analyzed result
object.

```python
result = exp.obtain_rabi_params(targets=["Q00"], n_shots=1024)

record = result.save(
    name="q00_rabi",
    description="Rabi calibration for Q00 after frequency update.",
)

print(record.file_name)
```

The convenience method creates an `ExperimentRecord`, saves it immediately, and
returns the record. Files are written under `data/` with names like
`YYYYMMDD_<name>_<counter>.json`. The counter is incremented to avoid
overwriting an existing file for the same date and name.

Load the record with `ExperimentRecord.load()` or the `Experiment` convenience
method:

```python
loaded = exp.load_record(record.file_name)
restored_result = loaded.data
```

Use `ExperimentRecord` directly if you need a non-default directory:

```python
from qubex.experiment.models import ExperimentRecord

record = ExperimentRecord(
    data=result,
    name="q00_rabi",
    description="Rabi calibration for Q00 after frequency update.",
)
record.save(data_path="results/experiment-records")
```

### Notes for `ExperimentRecord`

- `ExperimentResult.save()` stores the high-level result object. It does not
  automatically store every intermediate `MeasurementResult` produced while the
  experiment was running.
- The filename uses `name` as provided. Use short, filesystem-safe names without
  spaces or path separators.
- `ExperimentRecord` files are jsonpickle-encoded Python object graphs. Treat
  them as Qubex/Python persistence files, not as a stable interchange schema for
  other tools.
- Only load `ExperimentRecord` files that you trust. jsonpickle decoding can
  recreate Python objects.
- Restoring old records depends on compatible Python packages and import paths
  for the stored classes.

## Save raw `MeasurementResult` files for a specific experiment operation

`MeasurementResult` is the Qubex standard result model for one measurement run.
It stores target-keyed raw data, the `MeasurementConfig`, optional device
configuration, and classifier references.

Use `save_rawdata()` when you want raw data and metadata only for a specific
experiment operation:

```python
with exp.system_manager.save_rawdata(rawdata_dir=".rawdata", tag="q00-rabi"):
    analyzed = exp.obtain_rabi_params(targets=["Q00"], n_shots=1024)
```

Each measurement run inside this context is saved automatically as a
`MeasurementResult`. Files are written as
`.rawdata/q00-rabi/YYYYMMDD_HHMMSS_microseconds.nc`.

Read a saved raw file with the model loader:

```python
from qubex.measurement.models import MeasurementResult

raw_result = MeasurementResult.load(".rawdata/q00-rabi/20260604_101530_123456.nc")
```

The context manager restores the previous raw-data setting when the context
exits, including when an exception is raised.

## Save every measurement in a session

For notebook or lab sessions where every measurement should be captured, set
the raw-data directory once and disable it explicitly when the session is over.

```python
exp.system_manager.set_rawdata_dir(".rawdata/20260604-q00-session")

try:
    result_a = exp.execute(schedule_a, n_shots=1024)
    result_b = exp.measure(sequence_b, n_shots=1024)
finally:
    exp.system_manager.set_rawdata_dir(None)
```

!!! note
    Enabling raw-data capture for a whole session can use a large amount of
    storage depending on the measurement settings. For long-running sessions or
    shared servers, pay attention to available storage and session-level
    organization.

Because `SystemManager` state can be shared by the active experiment stack,
prefer `try` / `finally` when enabling session-wide raw-data capture.

## Save returned `MeasurementResult` objects explicitly

When an API returns a `MeasurementResult` directly, save the return value
explicitly:

```python
from qubex.measurement.models import MeasurementResult

raw_result = await exp.run_measurement(
    schedule=measurement_schedule,
    n_shots=1024,
)

path = raw_result.save("results/q00-run-measurement.nc")
restored = MeasurementResult.load(path)
```

This explicit form is also useful when you want to choose a filename yourself
instead of using timestamped automatic raw-data files.

## `MeasurementResult` structure

The logical structure is:

- `MeasurementResult.data`: `dict[str, list[CaptureData]]`, keyed by target
  label.
- `MeasurementResult.measurement_config`: number of shots, shot interval,
  averaging mode, integration mode, classification mode, and requested return
  items.
- `MeasurementResult.device_config`: optional device or backend snapshot.
- `MeasurementResult.classifier_refs`: optional classifier metadata keyed by
  target.

Each `CaptureData` entry stores:

- `target`
- `config`
- `payload`
- `sampling_period`
- `classifier_ref`

The primary capture array is selected by `config.primary_return_item` and is
available as `capture.data`. Payload fields use these standard shapes:

| Payload field | Shape |
| --- | --- |
| `waveform_series` | `(n_shots, capture_length)` |
| `iq_series` | `(n_shots,)` |
| `state_series` | `(n_shots,)` |
| `averaged_waveform` | `(capture_length,)` |
| `averaged_iq` | scalar |

## NetCDF4 file contract

`MeasurementResult.save()` is an alias of `save_netcdf()`. Files are written as
NetCDF4 with Qubex/qxcore metadata:

- global attribute `format`
- global attribute `format_version`
- global attribute `model_class`
- global attribute `payload_json`

The file is HDF5-based because it uses NetCDF4, but the public contract is the
Qubex `DataModel` loader, not a hand-authored HDF5 group layout. Top-level
arrays may be stored as NetCDF variables. Nested model payloads may be encoded
inside `payload_json`. Use `MeasurementResult.load()` for reliable restoration
instead of depending on internal variable names.

If you inspect files manually with `netCDF4`, open them with complex support:

```python
from netCDF4 import Dataset

with Dataset("result.nc", mode="r", auto_complex=True) as ds:
    print(ds.ncattrs())
```
