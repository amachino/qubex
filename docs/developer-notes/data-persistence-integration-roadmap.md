# Data Persistence Integration Roadmap

## Status

- State: `PROPOSED`
- Last updated: `2026-06-04`
- Scope: design investigation and roadmap only

## Goal

Define how Qubex should evolve data persistence from "files can be saved" to a
lab-scale data-management workflow that supports raw-data capture, later
analysis, provenance, and external catalog tools.

This note follows the user-guide persistence page:

- [Data persistence](../user-guide/experiment/data-persistence.md)

## Current Qubex persistence model

Qubex currently has two persistence paths.

| Path | Primary API | Stored object | Format | Current role |
| --- | --- | --- | --- | --- |
| Analyzed experiment result | `ExperimentResult.save()` | `ExperimentRecord[ExperimentResult]` | jsonpickle JSON | Python/Qubex-side restoration of high-level results |
| Raw measurement result | `MeasurementResult.save()` / `SystemManager.save_rawdata()` / `set_rawdata_dir()` | `MeasurementResult` | NetCDF4 `.nc` | Raw measurement data and metadata preservation |

Important boundaries:

- `ExperimentRecord` is a Python object graph persistence path. It is useful for
  Qubex-side restoration, but it is not a stable external interchange schema.
- `MeasurementResult` is the better integration point for external tools because
  it stores raw array payloads, `MeasurementConfig`, optional `device_config`,
  and classifier references.
- `set_rawdata_dir()` is a state setting. Saving occurs only in execution paths
  that check `SystemManager.rawdata_dir`.
- Today, `execute()` saves the raw `MeasurementResult` before converting it to
  the legacy-compatible return type. Direct-return APIs such as
  `run_measurement()` should also pass through the same persistence policy.

## Labber-like raw-data retention

Many existing labs expect raw data to be saved automatically for almost every
measurement, similar to Labber-style workflows. Qubex should support that
workflow, but it should remain opt-in.

Recommended user-facing position:

- Qubex can support "save raw data by default for this lab/session" workflows.
- The default product behavior should not silently save every measurement for
  every user because raw waveform data can grow quickly with shot count, capture
  length, waveform-series retention, and sweep size.
- The lab-scale mode should be expressed as a session or lab profile, not as an
  unconditional global default.

Expected contract:

```python
with exp.system_manager.save_rawdata(rawdata_dir=".rawdata", tag="q00-rabi"):
    analyzed = exp.obtain_rabi_params(targets=["Q00"], n_shots=1024)
    raw = await exp.run_measurement(schedule=measurement_schedule, n_shots=1024)
```

Within this block, both high-level experiment workflows and direct-return
measurement APIs should save raw `MeasurementResult` objects when those results
are produced by Qubex measurement execution.

Out of scope for automatic saving:

- user-constructed `MeasurementResult` instances
- objects loaded from old files
- derived or post-processed results that are not produced by measurement
  execution

## Recommended implementation boundary

Do not duplicate the current raw-save snippet in each public API. Add one
internal hook in the measurement execution service:

```python
def _save_raw_measurement_result_if_enabled(
    self,
    result: MeasurementResult,
    *,
    operation: str | None = None,
) -> Path | None:
    ...
```

Use it immediately after each Qubex measurement execution path creates a raw
`MeasurementResult`:

- `execute()` after synchronous execution and before return-value conversion
- `run_measurement()` after asynchronous execution and before returning
- sweep execution, once the desired per-point or aggregate-save semantics are
  decided

This keeps filename generation, persistence policy, future manifest updates, and
save-error behavior in one place.

## Tiled integration feasibility

[Tiled](https://github.com/bluesky/tiled) is a reasonable match for Qubex raw
measurement data because it focuses on metadata search, remote array/table
slicing, format conversion, locating/downloading underlying files, and
registering uploaded datasets. Its service model supports standard structure
families such as containers, arrays, tables, and xarray-like datasets.

Qubex should not treat Tiled as a replacement for the current file-level
persistence contract. The safer integration is:

1. Qubex continues to write raw `MeasurementResult` artifacts.
2. A run manifest records metadata and paths.
3. Tiled catalogs those artifacts and exposes arrays through a Qubex-aware
   adapter or an xarray/Zarr export path.

Short-term Tiled integration:

- register `.nc` files as external artifacts
- index run metadata, target labels, experiment operation, timestamp, Qubex
  versions, and file paths
- restore Qubex semantics with `MeasurementResult.load(path)`

Medium-term Tiled integration:

- add a Qubex-aware Tiled adapter that reads `MeasurementResult` and exposes
  target/capture payloads as arrays
- expose metadata separately from large payload arrays
- preserve the Qubex loader as the source of truth for `.nc` semantics

## xarray and Zarr comparison

`xarray` and `Zarr` solve different parts of the problem.

| Topic | xarray | Zarr |
| --- | --- | --- |
| Primary role | Labeled N-dimensional data model | Chunked and compressed array storage format |
| Qubex value | Gives names and coordinates to axes such as target, capture, shot, capture time, and sweep point | Enables scalable storage, partial reads, compression, object-store use, and Tiled-friendly serving |
| Persistence role | Representation and analysis layer; can write via backends | Storage backend |
| Integration risk | Moderate. Requires careful shape and metadata design. | Higher. Requires chunking policy, version choice, dependency policy, and storage hygiene. |
| Best first use | `MeasurementResult.to_xarray()` | optional `save_zarr()` / Tiled backend after xarray semantics stabilize |

References:

- [xarray](https://xarray.dev/) describes itself as labeled N-dimensional arrays
  and datasets in Python, with dimensions, coordinates, and attributes over
  NumPy-like arrays.
- [xarray data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html)
  define `DataArray` as values plus `dims`, `coords`, and `attrs`.
- [xarray Zarr encoding](https://docs.xarray.dev/en/v2026.01.0/internals/zarr-encoding-spec.html)
  documents that xarray reads Zarr stores only when dimension metadata is
  present, using `_ARRAY_DIMENSIONS` for Zarr v2 and `dimension_names` for Zarr
  v3.
- [Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/main/v3/core/)
  defines chunked array storage, codecs, and stores.
- [Zarr-Python 3](https://zarr.dev/blog/zarr-python-3-release/) added full Zarr
  v3 support, but the Python package is documented as Python 3.11+ while Qubex
  currently supports Python 3.10+.

## Recommended xarray shape policy

Start with semantic export, not round-trip persistence.

Recommended first APIs:

```python
dataset = raw_result.to_xarray()
captures = raw_result.to_xarray_by_capture()
```

Initial design rules:

- Use `CaptureData` payload normalization as the source of shape truth.
- Prefer dimension names over positional-axis assumptions.
- Preserve target labels and capture indices as coordinates.
- Add `capture_time_ns` for waveform-domain payloads using each capture's
  `sampling_period`.
- Keep `measurement_config`, `device_config`, classifier references, and Qubex
  version information in attributes or companion metadata.
- Do not force heterogeneous captures into a single rectangular dataset if
  capture length, return item, or payload shape differs.

Possible representation:

```text
MeasurementResult.to_xarray_by_capture()
  (target, capture_index) -> xarray.Dataset

Dataset data variables:
  waveform_series(shot, capture_time)
  iq_series(shot)
  state_series(shot)
  averaged_waveform(capture_time)
  averaged_iq()

Dataset attrs:
  target
  capture_index
  measurement_config
  device_config
  classifier_ref
  qubex_version
```

A single combined `Dataset` can be added later for homogeneous results:

```text
waveform_series(target, capture, shot, capture_time)
iq_series(target, capture, shot)
state_series(target, capture, shot)
averaged_waveform(target, capture, capture_time)
averaged_iq(target, capture)
```

## Recommended Zarr policy

Do not replace NetCDF4 as the default Qubex persistence format yet.

Use Zarr when at least one of the following is true:

- remote slicing through Tiled is a primary use case
- raw waveform-series data is too large for convenient single-file handling
- sweep results need partial reads by sweep point, target, or shot
- object storage or shared storage backends are part of the lab deployment

Introduce Zarr as an optional integration:

```toml
[project.optional-dependencies]
persistence = ["xarray", "zarr", "dask"]
```

or a narrower extra:

```toml
tiled = ["xarray", "zarr"]
```

Do not add `zarr` as a required dependency until the Python-version constraint
and long-term storage contract are settled.

## Run manifest recommendation

The highest-value next feature is a run manifest that links analyzed and raw
artifacts.

Suggested fields:

- `run_id`
- `created_at`
- `operation`
- `tag`
- `targets`
- `chip_id`
- `system_id`
- `measurement_config`
- `device_config`
- `system_state_hash`
- `qubex_version`
- companion package versions
- raw artifact paths
- analyzed `ExperimentRecord` paths
- user note

This is the bridge between Qubex file persistence and catalog tools such as
Tiled.

## Roadmap

### Phase 1: Persistence coverage and policy

- Add a shared raw-result persistence hook in measurement execution.
- Apply it to `execute()` and direct-return `run_measurement()`.
- Keep the mode opt-in through `save_rawdata()` / `set_rawdata_dir()`.
- Add tests for sync and async paths with stubbed execution.

### Phase 2: Run manifest

- Create a manifest writer for raw `MeasurementResult` files.
- Link raw files to analyzed `ExperimentRecord` artifacts when both are saved in
  the same operation or session.
- Include environment and version metadata.
- Add lightweight CLI or helper functions for listing session artifacts.

### Phase 3: xarray semantic export

- Add `CaptureData.to_xarray()` and `MeasurementResult.to_xarray_by_capture()`.
- Add a homogeneous `MeasurementResult.to_xarray()` only when shape rules are
  explicit.
- Keep xarray as an optional dependency.
- Validate target labels, capture indices, dimension names, complex payloads,
  and capture-time coordinates in tests.

### Phase 4: Tiled proof of concept

- Register Qubex raw artifacts and manifests in a local Tiled catalog.
- Decide whether the PoC uses `.nc` files through a Qubex-aware adapter or
  xarray/Zarr export.
- Verify metadata search and array slicing for common lab queries.

### Phase 5: Optional Zarr backend

- Add `save_zarr()` only after xarray semantics are stable.
- Decide Zarr v2 vs v3 based on Tiled compatibility, Python support, and target
  deployment environments.
- Define chunking and compression defaults for waveform-series and sweep data.
- Keep NetCDF4 as the default until lab deployments show that Zarr should become
  a primary backend.

## Non-goals

- Do not make global raw-data persistence the default for all Qubex users.
- Do not expose jsonpickle `ExperimentRecord` as a stable external schema.
- Do not require Tiled, xarray, or Zarr for normal Qubex calibration workflows.
- Do not encode Qubex semantics by relying on internal HDF5 variable names in
  current NetCDF4 files.
