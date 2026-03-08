# Basic measurement

This guide shows the v1.5.0 `Measurement` workflow for prepared schedules,
async execution, and NetCDF-based result storage.

For most user workflows, prefer the higher-level `Experiment` APIs.

## When to use this layer

- Use `run_measurement()`, `run_sweep_measurement()`, and
  `run_ndsweep_measurement()` when you need explicit
  `MeasurementSchedule` / `MeasurementConfig` objects.
- Use this layer when you want canonical `MeasurementResult` models that save
  directly to NetCDF.

## 1. Create a measurement session

```python
import asyncio
import numpy as np
import qubex as qx

session = qx.Measurement(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

q0, q1 = session.qubit_labels[:2]
```

## 2. Connect to devices

```python
session.connect()
```

## 3. Run one prepared measurement with `run_measurement`

```python
async def main() -> None:
    with qx.PulseSchedule([q0, q1]) as pulse_schedule:
        pulse_schedule.add(
            q0,
            qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
        )
        pulse_schedule.add(
            q1,
            qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
        )
        pulse_schedule.barrier()

    schedule = session.build_measurement_schedule(
        pulse_schedule=pulse_schedule,
        final_measurement=True,
    )
    config = session.create_measurement_config(
        n_shots=1024,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )

    result = await session.run_measurement(
        schedule=schedule,
        config=config,
    )
    result.plot()
    print(result)

asyncio.run(main())
```

`run_measurement()` returns `MeasurementResult`, the canonical serializable result
model for the measurement layer.

## 4. Sweep one parameter with `run_sweep_measurement`

```python
async def main() -> None:
    config = session.create_measurement_config(
        n_shots=1024,
        shot_averaging=True,
        time_integration=True,
    )

    def build_schedule(amplitude: float):
        with qx.PulseSchedule([q0]) as pulse_schedule:
            pulse_schedule.add(
                q0,
                qx.pulse.Rect(duration=64, amplitude=amplitude),
            )
        return session.build_measurement_schedule(
            pulse_schedule=pulse_schedule,
            final_measurement=True,
        )

    sweep = await session.run_sweep_measurement(
        schedule=build_schedule,
        sweep_values=np.linspace(0.0, 0.08, 21),
        config=config,
    )

    print(sweep.sweep_values)
    print(sweep.data[q0][0].shape)

asyncio.run(main())
```

`SweepMeasurementResult.data` reshapes the collected captures into sweep order,
indexed as `target -> capture_index -> ndarray`.

## 5. Sweep multiple parameters with `run_ndsweep_measurement`

```python
async def main() -> None:
    config = session.create_measurement_config(
        n_shots=512,
        shot_averaging=True,
        time_integration=True,
    )

    def build_schedule(point: dict[str, float | int]):
        with qx.PulseSchedule([q0]) as pulse_schedule:
            pulse_schedule.add(
                q0,
                qx.pulse.Rect(
                    duration=int(point["duration"]),
                    amplitude=float(point["amplitude"]),
                ),
            )
        return session.build_measurement_schedule(
            pulse_schedule=pulse_schedule,
            final_measurement=True,
        )

    ndsweep = await session.run_ndsweep_measurement(
        schedule=build_schedule,
        sweep_points={
            "amplitude": [0.02, 0.04, 0.06],
            "duration": [32, 64, 96],
        },
        sweep_axes=("amplitude", "duration"),
        config=config,
    )

    print(ndsweep.shape)
    print(ndsweep.get_sweep_point((1, 2)))
    print(ndsweep.get((1, 2)))

asyncio.run(main())
```

Use `get()` to retrieve one point result and `get_sweep_point()` to inspect the
resolved sweep coordinates for an index.

## 6. Save results as NetCDF

All measurement-layer result models inherit from `DataModel` and can be written
to `.nc` files.

```python
path = result.save("data/single-run.nc")
restored_result = type(result).load(path)

sweep_path = sweep.save_netcdf("data/amplitude-sweep.nc")
restored_sweep = type(sweep).load_netcdf(sweep_path)
```

Use `save()` / `load()` for `MeasurementResult`.
Use `save_netcdf()` / `load_netcdf()` for sweep result models such as
`SweepMeasurementResult` and `NDSweepMeasurementResult`.
