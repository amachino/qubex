# Build a measurement schedule

This guide shows how to build reusable `PulseSchedule` objects and convert them
into `MeasurementSchedule` objects for `run_measurement()`.

For the end-to-end `run_*` workflow and NetCDF result models, see
[Basic measurement](basic-measurement.md).

## 1. Create a measurement session

```python
import qubex as qx

session = qx.Measurement(
    chip_id="64Q",
    qubits=["Q00", "Q01"],
    config_dir="/path/to/config",
    params_dir="/path/to/params",
)

q0, q1 = session.qubit_labels[:2]
```

## 2. Build and reuse a pulse schedule

```python
with qx.PulseSchedule([q0, q1]) as base:
    base.add(
        q0,
        qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
    )
    base.add(
        q1,
        qx.pulse.Gaussian(duration=64, amplitude=0.05, sigma=16),
    )
    base.barrier()
    base.add(q0, qx.pulse.Blank(duration=32))
    base.add(q1, qx.pulse.Blank(duration=32))

with qx.PulseSchedule([q0, q1]) as schedule:
    schedule.call(base)
    schedule.call(base)

schedule.plot()
```

Use `call()` to reuse a prebuilt sub-sequence without rebuilding each pulse by hand.

## 3. Convert the pulse schedule into a measurement schedule

```python
measurement_schedule = session.build_measurement_schedule(
    pulse_schedule=schedule,
    final_measurement=True,
)
```

`final_measurement=True` appends the readout instructions, so the pulse schedule
itself only has to describe the control sequence.

## 4. Run the prepared schedule

```python
import asyncio


async def main() -> None:
    config = session.create_measurement_config(
        n_shots=1024,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )

    result = await session.run_measurement(
        schedule=measurement_schedule,
        config=config,
    )
    result.plot()


asyncio.run(main())
```

Use the same pattern when you want to hand a `MeasurementSchedule` factory to
`run_sweep_measurement()` or `run_ndsweep_measurement()`.
