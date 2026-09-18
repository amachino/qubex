# QuEL-3 adapter interface

## Integration boundary

`Quel3MeasurementBackendAdapter` converts measurement schedules into
`Quel3ExecutionPayload` and backend results into `MeasurementResult`.
`Quel3BackendController` coordinates instrument configuration and hardware
operations; its execution manager runs timelines and retrieves results. One
active experiment uses one backend family (`quel1` or `quel3`).

## Target and instrument identity

- A logical target name is its instrument alias. The planner selects a
  unit-qualified port such as `unit-a:trx_p00p01`.
- `Quel3ExecutionPayload.fixed_timelines` is keyed directly by target alias.
- Each target has one instrument. A readout target's transceiver timeline
  contains both waveform events and capture windows.
- Result keys are the same aliases. The adapter converts target names to qubit
  or registry output labels and preserves each target's capture sequence.

## Instrument configuration

`InstrumentSpec` defines one instrument using five fields: `port_id`, `alias`,
`role`, `frequency_range_min_hz`, and `frequency_range_max_hz`.
`InstrumentConfiguration.instruments` groups these specifications.

The controller provides the public configuration workflow:

```python
controller.refresh_instrument_cache(unit_labels=["unit-a"])
configuration = controller.get_instrument_configuration()
path = controller.save_instrument_configuration("instruments.yaml")
loaded = controller.load_instrument_configuration(path)
controller.deploy_instruments(configuration=loaded)
```

Get and save read the current cache. Load parses configuration data without
hardware access or runtime state changes; deployment is a separate operation.
YAML contains instrument specifications only. Resource IDs and driver
configuration remain runtime data acquired from hardware.

## Instrument cache and acquisition

The controller privately owns one `InstrumentCache` of complete hardware
`InstrumentInfo` objects. It retains resource IDs, ports, profiles, and driver
configuration. Cache indexing removes the matching unit prefix from quelware
aliases and requires unique local aliases across all units.

- Deploy writes the selected definitions per port with `append=False`, then
  reads complete instrument information back from hardware. Only touched ports
  are replaced in the cache. Include every instrument that should remain on a
  touched port.
- `deploy_instrument(instrument=spec)` adds or replaces one alias with
  `append=True` by default and refreshes the entire port. Pass `append=False`
  to replace that port with just this instrument. Append is attempted once.
- `refresh_instrument_cache()` explicitly refreshes all units.
  `unit_labels=[...]` limits replacement to those units; an empty selection
  performs no reads or updates.
- Execution consumes cached resource IDs and driver configuration. Missing
  instruments require deployment or an explicit refresh before execution.
- SystemManager pull updates backend-settings snapshots. These snapshots are
  independent of runtime instrument information.
- A new controller starts empty. Connect reloads all existing instruments from
  hardware; connection or readback failure leaves it disconnected with an empty
  cache. Disconnect clears the cache.
- A failed deploy or readback leaves touched ports uncached; unrelated ports
  remain available. An empty instrument configuration changes nothing.

## Hardware inspection

`get_hardware_state()` and `print_hardware_state()` collect diagnostic
snapshots. Individual read failures appear as issues, so a snapshot can be
partial. Hardware-state snapshots and backend settings are never execution
cache inputs. Inspection and `is_synced()` leave runtime instruments unchanged.

## Sequencer and result contract

| Qubex input or operation | quelware integration |
| --- | --- |
| Target waveform | Register samples with `Sequencer.register_waveform`. |
| Instrument timing | Bind cached driver configuration before export. |
| Waveform event | Place the waveform at its start offset in ns, preserving input order for equal starts. |
| Capture window | Preserve supplied names; order by start offset, length, then input order. Names are unique within each timeline; the adapter generates `{target}:{capture_index}`. |
| Execution | Open cached instrument resource IDs, apply directives, and trigger selected instruments together. |
| Results | Read the driver's result and preserve per-target capture order. |

The adapter validates schedules and builds timing in ns. Sequencer export
handles sample-grid constraints. Capture modes follow shot averaging and time
integration: `RAW_WAVEFORMS`, `AVERAGED_WAVEFORM`, `VALUES_PER_ITER`, or
`AVERAGED_VALUE`. Result metadata includes `sampling_period_ns`.
