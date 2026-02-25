# Skew adjustment

This page documents the current skew-adjustment mechanism in Qubex, including how `skew.yaml` is loaded, which fields are actually consumed, and how to run skew checks.

## Overview

Skew settings are used in two different paths:

1. Runtime timing application path
2. Interactive skew measurement path

### Runtime timing application path

- `skew.yaml` is loaded during session setup (`ExperimentContext` / `MeasurementSessionService`).
- Qubex calls `backend_controller.load_skew_yaml(...)` only if the active backend supports it.
- For QuEL-1, loading is delegated to `qubecalib.sysdb.load_skew_yaml(...)`.
- During execution, sequencer padding is adjusted by:
  - box skew: `sysdb.skew[box_name]`
  - per-port skew: `sysdb.port_skew[box_name][port]`
- `slot` and `time_to_start` are applied to hardware timing fields (`timing_shift`, `displacement`).

### Interactive skew measurement path

- `exp.tool.check_skew(...)` runs a measurement workflow on selected boxes.
- For QuEL-1, it delegates to `run_skew_measurement(...)`, which does:
  1. `Skew.from_yaml(...)`
  2. `skew.system.resync()`
  3. `skew.measure()`
  4. optional `skew.estimate()`
  5. `skew.plot()`
- The helper returns in-memory results (`skew`, `fig`) and shows a plot.
- It does not write updated values back to `skew.yaml`.

## `skew.yaml` fields and current behavior

Two loaders consume different subsets of `skew.yaml`.

### Fields consumed by runtime timing application (`SystemConfigDatabase.load_skew_yaml`)

- `box_setting.<box>.slot` (required): converted to `timing_shift = slot * 16`
- `box_setting.<box>.wait` (required): stored as per-box skew
- `box_setting.<box>.port_wait` (optional): per-port skew map; each value must be non-negative
- `time_to_start` (required): stored as system displacement

### Runtime field mapping and units

| YAML field | Stored in `sysdb` | Applied at runtime | Unit |
| --- | --- | --- | --- |
| `box_setting.<box>.wait` | `sysdb.skew[box]` | Added to generator-sequence `padding` | samples (`Sa`) |
| `box_setting.<box>.port_wait.<port>` | `sysdb.port_skew[box][port]` | Added to generator-sequence `padding` | samples (`Sa`) |
| `box_setting.<box>.slot` | `sysdb.timing_shift[box] = slot * 16` | Added to per-box emission reservation time | block count in YAML (`1 slot = 1 block`) |
| `time_to_start` | `sysdb.time_to_start` | Copied to `Quel1System.displacement` and added to base reservation time | system timecounter ticks |

`wait` / `port_wait` are waveform-level offsets.  
`slot` / `time_to_start` are hardware reservation-time offsets.

### Timing units (QuEL-1)

This document uses the following unit relationship:

- `1 sample = 2 ns`
- `1 word = 4 samples = 8 ns`
- `1 block = 16 words = 64 samples = 128 ns`

For current QuEL-1 execution path, `timecounter` tick is handled in the same `8 ns` granularity as one AWG word.

### `timing_shift`, `time_to_start`, `timecounter` unit summary

These three values are in the same reservation-time domain at execution time.

| Name | Runtime location | Unit | QuEL-1 conversion | Meaning |
| --- | --- | --- | --- | --- |
| `timecounter` | `get_current_timecounter()` and `start_wavegen(..., timecounter=...)` | timecounter tick | `1 tick = 8 ns = 1 word` | Absolute hardware time axis used to reserve emission start. |
| `time_to_start` | `Quel1System.displacement` | timecounter tick | `1 tick = 8 ns = 1 word` | Global offset added to `base_time` for all boxes in one action. |
| `timing_shift` | `Quel1System.timing_shift[box]` | timecounter tick | `1 tick = 8 ns = 1 word` | Per-box offset added after global base-time calculation. |

In other words (current QuEL-1):

- `timecounter`, `time_to_start`, and `timing_shift` are all tick values.
- Tick and word are numerically identical (`1 tick == 1 word`).
- `slot` is the YAML-side coarse unit, and `timing_shift = slot * 16`, so `1 slot = 16 ticks = 16 words = 1 block`.

### Unit guide

- `wait`
  - Meaning: extra leading zero length for all generator targets on that box.
  - Unit: samples (`Sa`), not timecounter ticks.
  - Practical conversion: `wait * sampling_period`.
  - Example (QuEL-1 default `dt=2.0 ns`): `wait=8` means `16 ns` waveform delay.

- `port_wait`
  - Meaning: extra leading zero length for one specific output port.
  - Unit: samples (`Sa`), same as `wait`.
  - Effective waveform delay on one target: `(wait + port_wait) * sampling_period`.
  - Example (`dt=2.0 ns`): `wait=4`, `port_wait=3` means `14 ns` for that port.

- `slot`
  - Meaning: per-box scheduling offset used when reserving emission time.
  - Unit in YAML: block count (`1 slot = 1 block = 128 ns`).
  - Runtime conversion: `timing_shift = slot * 16` (timecounter ticks, equivalently 16 words).
  - Important: this is in the hardware reservation-time domain, not waveform samples.
  - Example: `slot=3` means `3 blocks = 48 words = 384 ns` shift per box.

- `time_to_start`
  - Meaning: global scheduling offset applied to all boxes in one action.
  - Unit: timecounter ticks (same counter domain as `current_timecounter` and `start_wavegen(..., timecounter=...)`).
  - Conversion (QuEL-1): `1 tick = 1 word = 8 ns`.
  - Used as `Quel1System.displacement`, then added to `base_time`.

Do not mix these domains:

- waveform domain: `wait`, `port_wait` (`Sa`)
- reservation-time domain: `slot`, `time_to_start` (timecounter ticks)

## Detailed runtime flow

### 1. Load `skew.yaml` into `sysdb`

Session initialization calls backend optional capability `load_skew_yaml(...)`.
For QuEL-1 this reaches `SystemConfigDatabase.load_skew_yaml`, which stores:

- `timing_shift[box] = slot * 16`
- `skew[box] = wait`
- `port_skew[box][port] = port_wait`
- `time_to_start`

### 2. Apply waveform skew (`wait`, `port_wait`)

When the sequencer is created, each generator target resolves to one physical `(box, port)`.
Then:

- `box_skew = sysdb.skew.get(box, 0)`
- `port_skew = sysdb.port_skew[box].get(port, 0)` (if defined)
- `gss.padding += box_skew + port_skew`

`padding` is sample-count based (`Sa`), so these values are sample offsets.
With QuEL-1 default `dt = 2.0 ns`, `wait = 1` means 2 ns.

### 2.1 What actually changes in waveform sequence

Skew changes are applied in two stages before hardware settings are created:

1. Sequencer construction stage
   - `wait`/`port_wait` are added to each generator target's `GenSampledSequence.padding`.
   - This is where skew enters the waveform path.

2. Converter stage (`generate_e7_settings` -> converter)
   - Converter reads `sequence.padding`.
   - For generation sequence with `padding > 0`, zeros are prepended to the first subsequence:
     - `first_subseq.real = [0...0] + original_real`
     - `first_subseq.imag = [0...0] + original_imag`
   - Result: pulse edge appears later by `padding * dt`.

Important distinction:

- `wait`/`port_wait`: modify waveform sample stream (prepend zeros).
- `slot`/`time_to_start`: do not edit waveform samples; they shift hardware reservation time.

### 2.2 Why sequence-length mismatch usually does not break execution

`wait`/`port_wait` で先頭にゼロを足したぶんは、基本的に後ろ側の blank を
削って吸収します。実装は次の式です。

- `num_blank_words = max(0, interval_words - total_duration_in_words)`

ここで:

- `interval_words`: 設定された繰り返し interval（word単位）
- `total_duration_in_words`: 先頭paddingを含んだ実波形長（word単位）

つまり:

- 後ろblankに余裕があれば、`wait` 分だけ後ろblankが減る
- 余裕がなければ `num_blank_words = 0` でクランプされる

このため、interval は「厳密固定値」ではなく、実装上は下限として扱われます。

補足:

- 変換は hardware ID (`box`, `port`, `channel`) ごとに独立です。
- 同一 hardware ID 内で波形幾何が一致しない場合は `ValueError` で停止します
  （自動で辻褄合わせはしません）。

### 2.3 Cross-port / cross-box timing interpretation

For multi-qubit sequences (different ports and boxes), runtime timing is handled
as "coarse reservation + fine per-target shift":

- Coarse reservation time (box-level):
  - `scheduled_time(box) = base_time + estimated_timediff(box) + timing_shift(box)`
  - `base_time` already includes `time_to_start` (`displacement`).

- Fine target shift (target-level waveform domain):
  - `target_delay_samples = base_padding + wait(box) + port_wait(box, port)`
  - `target_delay_time = target_delay_samples * dt`

So the effective first-pulse timing of target `t` is:

- `effective_start(t) ~= scheduled_time(box_of_t) + target_delay_time(t)`

This means:

- Different `wait`/`port_wait` values across ports are expected and are exactly
  how relative skew correction is applied.
- Runtime does not "re-equalize" these shifts after applying them, because that
  would cancel calibration intent.
- Alignment responsibility is split by domain:
  - reservation-time domain: `time_to_start`, `slot` (`timing_shift`)
  - waveform-time domain: `wait`, `port_wait`

In short, cross-port differences are not an inconsistency by themselves; they
are the correction mechanism.

### 3. Apply reservation-time skew (`slot`, `time_to_start`)

`slot` and `time_to_start` are first copied from `sysdb` into `Quel1System` timing fields:

- `system.timing_shift[box] <- sysdb.timing_shift[box]`
- `system.displacement <- sysdb.time_to_start`

Then in synchronized multi-box emission:

- `base_time = current_time + min_time_offset`
- `base_time += align_offset + displacement + TIMING_OFFSET`
- `scheduled_time(box) = base_time + estimated_timediff[box] + timing_shift[box]`

All terms in the above equations are in the timecounter-tick domain.

This is where `slot` affects each box's start reservation and `time_to_start` shifts all boxes together.

### 4. Notes on where timing fields are copied

The copy from `sysdb` to `Quel1System` happens in driver-side system refresh/build paths
(`refresh_quel1system` / `create_quel1system`) and in sequencer internal system construction paths.
The emission scheduler itself reads from `Quel1System` (`timing_shift`, `displacement`), not directly from YAML.

## Operational cautions

- Keep `wait` and `port_wait` non-negative.
  - `port_wait` is validated as non-negative during YAML load.
  - `wait` should also be treated as non-negative in practice.

- `wait` and `port_wait` are additive.
  - Effective waveform shift for one target is `(box wait + port wait) * dt`.

- Large `wait`/`port_wait` increase leading zeros and can reduce timing margin.
  - They lengthen effective waveform occupancy in the interval budget.

- Do not mix unit domains when tuning.
  - Waveform-domain tuning: `wait`, `port_wait` (`Sa`).
  - Reservation-time tuning: `slot`, `time_to_start` (timecounter ticks / words on current QuEL-1).

### Fields consumed by interactive measurement (`SkewSetting.from_yaml_dict`)

- `reference_port` (required): `<BOX>-<PORT>`
- `monitor_port` (required): `<BOX>-<PORT>`
- `trigger_nport` (required): monitor-box trigger port number
- `target_port` (required): list/tuple/set of `<BOX>-<PORT>` strings
- `scale` (required): map of `<BOX>-<PORT>` to pulse scale
- `repeats` (optional): per-target repeat override
- `rf_switches` (optional): RF switch state map by box/port
- `clockmaster_ip` (required unless passed as function argument)

## Known limitation in current implementation

`qubex init-config` currently generates `target_port` as a YAML mapping (dictionary style), while `SkewSetting.from_yaml_dict` requires list/tuple/set semantics.

Practical impact:

- Runtime timing application can still load `box_setting` and `time_to_start`.
- `check_skew(...)` may fail with `TypeError: target_port must be a list/tuple/set of strings` unless `target_port` is written in set/list/tuple form.

## Recommended `skew.yaml` example for skew check workflow

```yaml
clockmaster_ip: 10.0.0.10

box_setting:
  Q2A:
    slot: 3
    wait: 0
    port_wait:
      4: 0

reference_port: Q2A-4
monitor_port: Q2A-5
trigger_nport: 3
time_to_start: 4

scale:
  Q2A-4: 0.125

target_port: !!set
  Q2A-4: null
```

## How to use

1. Place `skew.yaml` in `<chip_id>/config/skew.yaml`.
2. Initialize an experiment/measurement session normally. Skew is loaded automatically if the backend supports it.
3. Run skew check from an `Experiment` instance:

```python
from qubex import Experiment

exp = Experiment(chip_id="64Qv2")
result = exp.tool.check_skew(["Q2A"], estimate=True)
```

4. Inspect the returned plot and in-memory `result["skew"]`.
5. Manually update `box_setting.wait` / `box_setting.port_wait` in `skew.yaml` if calibration needs to be persisted.

## Backend support

- QuEL-1: supports `load_skew_yaml` and `run_skew_measurement`.
- Other backends: skew loading/check calls are skipped or unavailable unless those optional methods are implemented.
