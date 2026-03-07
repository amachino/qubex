# Qubit and resonator spectroscopy

This tutorial walks through a typical spectroscopy flow for locating
resonator and qubit frequencies.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    qubits=[0],
)

exp.connect()
Q0 = exp.qubit_labels[0]
```

## 2. Sweep resonator frequency and power

Start by mapping the resonator response over frequency and readout power.

```python
resonator_freq_range = np.arange(10.05, 10.55, 0.002)

exp.resonator_spectroscopy(
    Q0,
    frequency_range=resonator_freq_range,
    power_range=np.arange(-60, 5, 5),
)
```

## 3. Refine the resonator frequency

After choosing a readout power, scan the resonator frequency again with that
setting.

```python
exp.scan_resonator_frequencies(
    Q0,
    frequency_range=resonator_freq_range,
    readout_amplitude=10 ** (-40 / 20),  # example: -40 dB
)
```

## 4. Sweep the qubit frequency

Once the resonator frequency is in a good range, sweep the qubit drive
frequency.

```python
exp.scan_qubit_frequencies(
    Q0,
    frequency_range=np.arange(6.5, 8.5, 0.002),
)
```

## 5. Reload updated frequencies

If you update `resonator_frequency` or `qubit_frequency` in the parameter
files, reload the experiment before continuing.

```python
exp.reload()
```
