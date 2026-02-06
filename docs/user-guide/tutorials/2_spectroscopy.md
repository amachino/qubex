# Qubit and Resonator Spectroscopy

This tutorial covers a practical spectroscopy sequence:

1. Sweep resonator frequency and power.
2. Refine resonator frequency.
3. Sweep qubit frequency.
4. Reload updated frequencies.

## 1. Setup

```python
import numpy as np
import qubex as qx

exp = qx.Experiment(
    chip_id="xxx",
    muxes=[0],
)

exp.connect()
q = exp.qubit_labels[0]
```

## 2. Resonator spectroscopy (frequency vs power)

```python
readout_freq_range = np.arange(10.05, 10.55, 0.002)

exp.resonator_spectroscopy(
    q,
    frequency_range=readout_freq_range,
    power_range=np.arange(-60, 5, 5),
)
```

## 3. Resonator frequency scan with selected readout power

```python
exp.scan_resonator_frequencies(
    q,
    frequency_range=readout_freq_range,
    readout_amplitude=10 ** (-40 / 20),  # example: -40 dB
)
```

## 4. Qubit spectroscopy

```python
exp.scan_qubit_frequencies(
    q,
    frequency_range=np.arange(6.5, 8.5, 0.002),
)
```

After updating `props.yaml` (`resonator_frequency`, `qubit_frequency`), reload:

```python
exp.reload()
```

## Related example

- `docs/examples/experiment/1_spectroscopy.ipynb`
