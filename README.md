# qubex

`qubex` is a library designed for conducting superconducting quantum experiments using QuBE. It offers a user-friendly and straightforward interface, allowing experimenters to easily write their experimental scripts.


## Requirements

- Python 3.9 or higher
- `qube-calib` package installed


## Installation

To install qubex, follow these steps:

1. Clone the repository to your machine.

2. Install the package using `pip`:

   ```bash
   pip install -e .
   ```


## Sample usage

### Create a config file

- See [example.json](./qubex/configs/example.json) for an example.

### Run a Rabi experiment

```python
from qubex.experiment import Experiment
%matplotlib inline

# Create an experiment object
exp = Experiment(
    config_file="/path/to/your/config.json",
)

# Connect to the QuBE device
exp.connect()

# Run a Rabi experiment
result = exp.rabi_check()
```

### Sweep parameters of a pulse sequence

```python
import numpy as np
from qubex.experiment import Experiment
from qubex.pulse import Blank, Drag, PulseSequence 
%matplotlib inline

# Create an experiment object
exp = Experiment(
    config_file="/path/to/your/config.json",
)

# Connect to the QuBE device
exp.connect()

# Create a pulse object
U = Drag(duration=20, amplitude=0.03, beta=-0.1)
U_inv = U.shifted(np.pi)

# Check the pulse shape
U.plot()
U_inv.plot()

# Create a parameterized waveform
waveform = lambda x: PulseSequence(
    [
        U,
        Blank(duration=x),
        U_inv,
    ]
)

# Check the sequence shape
waveform(0).plot()
waveform(100).plot()

# Run a parameter sweep experiment
result = exp.sweep_parameter(
    sweep_range=np.arange(0, 1000, 100),
    parametric_waveforms={
        "Qxx": waveform,
    },
)
