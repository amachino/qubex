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
ex1 = Experiment(
    config_file="/path/to/your/config.json",
)

# Connect to the QuBE device
ex1.connect()

# Run a Rabi experiment
result1 = ex1.rabi_check()
```

### Sweep parameters of a pulse sequence

```python
import numpy as np
from qubex.experiment import Experiment
from qubex.pulse import Blank, DragCos, PulseSequence 
%matplotlib inline

# Create an experiment object
ex2 = Experiment(
    config_file="/path/to/your/config.json",
)

# Connect to the QuBE device
ex2.connect()

# Create a pulse object
U = DragCos(duration=20, amplitude=0.03, anharmonicity=-400e6)
U_inv = U.inverted()

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
result2 = ex2.sweep_parameter(
    sweep_range=np.arange(0, 1000, 100),
    parametric_waveforms={
        "Qxx": waveform,
    },
)
