# qubex

`qubex` is a library designed for conducting superconducting quantum experiments using QuBE. It offers a user-friendly and straightforward interface, allowing experimenters to easily write their experimental scripts.


## Requirements

- Python 3.9 or higher
- `qube-calib` package installed


## Installation

To install qubex, follow these steps:

1. Clone the repository to your local machine.

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

# Create an experiment object
exp = Experiment(
    config_file="./path/to/your/config.json",
)

# Connect to the QuBE device
exp.connect()

# Run a Rabi experiment
result = experiment.rabi_experiment(
    time_range=np.arange(0, 201, 10),
    amplitudes={
        "Q01": 0.03,
        "Q02": 0.03,
    },
)
```

### Sweep parameters of a pulse sequence

```python
from qubex.experiment import Experiment
from qubex.pulse import PulseSequence, Rect

# Create a pulse object
hpi = Rect(duration=20, amplitude=0.03),
hpi_inv = hpi.inverted()

# Create a parameterized waveform
waveform = lambda x: PulseSequence(
    [
        hpi,
        Blank(duration=x),
        hpi_inv,
    ]
)

# Run a parameter sweep experiment
result = ex1.sweep_parameter(
    sweep_range=np.arange(0, 1000, 100),
    parametric_waveforms={
        "Q08": waveform,
    },
)
