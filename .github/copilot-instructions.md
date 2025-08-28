# Qubex: Qubit Experiment Framework

Qubex is a Python library for quantum experiments with pulse-level control, readout, and calibration on quantum hardware. It includes quantum simulation capabilities and experimental control for qubit characterization, gate calibration, and benchmarking.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Build Repository

**CRITICAL: Network Issues are Common**
All pip installation commands frequently fail due to PyPI network timeouts. This is a known issue with the dependencies, not your setup. When installations fail, this is EXPECTED behavior.

- Install system dependencies first:
  ```bash
  sudo apt-get update && sudo apt-get install -y libgirepository1.0-dev libcairo2-dev
  ```
  Takes ~65 seconds. NEVER CANCEL - Set timeout to 120+ seconds.

- Install Python dependencies and test tools:
  ```bash
  python -m pip install pip==24.0
  python -m pip install ruff pytest pytest-xdist
  ```
  Takes ~1-4 seconds for each command (if already cached).

- **IMPORTANT**: Fresh installations frequently fail with network timeouts. If working in an existing environment where qubex is already installed, use that installation. The core functionality (pulse and simulation) works without fresh installation.

- Install qubex for development (**Often fails with timeout errors**):
  ```bash
  pip install -e .
  ```
  **WARNING**: This frequently fails due to PyPI timeout issues. Network timeouts are EXPECTED. If this fails, the existing installation in the environment may still be functional.

- Backend installation (**Always fails with network timeouts**):
  ```bash
  pip install .[backend]
  ```
  **WARNING**: This consistently fails due to PyPI timeout issues with the qubecalib dependency. Backend functionality is not available without this installation.

### Running Tests
- Run pulse tests (core functionality, always works):
  ```bash
  pytest tests/pulse/ -v --tb=short --maxfail=1 --durations=10 -n auto
  ```
  Takes ~3 seconds. Always run these tests - they validate core pulse generation.

- Run all tests (requires backend installation):
  ```bash
  pytest -v --tb=short --maxfail=1 --durations=10 -n auto
  ```
  **NOTE**: Will fail without backend dependencies installed. Backend-dependent tests require qubecalib module.

### Linting
- Always run linting before committing:
  ```bash
  ruff check --output-format=github .
  ```
  Takes <1 second. CI will fail if linting errors exist.

## Validation Scenarios

### Core Functionality Testing
**IMPORTANT**: Only test if qubex is already available in the environment. Fresh installations frequently fail.

Always validate qubex installation and basic functionality:

```python
# Test basic import and module availability
import qubex as qx
print([x for x in dir(qx) if not x.startswith('_')])
# Should show: Blank, Pulse, PulseArray, PulseChannel, PulseSchedule, VirtualZ, fit, viz, pulse

# Test pulse creation and manipulation
dt = qx.pulse.get_sampling_period()  # Should return 2.0 ns
drag_pulse = qx.pulse.Drag(duration=20*dt, amplitude=1, beta=0.5)
rect_pulse = qx.pulse.Rect(duration=10*dt, amplitude=0.5)
```

### Quantum Simulation Testing
Test the quantum simulator functionality (works without hardware):

```python
import numpy as np
from qubex.simulator import Control, QuantumSimulator, QuantumSystem, Transmon

# Create a transmon qubit
qubit = Transmon(
    label="Q01", dimension=3, frequency=7.648, anharmonicity=-0.333,
    relaxation_rate=0.00005, dephasing_rate=0.00005
)

# Run simulation
system = QuantumSystem(objects=[qubit])
simulator = QuantumSimulator(system)
drive = qx.pulse.Rect(duration=100, amplitude=2 * (2 * np.pi) / 100)
control = Control(target=qubit, waveform=drive)
result = simulator.mesolve(
    controls=[control], initial_state={"Q01": "0"}, n_samples=11
)
result.show_last_population(qubit.label)  # Should show qubit state populations
```

### Pulse Schedule Testing
Always test pulse scheduling capabilities:

```python
targets = ["Q01", "Q02"]
with qx.PulseSchedule(targets) as sched:
    sched.add("Q01", drag_pulse)
    sched.barrier()
    sched.add("Q02", rect_pulse)

sequences = sched.get_sequences()  # Should return sequences for both targets
```

## Installation Methods

**CRITICAL: All installation methods frequently fail due to PyPI network timeouts. This is expected behavior.**

### Using Existing Environment (Recommended)
If you are working in an environment where qubex is already installed, use that installation:
```python
import qubex as qx  # Test if already available
```
This is the most reliable approach.

### Development Installation (Often fails)
```bash
git clone https://github.com/amachino/qubex.git
cd qubex
pip install -e .
```
**WARNING**: Frequently fails with PyPI timeout errors during dependency installation. This is normal behavior.

### Direct Installation from GitHub (Always fails)
```bash
pip install git+https://github.com/amachino/qubex.git
```
**WARNING**: Consistently fails due to network timeouts when downloading dependencies. Do not use this method.

### Backend Support (Never works)
```bash
pip install .[backend]
# OR
pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```
**WARNING**: Always fails due to network issues with qubecalib dependency. Backend functionality requires hardware setup and is not available in most environments.

## Repository Structure

### Key Directories
- `src/qubex/` - Main package source code
  - `pulse/` - Pulse generation and manipulation
  - `simulator/` - Quantum simulation capabilities
  - `experiment/` - Hardware experiment control (requires backend)
  - `analysis/` - Data analysis and visualization tools
- `tests/` - Test suite (pulse tests always work, experiment tests need backend)
- `docs/examples/` - Jupyter notebook examples
  - `simulator/` - Simulation examples (work without hardware)
  - `experiment/` - Hardware experiment examples (need backend and hardware)
  - `pulse/` - Pulse manipulation tutorials

### Important Files
- `pyproject.toml` - Project configuration, dependencies, and build settings
- `.github/workflows/python-package.yml` - CI/CD pipeline configuration
- `src/qubex/__init__.py` - Main module imports and initialization

## Common Workflows

### Typical Development Workflow
1. Clone repository and install development version
2. Make changes to source code
3. Run pulse tests to validate core functionality
4. Run linting to check code style
5. Test changes with simulation examples
6. Commit and push (CI will run full test suite)

### Experiment Development Workflow (Requires Hardware)
1. Install with backend support (if possible)
2. Configure hardware connections in config files
3. Create experiment scripts using qx.Experiment class
4. Run calibration and characterization experiments
5. Analyze results using qx.fit and qx.viz modules

### Simulation Workflow
1. Define quantum system with Transmon qubits
2. Create pulse sequences using qx.pulse module
3. Run simulations with QuantumSimulator
4. Analyze results with plotting and visualization tools

## Timing Expectations

### Build and Installation
- System dependencies: ~65 seconds
- Core package installation: ~75 seconds
- Test dependencies: ~4 seconds each
- **NEVER CANCEL** these operations - they take time but complete successfully

### Testing
- Pulse tests: ~3 seconds (71 tests)
- Core functionality validation: ~2 seconds
- Simulation tests: ~2 seconds
- Linting: <1 second

### Development Operations
- Import qubex: ~2 seconds (first time), <1 second (subsequent)
- Create basic pulses: <1 second
- Run simple simulations: ~2 seconds

## Known Issues and Limitations

### Network Dependencies
- Backend installation frequently fails due to PyPI timeouts
- Direct GitHub installation may timeout during dependency resolution
- Use development installation (`pip install -e .`) as the reliable method

### Hardware Dependencies
- Full experiment functionality requires quantum hardware and backend installation
- Many tests will fail without backend dependencies (this is expected)
- Simulation and pulse functionality work completely without hardware

### CI/CD
- CI runs on Python 3.9, 3.10, 3.11, and 3.12
- Full test suite requires backend installation
- Linting must pass or CI fails

## Troubleshooting

### Import Errors
- If `from qubex.experiment import Experiment` fails, backend dependencies are missing
- Core pulse and simulation functionality should always work
- Check imports individually: `import qubex.pulse`, `import qubex.simulator`

### Installation Issues
- Use `pip install -e .` instead of GitHub direct installation
- Install system dependencies before Python packages
- Backend dependencies are optional for core functionality

### Test Failures
- Pulse tests should always pass
- Experiment tests require backend installation
- Use `pytest tests/pulse/` to test core functionality only