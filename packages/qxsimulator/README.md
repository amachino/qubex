# qxsimulator

`qxsimulator` is the quantum system simulation layer extracted from the Qubex project. It contains quantum system models, control signal definitions, and simulation utilities that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxsimulator` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxsimulator` and provides the full experiment framework (backend, measurement, etc.).
- If you only need simulation utilities, install `qxsimulator` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxsimulator"

# uv
uv pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxsimulator"
```

## Gate targets and fidelity spaces

Build target operations by object label. Named static gates include the Qubex
Clifford names, common one- and two-qubit gates, `ZX90`, `BSWAP`, and
`SQRT_BSWAP`. Build parameterized operations with a Hermitian generator and
`gates.rotation()`.

```python
import numpy as np

from qxsimulator import QuantumSystem, Transmon, gates

q04 = Transmon(label="Q04", dimension=3, frequency=5.0)
q01 = Transmon(label="Q01", dimension=3, frequency=5.2)
system = QuantumSystem(objects=[q04, q01])

cz = system.unitary({"Q04-Q01": "CZ"})
ef_x = system.unitary({"Q04": gates.X}, levels={"Q04": (1, 2)})
exchange = gates.rotation((gates.XX + gates.YY) / 2, np.pi / 2)
```

`QuantumSystem.unitary()` embeds each gate in the full physical Hilbert space
and leaves unselected levels unchanged. Fidelity methods evaluate the
computational subspace by default. Pass `levels="full"` for the complete
physical space or a mapping such as `levels={"Q04": (1, 2)}` to override the
evaluated levels of selected objects. Fidelity methods also accept the labeled
gate mapping directly as `target_unitary`; a level mapping then controls both
target embedding and evaluation.

`gates.rotation(generator, angle)` evaluates
`exp(-1j * angle * generator / 2)`. The generator expression therefore owns
the interaction normalization and sign.

## pyproject.toml

```toml
[project]
dependencies = [
  "qxsimulator @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxsimulator",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxsimulator

uv sync
```
