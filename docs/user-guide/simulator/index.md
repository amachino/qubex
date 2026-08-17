# QuantumSimulator

`QuantumSimulator` is the offline entry point for pulse-level Hamiltonian studies.
Use it when you want to model quantum systems, drive them with pulses, and iterate on experiments without connecting to real hardware.

## Who should use QuantumSimulator

- Researchers who want to study pulse-level dynamics without using hardware
- Users exploring model behavior before moving to a real system
- Teams prototyping calibrations and pulse designs offline

## What QuantumSimulator gives you

- Pulse-level Hamiltonian simulation for qubits, resonators, and coupled systems
- Reuse of Qubex pulse objects in offline studies
- A safe path for iterating on calibrations before hardware time is available

## Build target unitaries

Use `QuantumSystem.unitary()` to construct a target in the system's physical
Hilbert space. Keys identify ordered object labels. A hyphen-separated key is a
convenience form for a tuple, so `"Q04-Q01"` and `("Q04", "Q01")` have the
same orientation.

```python
from qxsimulator import gates

target = system.unitary(
    {
        "Q04": "X",
        "Q01": "H",
    }
)
cz_target = system.unitary({"Q04-Q01": "CZ"})
ef_target = system.unitary(
    {"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

String values resolve named static gates. They include the gate names used by
`qubex.clifford`, common one- and two-qubit gates, `ZX90`, `BSWAP`, and
`SQRT_BSWAP`. Build a parameterized gate from a Hermitian generator:

```python
x_rotation = gates.rotation(gates.X, angle)
zx_rotation = gates.rotation(gates.ZX, angle)
exchange_rotation = gates.rotation((gates.XX + gates.YY) / 2, angle)
bswap_rotation = gates.rotation((gates.YY - gates.XX) / 2, angle)
```

`rotation(generator, angle)` evaluates
`exp(-1j * angle * generator / 2)`, so the generator expression determines the
normalization and sign. Operations in one mapping must target disjoint objects;
multiply separately constructed unitaries for a sequence.

A smaller gate is embedded on the first matching physical levels and acts as
identity outside that subspace. Pass `levels` to select different physical
levels explicitly.

## Select the fidelity evaluation space

`process_fidelity()` and `average_gate_fidelity()` project the physical
propagator onto a selected tensor-product subspace before comparison:

- `levels="computational"` is the default and selects levels 0 and 1.
- `levels="full"` evaluates every physical level.
- A mapping overrides named objects while unspecified objects remain in their
  computational subspaces, for example `levels={"Q04": (1, 2)}`.

The target may be supplied directly as the same labeled gate mapping accepted
by `system.unitary()`:

```python
fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04-Q01": "CZ"},
)
ef_fidelity = simulator.average_gate_fidelity(
    controls,
    target_unitary={"Q04": gates.X},
    levels={"Q04": (1, 2)},
)
```

For a mapping target, an explicit level mapping controls both physical
embedding and fidelity evaluation. A `Qobj` target may instead already have
the selected subspace dimensions or the full physical-system dimensions. A
full-system target must preserve the selected evaluation space. Average gate
fidelity counts probability that leaves the selected space as leakage.

## Recommended path

1. Install Qubex: [Installation](../getting-started/installation.md)
2. Learn the shared pulse-sequence model if needed: [Build pulse sequences with PulseSchedule](../pulse-sequences/index.md)
3. Start with curated notebooks: [QuantumSimulator example workflows](examples.md)

You do not need hardware configuration files to begin with `QuantumSimulator` notebooks.

## Choose `Experiment` instead when

- You want to run experiments on real hardware
- You need measurement results and hardware-backed readout
- You want the higher-level workflow around connection, execution, and analysis

See [`Experiment`](../experiment/index.md) for that path.
