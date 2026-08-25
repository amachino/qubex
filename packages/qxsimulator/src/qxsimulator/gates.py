"""Define named one- and two-qubit unitary gates and rotations."""

from __future__ import annotations

import numpy as np
import qutip as qt

__all__ = [
    "BSWAP",
    "CH",
    "CNOT",
    "CX",
    "CY",
    "CZ",
    "II",
    "ISWAP",
    "IX90",
    "IY90",
    "IZ90",
    "SQRT_BSWAP",
    "SQRT_ISWAP",
    "SWAP",
    "SX",
    "X90",
    "XI90",
    "XX",
    "Y90",
    "YI90",
    "YY",
    "Z90",
    "ZI90",
    "ZX",
    "ZX90",
    "ZZ",
    "ZZ90",
    "H",
    "I",
    "S",
    "T",
    "X",
    "Y",
    "Z",
    "get",
    "names",
    "rotation",
]


def rotation(generator: qt.Qobj, angle: float) -> qt.Qobj:
    """
    Construct a unitary rotation from a Hermitian generator.

    Parameters
    ----------
    generator : qt.Qobj
        Hermitian operator whose normalization and sign define the rotation.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    qt.Qobj
        Unitary operator `exp(-1j * angle * generator / 2)` with the same
        tensor dimensions as `generator`.

    Raises
    ------
    TypeError
        If the generator is not a `qt.Qobj`.
    ValueError
        If `generator` is not a Hermitian operator or `angle` is not finite.
    """
    if not isinstance(generator, qt.Qobj):
        raise TypeError("The generator must be a Qobj.")
    if not generator.isoper or not generator.isherm:
        raise ValueError("The generator must be a Hermitian operator.")
    if not np.isfinite(angle):
        raise ValueError("The rotation angle must be finite.")
    return (-0.5j * angle * generator).expm()


I = qt.qeye(2)
X = qt.sigmax()
Y = qt.sigmay()
Z = qt.sigmaz()
XX = qt.tensor(X, X)
YY = qt.tensor(Y, Y)
ZZ = qt.tensor(Z, Z)
ZX = qt.tensor(Z, X)
H = qt.gates.snot()
S = qt.gates.s_gate()
T = qt.gates.t_gate()
SX = qt.gates.sqrtnot()

_ONE_OVER_SQRT_TWO = 1 / np.sqrt(2)

X90 = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO, -1j * _ONE_OVER_SQRT_TWO],
            [-1j * _ONE_OVER_SQRT_TWO, _ONE_OVER_SQRT_TWO],
        ],
        dtype=np.complex128,
    )
)
Y90 = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO, -_ONE_OVER_SQRT_TWO],
            [_ONE_OVER_SQRT_TWO, _ONE_OVER_SQRT_TWO],
        ],
        dtype=np.complex128,
    )
)
Z90 = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO * (1 - 1j), 0],
            [0, _ONE_OVER_SQRT_TWO * (1 + 1j)],
        ],
        dtype=np.complex128,
    )
)

II = qt.tensor(I, I)
IX90 = qt.tensor(I, X90)
IY90 = qt.tensor(I, Y90)
IZ90 = qt.tensor(I, Z90)
XI90 = qt.tensor(X90, I)
YI90 = qt.tensor(Y90, I)
ZI90 = qt.tensor(Z90, I)
ZX90 = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO, -1j * _ONE_OVER_SQRT_TWO, 0, 0],
            [-1j * _ONE_OVER_SQRT_TWO, _ONE_OVER_SQRT_TWO, 0, 0],
            [0, 0, _ONE_OVER_SQRT_TWO, 1j * _ONE_OVER_SQRT_TWO],
            [0, 0, 1j * _ONE_OVER_SQRT_TWO, _ONE_OVER_SQRT_TWO],
        ],
        dtype=np.complex128,
    ),
    dims=[[2, 2], [2, 2]],
)
ZZ90 = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO * (1 - 1j), 0, 0, 0],
            [0, _ONE_OVER_SQRT_TWO * (1 + 1j), 0, 0],
            [0, 0, _ONE_OVER_SQRT_TWO * (1 + 1j), 0],
            [0, 0, 0, _ONE_OVER_SQRT_TWO * (1 - 1j)],
        ],
        dtype=np.complex128,
    ),
    dims=[[2, 2], [2, 2]],
)

CNOT = qt.gates.cnot()
CX = CNOT
CY = qt.gates.cy_gate()
CZ = qt.gates.cz_gate()
CH = qt.Qobj(
    np.block(
        [
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), H.full()],
        ]
    ),
    dims=[[2, 2], [2, 2]],
)
SWAP = qt.gates.swap()
ISWAP = qt.gates.iswap()
SQRT_ISWAP = qt.gates.sqrtiswap()
BSWAP = qt.Qobj(
    np.array(
        [
            [0, 0, 0, 1j],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1j, 0, 0, 0],
        ],
        dtype=np.complex128,
    ),
    dims=[[2, 2], [2, 2]],
)
SQRT_BSWAP = qt.Qobj(
    np.array(
        [
            [_ONE_OVER_SQRT_TWO, 0, 0, 1j * _ONE_OVER_SQRT_TWO],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1j * _ONE_OVER_SQRT_TWO, 0, 0, _ONE_OVER_SQRT_TWO],
        ],
        dtype=np.complex128,
    ),
    dims=[[2, 2], [2, 2]],
)


_NAMED_GATES: dict[str, qt.Qobj] = {
    "BSWAP": BSWAP,
    "CH": CH,
    "CNOT": CNOT,
    "CX": CX,
    "CY": CY,
    "CZ": CZ,
    "H": H,
    "I": I,
    "II": II,
    "ISWAP": ISWAP,
    "IX90": IX90,
    "IY90": IY90,
    "IZ90": IZ90,
    "S": S,
    "SQRT_BSWAP": SQRT_BSWAP,
    "SQRT_ISWAP": SQRT_ISWAP,
    "SWAP": SWAP,
    "SX": SX,
    "T": T,
    "X": X,
    "X180": X,
    "X90": X90,
    "XI90": XI90,
    "Y": Y,
    "Y180": Y,
    "Y90": Y90,
    "YI90": YI90,
    "Z": Z,
    "Z180": Z,
    "Z90": Z90,
    "ZI90": ZI90,
    "ZX90": ZX90,
    "ZZ90": ZZ90,
}


def names() -> tuple[str, ...]:
    """
    Return the sorted canonical gate names accepted by `get`.

    Returns
    -------
    tuple[str, ...]
        Uppercase gate names, including lookup-only aliases such as `X180`.
    """
    return tuple(sorted(_NAMED_GATES))


def get(name: str) -> qt.Qobj:
    """
    Return an independent copy of a named static gate.

    Parameters
    ----------
    name : str
        Case-insensitive canonical gate name returned by `names`.

    Returns
    -------
    qt.Qobj
        Unitary operator with the registered tensor dimensions preserved.

    Raises
    ------
    ValueError
        If the gate name is unknown.
    """
    canonical_name = name.upper()
    try:
        return _NAMED_GATES[canonical_name].copy()
    except KeyError:
        available = ", ".join(names())
        raise ValueError(
            f"Unknown gate '{name}'. Available named gates: {available}."
        ) from None
