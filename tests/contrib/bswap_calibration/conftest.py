"""Portable ideal circuit assets generated once per test session."""

import json
from collections import deque

import numpy as np
import pytest


@pytest.fixture(scope="session")
def native_bswap_cache_file(tmp_path_factory):
    """Generate the minus-i native Clifford table without laboratory artifacts."""
    identity = np.eye(2, dtype=complex)
    x90 = (identity - 1j * np.array([[0, 1], [1, 0]])) / np.sqrt(2)
    z90 = np.diag(np.exp(-0.25j * np.pi * np.array([1, -1])))
    minus_bswap = np.array(
        [[0, 0, 0, -1j], [0, 1, 0, 0], [0, 0, 1, 0], [-1j, 0, 0, 0]],
        dtype=complex,
    )
    generators = {
        "XI90": np.kron(x90, identity),
        "IX90": np.kron(identity, x90),
        "ZI90": np.kron(z90, identity),
        "IZ90": np.kron(identity, z90),
        "BSWAP": minus_bswap,
    }

    def phase_key(unitary):
        """Canonicalize global phase independently of the cache implementation."""
        nonzero = np.flatnonzero(np.abs(unitary) > 1e-8)
        phase = unitary.flat[nonzero[0]]
        normalized = unitary * phase.conjugate() / abs(phase)
        return tuple(
            np.rint(np.r_[normalized.real.ravel(), normalized.imag.ravel()] * 1e7)
            .astype(np.int64)
            .tolist()
        )

    initial = np.eye(4, dtype=complex)
    seen = {phase_key(initial)}
    pending = deque([(initial, [])])
    rows = []
    while pending:
        unitary, sequence = pending.popleft()
        rows.append({"sequence": sequence})
        for name, generator in generators.items():
            candidate = generator @ unitary
            key = phase_key(candidate)
            if key not in seen:
                seen.add(key)
                pending.append((candidate, [*sequence, name]))
    assert len(rows) == 11520
    path = tmp_path_factory.mktemp("ideal-bswap-cache") / "clifford_list_2q_bswap.json"
    path.write_text(json.dumps(rows))
    return path
