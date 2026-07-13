"""Phase shift primitives for virtual Z operations."""

from __future__ import annotations


class PhaseShift:
    """Phase shift value for virtual operations."""

    def __init__(self, theta: float):
        self.theta = theta

    def __repr__(self):
        """Return a concise string representation."""
        return f"{self.__class__.__name__}({self.theta:.2f})"


class VirtualZ(PhaseShift):
    """Virtual Z rotation implemented via phase shift."""

    def __init__(self, theta: float):
        super().__init__(-theta)
