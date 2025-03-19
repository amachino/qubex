from __future__ import annotations


class PhaseShift:
    def __init__(self, theta: float):
        self.theta = theta

    def __repr__(self):
        return f"{self.__class__.__name__}({self.theta:.2f})"


class VirtualZ(PhaseShift):
    def __init__(self, theta: float):
        super().__init__(-theta)
