from __future__ import annotations


class PhaseShift:
    def __init__(self, theta: float):
        self.theta = theta


class VirtualZ(PhaseShift):
    def __init__(self, theta: float):
        super().__init__(-theta)
