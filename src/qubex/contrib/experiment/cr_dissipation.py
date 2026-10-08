"""Expose CR dissipation characterization as a stable public module."""

from ._cr_dissipation.experiment import characterize_cr_dissipation

__all__ = ["characterize_cr_dissipation"]
