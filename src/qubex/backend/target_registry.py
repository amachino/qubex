"""Registry for resolving target labels without ad-hoc string parsing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .target import CapTarget, Target


@dataclass(frozen=True)
class TargetResolution:
    """Resolved target properties used by higher-level mapping logic."""

    label: str
    kind: Literal["gen", "cap"]
    qubit_label: str | None


class TargetRegistry:
    """Resolve target metadata and label conversions from registered targets."""

    def __init__(
        self,
        *,
        gen_targets: Mapping[str, Target] | Iterable[Target] | None = None,
        cap_targets: Mapping[str, CapTarget] | Iterable[CapTarget] | None = None,
    ) -> None:
        self._gen_target_dict = self._to_gen_target_map(gen_targets)
        self._cap_target_dict = self._to_cap_target_map(cap_targets)
        self._build_indexes()

    @staticmethod
    def _to_gen_target_map(
        targets: Mapping[str, Target] | Iterable[Target] | None,
    ) -> dict[str, Target]:
        """Normalize generator targets into a label-keyed mapping."""
        if targets is None:
            return {}
        values = targets.values() if isinstance(targets, Mapping) else targets
        result: dict[str, Target] = {}
        for target in values:
            if not isinstance(target, Target):
                raise TypeError("gen_targets entries must be Target instances.")
            result[target.label] = target
        return result

    @staticmethod
    def _to_cap_target_map(
        targets: Mapping[str, CapTarget] | Iterable[CapTarget] | None,
    ) -> dict[str, CapTarget]:
        """Normalize capture targets into a label-keyed mapping."""
        if targets is None:
            return {}
        values = targets.values() if isinstance(targets, Mapping) else targets
        result: dict[str, CapTarget] = {}
        for target in values:
            if not isinstance(target, CapTarget):
                raise TypeError("cap_targets entries must be CapTarget instances.")
            result[target.label] = target
        return result

    def _build_indexes(self) -> None:
        """Build label conversion indexes from registered target metadata."""
        self._target_to_qubit: dict[str, str] = {}
        self._qubit_labels: set[str] = set()
        self._read_label_by_qubit: dict[str, str] = {}
        self._ge_label_by_qubit: dict[str, str] = {}
        self._ef_label_by_qubit: dict[str, str] = {}
        self._cr_default_label_by_control: dict[str, str] = {}
        self._cr_pair_label_by_pair: dict[tuple[str, str], str] = {}

        for label, target in self._gen_target_dict.items():
            qubit_label = target.qubit
            if qubit_label:
                self._target_to_qubit[label] = qubit_label
                self._qubit_labels.add(qubit_label)
            if target.is_read and qubit_label:
                self._read_label_by_qubit.setdefault(qubit_label, label)
            if target.is_ge and qubit_label:
                self._ge_label_by_qubit.setdefault(qubit_label, label)
            if target.is_ef and qubit_label:
                self._ef_label_by_qubit.setdefault(qubit_label, label)
            if target.is_cr:
                pair = self._extract_cr_pair_from_label(label)
                if pair is not None:
                    control_qubit, target_qubit = pair
                    if target_qubit == "CR":
                        self._cr_default_label_by_control.setdefault(
                            control_qubit, label
                        )
                    else:
                        self._cr_pair_label_by_pair.setdefault(pair, label)

        for label, target in self._cap_target_dict.items():
            qubit_label = getattr(target.object, "qubit", None)
            if isinstance(qubit_label, str) and qubit_label:
                self._target_to_qubit[label] = qubit_label
                self._qubit_labels.add(qubit_label)
                self._read_label_by_qubit.setdefault(qubit_label, label)

    def _extract_cr_pair_from_label(self, label: str) -> tuple[str, str] | None:
        """
        Extract CR pair from a registered CR target label.

        Notes
        -----
        This does not perform free-form label parsing. It only interprets labels
        already registered as CR targets and only for known qubit labels.
        """
        if "-" not in label:
            return None
        control, target = label.split("-", maxsplit=1)
        if control not in self._qubit_labels:
            return None
        if target == "CR":
            return (control, target)
        if target in self._qubit_labels:
            return (control, target)
        return None

    @property
    def gen_targets(self) -> dict[str, Target]:
        """Return generator targets keyed by label."""
        return dict(self._gen_target_dict)

    @property
    def cap_targets(self) -> dict[str, CapTarget]:
        """Return capture targets keyed by label."""
        return dict(self._cap_target_dict)

    def resolve_qubit_label(self, label: str) -> str:
        """Resolve a qubit label from a registered target or qubit label."""
        if label in self._qubit_labels:
            return label
        resolved = self._target_to_qubit.get(label)
        if resolved is not None:
            return resolved
        raise ValueError(f"Qubit label could not be resolved from `{label}`.")

    def resolve_ge_label(self, label: str) -> str:
        """Resolve the GE target label for a qubit or registered target label."""
        qubit_label = self.resolve_qubit_label(label)
        resolved = self._ge_label_by_qubit.get(qubit_label)
        if resolved is None:
            raise ValueError(f"GE target is not registered for qubit `{qubit_label}`.")
        return resolved

    def resolve_ef_label(self, label: str) -> str:
        """Resolve the EF target label for a qubit or registered target label."""
        qubit_label = self.resolve_qubit_label(label)
        resolved = self._ef_label_by_qubit.get(qubit_label)
        if resolved is None:
            raise ValueError(f"EF target is not registered for qubit `{qubit_label}`.")
        return resolved

    def resolve_read_label(self, label: str) -> str:
        """Resolve the readout target label for a qubit or registered target label."""
        qubit_label = self.resolve_qubit_label(label)
        resolved = self._read_label_by_qubit.get(qubit_label)
        if resolved is None:
            raise ValueError(
                f"Readout target is not registered for qubit `{qubit_label}`."
            )
        return resolved

    def resolve_cr_label(
        self,
        control_label: str,
        target_label: str | None = None,
    ) -> str:
        """Resolve a CR target label from control/target labels."""
        if target_label is None and control_label in self._gen_target_dict:
            candidate = self._gen_target_dict[control_label]
            if candidate.is_cr:
                return control_label

        control_qubit = self.resolve_qubit_label(control_label)
        if target_label is None:
            resolved = self._cr_default_label_by_control.get(control_qubit)
            if resolved is None:
                raise ValueError(
                    f"Default CR target is not registered for `{control_qubit}`."
                )
            return resolved

        target_qubit = self.resolve_qubit_label(target_label)
        resolved = self._cr_pair_label_by_pair.get((control_qubit, target_qubit))
        if resolved is None:
            raise ValueError(
                f"CR target is not registered for pair `{control_qubit}-{target_qubit}`."
            )
        return resolved

    def measurement_output_label(self, target_label: str) -> str:
        """Resolve canonical measurement output label for one target."""
        try:
            return self.resolve_qubit_label(target_label)
        except ValueError:
            return target_label

    def get(self, label: str) -> TargetResolution:
        """Return resolved metadata for one registered target label."""
        if label in self._gen_target_dict:
            qubit_label = self._target_to_qubit.get(label)
            return TargetResolution(label=label, kind="gen", qubit_label=qubit_label)
        if label in self._cap_target_dict:
            qubit_label = self._target_to_qubit.get(label)
            return TargetResolution(label=label, kind="cap", qubit_label=qubit_label)
        raise KeyError(f"Target `{label}` is not registered.")
