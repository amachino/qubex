"""Registry for resolving target labels without ad-hoc string parsing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .target import CapTarget, Target


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

    def register(self, target: Target | CapTarget) -> None:
        """Register one target and rebuild label-resolution indexes."""
        if isinstance(target, Target):
            self._gen_target_dict[target.label] = target
        elif isinstance(target, CapTarget):
            self._cap_target_dict[target.label] = target
        else:
            raise TypeError("target must be a Target or CapTarget instance.")
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
        self._cr_pair_by_label: dict[str, tuple[str, str]] = {}

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
                    self._cr_pair_by_label[label] = pair
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

    def get_gen_target(self, label: str) -> Target:
        """Return one generator target by label."""
        try:
            return self._gen_target_dict[label]
        except KeyError:
            raise KeyError(f"Generator target `{label}` is not registered.") from None

    def get_cap_target(self, label: str) -> CapTarget:
        """Return one capture target by label."""
        try:
            return self._cap_target_dict[label]
        except KeyError:
            raise KeyError(f"Capture target `{label}` is not registered.") from None

    def resolve_qubit_label(
        self,
        label: str,
        *,
        allow_legacy: bool = False,
    ) -> str:
        """Resolve a qubit label from registry; optionally allow legacy parsing."""
        if label in self._qubit_labels:
            return label
        resolved = self._target_to_qubit.get(label)
        if resolved is not None:
            return resolved
        if allow_legacy:
            try:
                return Target.qubit_label(label)
            except ValueError:
                pass
        raise ValueError(f"Qubit label could not be resolved from `{label}`.")

    def resolve_ge_label(
        self,
        label: str,
        *,
        allow_legacy: bool = False,
    ) -> str:
        """Resolve a GE target label; optionally allow legacy parsing."""
        qubit_label = self.resolve_qubit_label(label, allow_legacy=allow_legacy)
        resolved = self._ge_label_by_qubit.get(qubit_label)
        if resolved is None:
            if allow_legacy:
                return Target.ge_label(label)
            raise ValueError(f"GE target is not registered for qubit `{qubit_label}`.")
        return resolved

    def resolve_ef_label(
        self,
        label: str,
        *,
        allow_legacy: bool = False,
    ) -> str:
        """Resolve an EF target label; optionally allow legacy parsing."""
        qubit_label = self.resolve_qubit_label(label, allow_legacy=allow_legacy)
        resolved = self._ef_label_by_qubit.get(qubit_label)
        if resolved is None:
            if allow_legacy:
                return Target.ef_label(label)
            raise ValueError(f"EF target is not registered for qubit `{qubit_label}`.")
        return resolved

    def resolve_read_label(
        self,
        label: str,
        *,
        allow_legacy: bool = False,
    ) -> str:
        """Resolve a readout target label; optionally allow legacy parsing."""
        qubit_label = self.resolve_qubit_label(label, allow_legacy=allow_legacy)
        resolved = self._read_label_by_qubit.get(qubit_label)
        if resolved is None:
            if allow_legacy:
                return Target.read_label(label)
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

    def resolve_cr_pair(
        self,
        label: str,
        *,
        allow_legacy: bool = False,
    ) -> tuple[str, str]:
        """Resolve CR pair from registry; optionally allow legacy parsing."""
        resolved = self._cr_pair_by_label.get(label)
        if resolved is None:
            if allow_legacy:
                return Target.cr_qubit_pair(label)
            raise ValueError(f"CR target `{label}` is not registered.")
        return resolved

    def measurement_output_label(self, target_label: str) -> str:
        """Resolve canonical measurement output label for one target."""
        try:
            return self.resolve_qubit_label(target_label)
        except ValueError:
            return target_label
