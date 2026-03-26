"""Qubex-specific Sequencer wrapper on top of selected driver Sequencer."""

from __future__ import annotations

from collections.abc import MutableSequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from .driver_loader import load_quel1_driver

driver = load_quel1_driver()

if TYPE_CHECKING:
    from .qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        PortConfigAcquirerProtocol,
        SequencerProtocol as Sequencer,
    )
else:
    BoxPool = driver.BoxPool
    Sequencer = driver.Sequencer


class Quel1Sequencer(Sequencer):
    """
    Sequencer variant with qubex-specific timing and resource-resolution behavior.

    Notes
    -----
    Besides disabling automatic first-padding insertion, this class keeps
    conversion-time port resolution on the `boxpool` path. That mirrors the
    legacy qubex wrapper behavior and avoids readout sideband regressions
    observed in older qubecalib driver-path handling.
    """

    @staticmethod
    def _extract_box_name_and_port(mapping: dict[str, Any]) -> tuple[str, Any]:
        """Extract box-name and port-number fields from one resource-map entry."""
        box_name = getattr(mapping.get("box"), "box_name", None)
        port_number = getattr(mapping.get("port"), "port", None)
        if not isinstance(box_name, str):
            raise TypeError("box_name is not defined")
        if port_number is None:
            raise TypeError("port is not defined")
        return box_name, port_number

    def generate_cap_resource_map(self, boxpool: BoxPool) -> dict[str, Any]:
        """Build target-to-capture resource map using boxpool port directions."""
        # Keep direction resolution on boxpool path to match legacy wrapper behavior.
        # This intentionally ignores driver-side port checks used by base Sequencer.
        cap_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, mappings in self.resource_map.items():
            for mapping in mappings:
                box_name, port_number = self._extract_box_name_and_port(mapping)
                if (
                    port_number in boxpool.get_box(box_name)[0].get_input_ports()
                    and target_name in self.cap_sampled_sequence
                ):
                    cap_resource_map.setdefault(target_name, []).append(mapping)
        return {
            target_name: next(iter(mappings))
            for target_name, mappings in cap_resource_map.items()
            if mappings
        }

    @override
    def generate_e7_settings(
        self,
        boxpool: BoxPool,
    ) -> tuple[dict[tuple[str, Any, int], Any], dict[tuple[str, Any, int], Any], dict]:
        """Generate settings via boxpool-resolved PortConfigAcquirer instances."""
        # Important: older qubecalib driver-path handling can mis-resolve readout
        # sideband on R8. We therefore keep conversion logic on boxpool path,
        # equivalent to historical qubex behavior in backend/sequencer_mod.py.
        module_name = (
            "qxdriver_quel1.qubecalib"
            if driver.package_name == "qxdriver_quel1"
            else "qubecalib.qubecalib"
        )
        port_config_acquirer_cls = cast(
            "type[PortConfigAcquirerProtocol]",
            import_module(module_name).PortConfigAcquirer,
        )
        cap_resource_map = self.generate_cap_resource_map(boxpool)
        unresolved_gen_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, mappings in self.resource_map.items():
            for mapping in mappings:
                box_name, port_number = self._extract_box_name_and_port(mapping)
                if (
                    port_number in boxpool.get_box(box_name)[0].get_output_ports()
                    and target_name in self.gen_sampled_sequence
                ):
                    unresolved_gen_resource_map.setdefault(target_name, []).append(
                        mapping
                    )
        gen_resource_map: dict[str, dict[str, Any]] = {
            target_name: next(iter(mappings))
            for target_name, mappings in unresolved_gen_resource_map.items()
            if mappings
        }

        cap_target_portconf: dict[str, Any] = {}
        for target_name, mapping in cap_resource_map.items():
            box_name, port_number = self._extract_box_name_and_port(mapping)
            channel_number = mapping.get("channel_number")
            if not isinstance(channel_number, int):
                raise TypeError("channel is not defined")
            box = boxpool.get_box(box_name)[0]
            # Do not pass `driver`; force boxpool-style port config acquisition.
            cap_target_portconf[target_name] = port_config_acquirer_cls(
                boxpool=boxpool,
                box_name=box_name,
                box=box,
                port=port_number,
                channel=channel_number,
            )

        gen_target_portconf: dict[str, Any] = {}
        for target_name, mapping in gen_resource_map.items():
            box_name, port_number = self._extract_box_name_and_port(mapping)
            channel_number = mapping.get("channel_number")
            if not isinstance(channel_number, int):
                raise TypeError("channel is not defined")
            box = boxpool.get_box(box_name)[0]
            # Do not pass `driver`; force boxpool-style port config acquisition.
            gen_target_portconf[target_name] = port_config_acquirer_cls(
                boxpool=boxpool,
                box_name=box_name,
                box=box,
                port=port_number,
                channel=channel_number,
            )

        interval_ns = self.interval if self.interval is not None else 10240
        cap_e7_settings = driver.Converter.convert_to_cap_device_specific_sequence(
            gen_sampled_sequence=self.gen_sampled_sequence,
            cap_sampled_sequence=self.cap_sampled_sequence,
            resource_map=cap_resource_map,
            port_config=cap_target_portconf,
            repeats=self.repeats,
            interval=interval_ns,
            integral_mode=self.integral_mode,
            dsp_demodulation=self.dsp_demodulation,
            software_demodulation=self.software_demodulation,
            enable_sum=self.enable_sum,
            enable_classification=self.enable_classification,
            line_param0=self.line_param0,
            line_param1=self.line_param1,
            line_param0_by_target=self.line_param0_by_target,
            line_param1_by_target=self.line_param1_by_target,
        )
        gen_e7_settings = driver.Converter.convert_to_gen_device_specific_sequence(
            gen_sampled_sequence=self.gen_sampled_sequence,
            cap_sampled_sequence=self.cap_sampled_sequence,
            resource_map=gen_resource_map,
            port_config=gen_target_portconf,
            repeats=self.repeats,
            interval=interval_ns,
        )
        return cap_e7_settings, gen_e7_settings, cap_resource_map

    @override
    def calc_first_padding(self) -> int:
        """
        Override Sequencer.calc_first_padding to disable first padding.

        Returns
        -------
        int
            Always 0.

        Examples
        --------
        >>> sequencer.calc_first_padding()
        0
        """
        return 0
