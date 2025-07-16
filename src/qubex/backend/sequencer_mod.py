from __future__ import annotations

from typing import Any, MutableSequence

from qubecalib.qubecalib import (
    BoxPool,
    BoxSetting,
    CaptureParam,
    Converter,
    PortConfigAcquirer,
    PortSetting,
    Quel1PortType,
    Sequencer,
    TargetBPC,
    WaveSequence,
)


class SequencerMod(Sequencer):
    def generate_cap_resource_map(self, boxpool: BoxPool) -> dict[str, Any]:
        _cap_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    port in boxpool.get_box(box_name)[0].get_input_ports()
                    and target_name in self.cap_sampled_sequence
                ):
                    if target_name in _cap_resource_map:
                        _cap_resource_map[target_name].append(m)
                    else:
                        _cap_resource_map[target_name] = [m]
        return {
            target_name: next(iter(maps))
            for target_name, maps in _cap_resource_map.items()
            if maps
        }

    def generate_e7_settings(
        self,
        boxpool: BoxPool,
    ) -> tuple[
        dict[tuple[str, int, int], CaptureParam],
        dict[tuple[str, Quel1PortType, int], WaveSequence],
        dict[str, Any],
    ]:
        cap_resource_map = self.generate_cap_resource_map(boxpool)
        _gen_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    port in boxpool.get_box(box_name)[0].get_output_ports()
                    and target_name in self.gen_sampled_sequence
                ):
                    if target_name in _gen_resource_map:
                        _gen_resource_map[target_name].append(m)
                    else:
                        _gen_resource_map[target_name] = [m]
        gen_resource_map: dict[str, Any] = {
            target_name: next(iter(maps))
            for target_name, maps in _gen_resource_map.items()
            if maps
        }

        cap_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
                box_name=m["box"].box_name,
            )
            for target_name, m in cap_resource_map.items()
        }
        gen_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
                box_name=m["box"].box_name,
            )
            for target_name, m in gen_resource_map.items()
        }
        cap_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in cap_target_bpc.items()
        }

        # first_blank = min(
        #     [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
        # )
        # first_padding = ((first_blank - 1) // 64 + 1) * 64 - first_blank  # Sa
        # ref_sequence = next(iter(csseq.values()))
        # first_padding = self.calc_first_padding()

        # for target_name, cseq in self.cap_sampled_sequence.items():
        #     cseq.padding += first_padding
        # for target_name, gseq in self.gen_sampled_sequence.items():
        #     gseq.padding += first_padding

        interval = self.interval if self.interval is not None else 10240
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam] = (
            Converter.convert_to_cap_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=cap_resource_map,
                # target_freq=target_freq,
                port_config=cap_target_portconf,
                repeats=self.repeats,
                interval=interval,
                integral_mode=self.integral_mode,
                dsp_demodulation=self.dsp_demodulation,
                software_demodulation=self.software_demodulation,
                enable_sum=self.enable_sum,
            )
        )
        # phase_offset_list_by_target = {
        #     target: [-2 * np.pi * cap_fmod[target] * t for t in reference_time_list]
        #     for target, reference_time_list in reference_time_list_by_target.items()
        # }

        gen_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in gen_target_bpc.items()
        }
        gen_e7_settings: dict[tuple[str, Quel1PortType, int], WaveSequence] = (
            Converter.convert_to_gen_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=gen_resource_map,
                port_config=gen_target_portconf,
                repeats=self.repeats,
                interval=interval,
            )
        )
        return cap_e7_settings, gen_e7_settings, cap_resource_map
