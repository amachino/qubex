# ruff: noqa: SLF001

"""Continuous-wave output manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

from qubex.backend.quel1.compat.box_adapter import adapt_quel1_box
from qubex.backend.quel1.quel1_backend_constants import (
    BLOCK_DURATION_NS,
    BLOCK_LENGTH,
    SAMPLING_PERIOD_NS,
)
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContextReader

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import PortType

MAX_DAC_CODE: Final[int] = 32767
MAX_CONTINUOUS_WAVE_REPEAT: Final[int] = 0xFFFF_FFFF
CONTINUOUS_WAVE_FREQUENCY_UNIT_HZ: Final[float] = 1e9 / BLOCK_DURATION_NS
FREQUENCY_MULTIPLE_TOLERANCE_HZ: Final[float] = 1e-3
CONTINUOUS_WAVE_ALIAS_WARNING_THRESHOLD_HZ: Final[float] = 800_000_000.0

logger = logging.getLogger(__name__)

ContinuousWaveKey: TypeAlias = tuple[str, Any, int]


@dataclass(frozen=True)
class Quel1ContinuousWaveConfig:
    """Configuration used for one active QuEL-1 continuous-wave output."""

    box_name: str
    port: PortType
    channel: int
    awg_freq_hz: float
    cycles_per_chunk: int
    amplitude: float
    phase_rad: float
    lo_freq_hz: float | None
    cnco_freq_hz: float | None
    fnco_freq_hz: float | None
    actual_output_freq_hz: float | None
    sideband: str | None
    vatt: int | None
    fullscale_current: int | None
    rfswitch: str | None
    configure_port: bool
    waveform_name: str
    chunk_repeats: int
    awg_repeats: int
    duration_s: float


class Quel1ContinuousWaveManager:
    """Manage long-running QuEL-1 continuous-wave AWG tasks."""

    def __init__(self, *, runtime_context: Quel1RuntimeContextReader) -> None:
        self._runtime_context = runtime_context
        self._configs: dict[ContinuousWaveKey, Quel1ContinuousWaveConfig] = {}
        self._tasks: dict[ContinuousWaveKey, Any] = {}

    def start_continuous_wave(
        self,
        *,
        box_name: str,
        port: PortType,
        channel: int,
        amplitude: float = 1.0,
        phase_rad: float = 0.0,
        lo_freq_hz: float | None = None,
        cnco_freq_hz: float | None = None,
        fnco_freq_hz: float | None = None,
        awg_freq_hz: float | None = None,
        sideband: str | None = None,
        vatt: int | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
        configure_port: bool = False,
        chunk_repeats: int = MAX_CONTINUOUS_WAVE_REPEAT,
        awg_repeats: int = MAX_CONTINUOUS_WAVE_REPEAT,
    ) -> Quel1ContinuousWaveConfig:
        """
        Configure and start one continuous-wave AWG output.

        Parameters
        ----------
        box_name : str
            Connected QuEL-1 box name.
        port : PortType
            Output port identifier.
        channel : int
            AWG channel number on the output port.
        amplitude : float, optional
            Peak amplitude normalized to the DAC code range. Valid range is
            0.0 to 1.0.
        phase_rad : float, optional
            Initial waveform phase in radians.
        lo_freq_hz : float | None, optional
            LO frequency to apply when `configure_port=True`.
        cnco_freq_hz : float | None, optional
            CNCO frequency to apply when `configure_port=True`.
        fnco_freq_hz : float | None, optional
            FNCO frequency to apply when `configure_port=True`.
        awg_freq_hz : float | None, optional
            AWG-baseband frequency in Hz. If omitted, uses 0 Hz. It must be an
            integer multiple of `CONTINUOUS_WAVE_FREQUENCY_UNIT_HZ`.
        sideband : str | None, optional
            Sideband setting to apply when `configure_port=True`.
        vatt : int | None, optional
            VATT setting to apply when `configure_port=True`.
        fullscale_current : int | None, optional
            Full-scale current setting to apply when `configure_port=True`.
        rfswitch : str | None, optional
            RF switch setting to apply when `configure_port=True`.
        configure_port : bool, optional
            Whether to update output path settings before starting the AWG.
            The default preserves current hardware output and phase state.
        chunk_repeats : int, optional
            Repeat count for the generated 128 ns chunk.
        awg_repeats : int, optional
            Repeat count for the AWG sequence.

        Returns
        -------
        Quel1ContinuousWaveConfig
            Resolved continuous-wave configuration.
        """
        self._validate_output_update_options(
            configure_port=configure_port,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            fnco_freq_hz=fnco_freq_hz,
            sideband=sideband,
            vatt=vatt,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )
        key = self._key(box_name=box_name, port=port, channel=channel)
        task = self._tasks.get(key)
        if task is not None and self._is_task_active(task):
            raise RuntimeError(
                f"Continuous wave is already running on box={box_name}, "
                f"port={port}, channel={channel}."
            )

        requested_awg_freq_hz = 0.0 if awg_freq_hz is None else float(awg_freq_hz)
        cycles_per_chunk, actual_awg_freq_hz = self._awg_frequency_to_chunk_cycles(
            requested_awg_freq_hz
        )
        iq = self._make_iq_chunk(
            cycles_per_chunk=cycles_per_chunk,
            amplitude=amplitude,
            phase_rad=phase_rad,
        )
        waveform_name = self._waveform_name(
            box_name=box_name,
            port=port,
            channel=channel,
            cycles_per_chunk=cycles_per_chunk,
        )
        box = self._resolve_connected_box(box_name)
        (
            resolved_lo_freq_hz,
            resolved_cnco_freq_hz,
            resolved_fnco_freq_hz,
            resolved_sideband,
            resolved_vatt,
            resolved_fullscale_current,
            resolved_rfswitch,
        ) = self._resolve_output_frequencies(
            box=box,
            port=port,
            channel=int(channel),
            configure_port=configure_port,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            fnco_freq_hz=fnco_freq_hz,
            sideband=sideband,
            vatt=vatt,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )
        config = Quel1ContinuousWaveConfig(
            box_name=box_name,
            port=port,
            channel=int(channel),
            awg_freq_hz=actual_awg_freq_hz,
            cycles_per_chunk=cycles_per_chunk,
            amplitude=float(amplitude),
            phase_rad=float(phase_rad),
            lo_freq_hz=resolved_lo_freq_hz,
            cnco_freq_hz=resolved_cnco_freq_hz,
            fnco_freq_hz=resolved_fnco_freq_hz,
            actual_output_freq_hz=self._resolve_actual_output_frequency_hz(
                lo_freq_hz=resolved_lo_freq_hz,
                cnco_freq_hz=resolved_cnco_freq_hz,
                fnco_freq_hz=resolved_fnco_freq_hz,
                awg_freq_hz=actual_awg_freq_hz,
                sideband=resolved_sideband,
            ),
            sideband=resolved_sideband,
            vatt=resolved_vatt,
            fullscale_current=resolved_fullscale_current,
            rfswitch=resolved_rfswitch,
            configure_port=configure_port,
            waveform_name=waveform_name,
            chunk_repeats=int(chunk_repeats),
            awg_repeats=int(awg_repeats),
            duration_s=(BLOCK_DURATION_NS * 1e-9)
            * int(chunk_repeats)
            * int(awg_repeats),
        )

        config_port_kwargs = self._build_config_port_kwargs(
            port=port,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            sideband=sideband,
            vatt=vatt,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
            configure_port=configure_port,
        )
        if config_port_kwargs is not None:
            box.config_port(**config_port_kwargs)
        box.register_wavedata(
            port=port,
            channel=int(channel),
            name=waveform_name,
            iq=iq,
            allow_update=True,
        )

        awg_param = self._build_awg_param(
            waveform_name=waveform_name,
            chunk_repeats=int(chunk_repeats),
            awg_repeats=int(awg_repeats),
        )
        config_channel_kwargs: dict[str, Any] = {
            "port": port,
            "channel": int(channel),
            "awg_param": awg_param,
        }
        if configure_port and fnco_freq_hz is not None:
            config_channel_kwargs["fnco_freq"] = fnco_freq_hz
        box.config_channel(**config_channel_kwargs)

        task = box.start_wavegen(
            {(port, int(channel))},
            disable_timeout=True,
            return_after_start_emission=True,
        )
        self._configs[key] = config
        self._tasks[key] = task
        self._log_started_frequencies(config)
        return config

    def stop_continuous_wave(
        self,
        *,
        box_name: str,
        port: PortType,
        channel: int,
        timeout: float = 2.0,
        polling_period: float = 0.01,
    ) -> bool:
        """
        Stop one active continuous-wave output.

        Returns
        -------
        bool
            `True` when a remembered task was stopped or cleared, otherwise
            `False`.
        """
        key = self._key(box_name=box_name, port=port, channel=channel)
        task = self._tasks.get(key)
        if task is None:
            return False
        try:
            if self._is_task_active(task):
                self._cancel_task(
                    task,
                    timeout=timeout,
                    polling_period=polling_period,
                )
        finally:
            self._tasks.pop(key, None)
            self._configs.pop(key, None)
        return True

    def stop_all_continuous_waves(
        self,
        *,
        timeout: float = 2.0,
        polling_period: float = 0.01,
    ) -> None:
        """Stop all remembered continuous-wave outputs."""
        for box_name, port, channel in list(self._tasks):
            self.stop_continuous_wave(
                box_name=box_name,
                port=port,
                channel=channel,
                timeout=timeout,
                polling_period=polling_period,
            )

    @staticmethod
    def _validate_output_update_options(
        *,
        configure_port: bool,
        lo_freq_hz: float | None,
        cnco_freq_hz: float | None,
        fnco_freq_hz: float | None,
        sideband: str | None,
        vatt: int | None,
        fullscale_current: int | None,
        rfswitch: str | None,
    ) -> None:
        """Reject implicit output path updates."""
        if configure_port:
            return
        if (
            lo_freq_hz is None
            and cnco_freq_hz is None
            and fnco_freq_hz is None
            and sideband is None
            and vatt is None
            and fullscale_current is None
            and rfswitch is None
        ):
            return
        raise ValueError(
            "LO/CNCO/FNCO/sideband/VATT/full-scale-current/RF-switch arguments require "
            "configure_port=True because output reconfiguration can reset "
            "hardware phase state."
        )

    @staticmethod
    def _key(
        *,
        box_name: str,
        port: PortType,
        channel: int,
    ) -> ContinuousWaveKey:
        """Return normalized state key."""
        return (box_name, port, int(channel))

    def _resolve_connected_box(self, box_name: str) -> Any:
        """Return connected box through the compatibility adapter."""
        self._runtime_context.validate_box_availability(box_name)
        boxpool = self._runtime_context.boxpool
        if box_name not in boxpool._boxes:
            raise ValueError(
                f"Box {box_name} is not connected. Call connect() method first."
            )
        return adapt_quel1_box(boxpool._boxes[box_name][0])

    @staticmethod
    def _is_task_active(task: Any) -> bool:
        """Return whether a wavegen task appears active."""
        done = getattr(task, "done", None)
        if callable(done) and bool(done()):
            return False
        cancelled = getattr(task, "cancelled", None)
        return not (callable(cancelled) and bool(cancelled()))

    @staticmethod
    def _cancel_task(
        task: Any,
        *,
        timeout: float,
        polling_period: float,
    ) -> None:
        """Cancel a wavegen task across task API variants."""
        cancel = getattr(task, "cancel", None)
        if not callable(cancel):
            raise TypeError("Continuous-wave task does not support cancellation.")
        try:
            cancel(timeout=timeout, polling_period=polling_period)
        except TypeError:
            cancel()

    @classmethod
    def _resolve_output_frequencies(
        cls,
        *,
        box: Any,
        port: PortType,
        channel: int,
        configure_port: bool,
        lo_freq_hz: float | None,
        cnco_freq_hz: float | None,
        fnco_freq_hz: float | None,
        sideband: str | None,
        vatt: int | None,
        fullscale_current: int | None,
        rfswitch: str | None,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        str | None,
        int | None,
        int | None,
        str | None,
    ]:
        """Resolve output path values used for one CW output."""
        (
            current_lo_freq_hz,
            current_cnco_freq_hz,
            current_fnco_freq_hz,
            current_sideband,
            current_vatt,
            current_fullscale_current,
            current_rfswitch,
        ) = cls._read_current_output_settings(
            box=box,
            port=port,
            channel=channel,
        )
        if not configure_port:
            return (
                current_lo_freq_hz,
                current_cnco_freq_hz,
                current_fnco_freq_hz,
                current_sideband,
                current_vatt,
                current_fullscale_current,
                current_rfswitch,
            )
        return (
            lo_freq_hz if lo_freq_hz is not None else current_lo_freq_hz,
            cnco_freq_hz if cnco_freq_hz is not None else current_cnco_freq_hz,
            fnco_freq_hz if fnco_freq_hz is not None else current_fnco_freq_hz,
            sideband if sideband is not None else current_sideband,
            vatt if vatt is not None else current_vatt,
            fullscale_current
            if fullscale_current is not None
            else current_fullscale_current,
            rfswitch if rfswitch is not None else current_rfswitch,
        )

    @classmethod
    def _read_current_output_settings(
        cls,
        *,
        box: Any,
        port: PortType,
        channel: int,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        str | None,
        int | None,
        int | None,
        str | None,
    ]:
        """Read current output path values from a port dump when available."""
        dump_port = getattr(box, "dump_port", None)
        if not callable(dump_port):
            return None, None, None, None, None, None, None
        try:
            port_dump = dump_port(port)
        except Exception:
            logger.exception(
                "Failed to dump port %s for continuous-wave logging.", port
            )
            return None, None, None, None, None, None, None
        if not isinstance(port_dump, dict):
            return None, None, None, None, None, None, None
        lo_freq_hz = cls._as_optional_float(port_dump.get("lo_freq"))
        cnco_freq_hz = cls._as_optional_float(port_dump.get("cnco_freq"))
        sideband = cls._as_optional_string(port_dump.get("sideband"))
        vatt = cls._as_optional_int(port_dump.get("vatt"))
        fullscale_current = cls._as_optional_int(port_dump.get("fullscale_current"))
        rfswitch = cls._as_optional_string(port_dump.get("rfswitch"))
        channel_dump = cls._lookup_channel_dump(port_dump, channel)
        fnco_freq_hz = (
            None
            if channel_dump is None
            else cls._as_optional_float(channel_dump.get("fnco_freq"))
        )
        return (
            lo_freq_hz,
            cnco_freq_hz,
            fnco_freq_hz,
            sideband,
            vatt,
            fullscale_current,
            rfswitch,
        )

    @staticmethod
    def _build_config_port_kwargs(
        *,
        port: PortType,
        lo_freq_hz: float | None,
        cnco_freq_hz: float | None,
        sideband: str | None,
        vatt: int | None,
        fullscale_current: int | None,
        rfswitch: str | None,
        configure_port: bool,
    ) -> dict[str, Any] | None:
        """Build `config_port` kwargs for explicitly supplied output settings."""
        if not configure_port:
            return None
        kwargs: dict[str, Any] = {"port": port}
        if lo_freq_hz is not None:
            kwargs["lo_freq"] = lo_freq_hz
        if cnco_freq_hz is not None:
            kwargs["cnco_freq"] = cnco_freq_hz
        if sideband is not None:
            kwargs["sideband"] = sideband
        if vatt is not None:
            kwargs["vatt"] = vatt
        if fullscale_current is not None:
            kwargs["fullscale_current"] = fullscale_current
        if rfswitch is not None:
            kwargs["rfswitch"] = rfswitch
        return kwargs if len(kwargs) > 1 else None

    @staticmethod
    def _lookup_channel_dump(
        port_dump: dict[str, Any],
        channel: int,
    ) -> dict[str, Any] | None:
        """Return channel dump by integer or string key."""
        channels = port_dump.get("channels")
        if not isinstance(channels, dict):
            return None
        channel_dump = channels.get(channel)
        if channel_dump is None:
            channel_dump = channels.get(str(channel))
        return channel_dump if isinstance(channel_dump, dict) else None

    @staticmethod
    def _as_optional_float(value: Any) -> float | None:
        """Return one optional numeric value as float."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_optional_int(value: Any) -> int | None:
        """Return one optional numeric value as int."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_optional_string(value: Any) -> str | None:
        """Return one optional value as string."""
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _log_started_frequencies(config: Quel1ContinuousWaveConfig) -> None:
        """Log CW frequency settings and warn about likely IF aliasing."""
        fnco_plus_awg_hz = (
            None
            if config.fnco_freq_hz is None
            else config.fnco_freq_hz + config.awg_freq_hz
        )
        actual_output_freq_ghz = (
            None
            if config.actual_output_freq_hz is None
            else config.actual_output_freq_hz * 1e-9
        )
        logger.info(
            "Continuous wave frequencies: box=%s port=%s channel=%s "
            "lo_freq_hz=%s cnco_freq_hz=%s fnco_freq_hz=%s "
            "awg_freq_hz=%s fnco_plus_awg_hz=%s actual_output_freq_ghz=%s",
            config.box_name,
            config.port,
            config.channel,
            config.lo_freq_hz,
            config.cnco_freq_hz,
            config.fnco_freq_hz,
            config.awg_freq_hz,
            fnco_plus_awg_hz,
            actual_output_freq_ghz,
        )
        if (
            fnco_plus_awg_hz is not None
            and abs(fnco_plus_awg_hz) > CONTINUOUS_WAVE_ALIAS_WARNING_THRESHOLD_HZ
        ):
            logger.warning(
                "Continuous wave FNCO + AWG frequency exceeds %s Hz: "
                "box=%s port=%s channel=%s fnco_plus_awg_hz=%s. "
                "The output may intentionally include aliasing.",
                CONTINUOUS_WAVE_ALIAS_WARNING_THRESHOLD_HZ,
                config.box_name,
                config.port,
                config.channel,
                fnco_plus_awg_hz,
            )

    @staticmethod
    def _resolve_actual_output_frequency_hz(
        *,
        lo_freq_hz: float | None,
        cnco_freq_hz: float | None,
        fnco_freq_hz: float | None,
        awg_freq_hz: float,
        sideband: str | None,
    ) -> float | None:
        """Resolve the expected RF output frequency from available settings."""
        if cnco_freq_hz is None or fnco_freq_hz is None:
            return None
        nco_and_awg_hz = cnco_freq_hz + fnco_freq_hz + awg_freq_hz
        if lo_freq_hz is None and sideband is None:
            return nco_and_awg_hz
        if lo_freq_hz is None or sideband is None:
            return None
        normalized_sideband = sideband.upper()
        if normalized_sideband == "U":
            return lo_freq_hz + nco_and_awg_hz
        if normalized_sideband == "L":
            return lo_freq_hz - nco_and_awg_hz
        return None

    @staticmethod
    def _awg_frequency_to_chunk_cycles(awg_freq_hz: float) -> tuple[int, float]:
        """Convert one AWG frequency to integer cycles per 128 ns chunk."""
        requested = float(awg_freq_hz)
        cycles_per_chunk_float = requested / CONTINUOUS_WAVE_FREQUENCY_UNIT_HZ
        cycles_per_chunk = round(cycles_per_chunk_float)
        allowed_frequency_hz = cycles_per_chunk * CONTINUOUS_WAVE_FREQUENCY_UNIT_HZ
        if not math.isclose(
            requested,
            allowed_frequency_hz,
            rel_tol=0.0,
            abs_tol=FREQUENCY_MULTIPLE_TOLERANCE_HZ,
        ):
            raise ValueError(
                "continuous-wave frequency is generated by repeating one "
                f"{BLOCK_DURATION_NS:g} ns wave chunk, so awg_freq_hz must be "
                f"an integer multiple of {CONTINUOUS_WAVE_FREQUENCY_UNIT_HZ:.0f} Hz. "
                f"requested={requested:.6f} Hz, "
                f"nearest_allowed={allowed_frequency_hz:.6f} Hz, "
                f"cycles_per_chunk={cycles_per_chunk}."
            )
        max_cycles_per_chunk = BLOCK_LENGTH // 2
        sample_rate_hz = 1e9 / SAMPLING_PERIOD_NS
        if not (-max_cycles_per_chunk <= cycles_per_chunk <= max_cycles_per_chunk):
            raise ValueError(
                "awg_freq_hz must be within AWG baseband Nyquist range "
                f"[-{sample_rate_hz / 2:.0f}, {sample_rate_hz / 2:.0f}] Hz. "
                f"requested cycles_per_chunk={cycles_per_chunk}."
            )
        return cycles_per_chunk, allowed_frequency_hz

    @staticmethod
    def _make_iq_chunk(
        *,
        cycles_per_chunk: int,
        amplitude: float,
        phase_rad: float,
    ) -> npt.NDArray[np.complex64]:
        """Build one complex IQ chunk for repeated CW output."""
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("amplitude must satisfy 0.0 <= amplitude <= 1.0")
        n = np.arange(BLOCK_LENGTH, dtype=np.float64)
        phase = 2.0 * np.pi * cycles_per_chunk * n / BLOCK_LENGTH + float(phase_rad)
        iq = float(amplitude) * MAX_DAC_CODE * np.exp(1j * phase)
        return np.asarray(
            np.round(iq.real) + 1j * np.round(iq.imag), dtype=np.complex64
        )

    @staticmethod
    def _waveform_name(
        *,
        box_name: str,
        port: PortType,
        channel: int,
        cycles_per_chunk: int,
    ) -> str:
        """Return deterministic wavedata name for one CW chunk."""
        port_text = (
            str(port)
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )
        cycles_text = (
            f"p{cycles_per_chunk}" if cycles_per_chunk >= 0 else f"m{-cycles_per_chunk}"
        )
        return f"cw_{box_name}_p{port_text}_c{int(channel)}_{cycles_text}cycles"

    @staticmethod
    def _build_awg_param(
        *,
        waveform_name: str,
        chunk_repeats: int,
        awg_repeats: int,
    ) -> Any:
        """Build `quel_ic_config` AWG parameters for repeated CW output."""
        from quel_ic_config import AwgParam, WaveChunk

        return AwgParam(
            num_wait_word=0,
            num_repeat=awg_repeats,
            chunks=[
                WaveChunk(
                    name_of_wavedata=waveform_name,
                    num_blank_word=0,
                    num_repeat=chunk_repeats,
                )
            ],
        )
