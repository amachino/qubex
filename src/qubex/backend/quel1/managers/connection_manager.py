"""Connection and lifecycle manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any, Literal

from qubex.backend.parallel_box_executor import run_parallel_each, run_parallel_map
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1ConfigOptionProtocol as Quel1ConfigOption,
        Quel1SystemProtocol as Quel1System,
    )


class Quel1ConnectionManager:
    """Handle connect/disconnect and box link-maintenance flows for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContext) -> None:
        self._runtime_context = runtime_context

    @property
    def is_connected(self) -> bool:
        """Return whether runtime state currently has a connected system."""
        return self._runtime_context.is_connected

    @property
    def boxpool(self) -> BoxPool | None:
        """Return connected boxpool when available."""
        return self._runtime_context.boxpool

    @property
    def quel1system(self) -> Quel1System | None:
        """Return connected Quel1System when available."""
        return self._runtime_context.quel1system

    @property
    def cap_resource_map(self) -> dict[str, dict] | None:
        """Return capture resource map when available."""
        return self._runtime_context.cap_resource_map

    @property
    def gen_resource_map(self) -> dict[str, dict] | None:
        """Return generator resource map when available."""
        return self._runtime_context.gen_resource_map

    def set_connected_state(
        self,
        *,
        boxpool: BoxPool | None,
        quel1system: Quel1System | None,
        cap_resource_map: dict[str, dict] | None,
        gen_resource_map: dict[str, dict] | None,
    ) -> None:
        """
        Replace full connected runtime state.

        Parameters
        ----------
        boxpool : BoxPool | None
            Connected boxpool.
        quel1system : Quel1System | None
            Connected Quel1System.
        cap_resource_map : dict[str, dict] | None
            Capture resource map.
        gen_resource_map : dict[str, dict] | None
            Generator resource map.
        """
        self._runtime_context.set_connected_state(
            boxpool=boxpool,
            quel1system=quel1system,
            cap_resource_map=cap_resource_map,
            gen_resource_map=gen_resource_map,
        )

    def set_boxpool(self, boxpool: BoxPool | None) -> None:
        """Update only boxpool state."""
        self._runtime_context.set_boxpool(boxpool)

    def set_quel1system(self, quel1system: Quel1System | None) -> None:
        """Update only Quel1System state."""
        self._runtime_context.set_quel1system(quel1system)

    def set_cap_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only capture resource map state."""
        self._runtime_context.set_cap_resource_map(resource_map)

    def set_gen_resource_map(self, resource_map: dict[str, dict] | None) -> None:
        """Update only generator resource map state."""
        self._runtime_context.set_gen_resource_map(resource_map)

    def clear_connected_state(self) -> None:
        """Clear connected runtime state."""
        self._runtime_context.clear_connected_state()

    def connect(
        self,
        *,
        box_names: str | list[str] | None,
        available_boxes: Callable[[], Collection[str]],
        parallel: bool | None,
        default_parallel_mode: bool,
        create_boxpool: Callable[[list[str], bool], BoxPool],
        create_quel1system_from_boxpool: Callable[[list[str]], Quel1System],
        create_resource_map: Callable[[Literal["cap", "gen"]], dict[str, dict]],
    ) -> None:
        """
        Resolve and create connected runtime state for requested boxes.

        Parameters
        ----------
        box_names : str | list[str] | None
            Target boxes. If None, all available boxes are selected.
        available_boxes : Callable[[], Collection[str]]
            Available box resolver used when `box_names` is None.
        parallel : bool | None
            Parallel creation/reconnect mode override.
        default_parallel_mode : bool
            Default parallel mode used when `parallel` is None.
        create_boxpool : Callable[[list[str], bool], BoxPool]
            Factory to create a connected boxpool.
        create_quel1system_from_boxpool : Callable[[list[str]], Quel1System]
            Factory to create a Quel1System from connected boxes.
        create_resource_map : Callable[[str], dict[str, dict]]
            Factory for resource maps (`"cap"` and `"gen"`).
        """
        if parallel is None:
            parallel = default_parallel_mode
        if self.is_connected:
            logger.info("Already connected. Skipping backend reconnect.")
            return
        resolved_box_names: list[str]
        if box_names is None:
            resolved_box_names = list(available_boxes())
        elif isinstance(box_names, str):
            resolved_box_names = [box_names]
        else:
            resolved_box_names = list(box_names)

        boxpool = create_boxpool(resolved_box_names, parallel)
        self.set_connected_state(
            boxpool=boxpool,
            quel1system=None,
            cap_resource_map=None,
            gen_resource_map=None,
        )
        try:
            quel1system = create_quel1system_from_boxpool(resolved_box_names)
            self.set_quel1system(quel1system)
            cap_resource_map = create_resource_map("cap")
            gen_resource_map = create_resource_map("gen")
        except Exception:
            self.clear_connected_state()
            raise
        self.set_cap_resource_map(cap_resource_map)
        self.set_gen_resource_map(gen_resource_map)

    def disconnect(
        self,
        *,
        collect_held_resources: Callable[[], list[object]],
        disconnect_resource_safely: Callable[[object], None],
    ) -> None:
        """
        Disconnect all currently held resources.

        Parameters
        ----------
        collect_held_resources : Callable[[], list[object]]
            Resource enumerator.
        disconnect_resource_safely : Callable[[object], None]
            Safe resource disconnect function.
        """
        for resource in collect_held_resources():
            disconnect_resource_safely(resource)
        self.clear_connected_state()

    def initialize_awg_and_capunits(
        self,
        *,
        is_connected: bool,
        box_names: str | Collection[str],
        parallel: bool | None,
        default_parallel_mode: bool,
        max_parallel_workers: int,
        initialize_box_awg_and_capunits: Callable[[str], None],
    ) -> None:
        """
        Initialize AWG and capture units for selected boxes.

        Parameters
        ----------
        is_connected : bool
            Whether backend resources are connected.
        box_names : str | Collection[str]
            Target boxes.
        parallel : bool | None
            Parallel initialization mode override.
        default_parallel_mode : bool
            Default parallel mode used when `parallel` is None.
        max_parallel_workers : int
            Maximum worker count for parallel initialization.
        initialize_box_awg_and_capunits : Callable[[str], None]
            Box-level initialization function.
        """
        if not is_connected:
            raise ValueError("Boxes not connected. Call connect() method first.")
        if isinstance(box_names, str):
            box_name_list = [box_names]
        else:
            box_name_list = list(box_names)
        unique_box_names = list(dict.fromkeys(box_name_list))
        if parallel is None:
            parallel = default_parallel_mode
        if not parallel:
            for box_name in unique_box_names:
                initialize_box_awg_and_capunits(box_name)
            return
        run_parallel_each(
            unique_box_names,
            initialize_box_awg_and_capunits,
            max_workers=max_parallel_workers,
        )

    def linkup(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
        relaxed_noise_threshold: int,
        check_box_availability: Callable[[str], None],
        get_existing_or_create_box: Callable[[str, bool], Quel1Box],
        **kwargs: Any,
    ) -> Quel1Box:
        """
        Linkup one box and return the connected box.

        Parameters
        ----------
        box_name : str
            Box name.
        noise_threshold : int | None
            Noise threshold override.
        relaxed_noise_threshold : int
            Default threshold when `noise_threshold` is None.
        check_box_availability : Callable[[str], None]
            Box availability validator.
        get_existing_or_create_box : Callable[[str, bool], Quel1Box]
            Resolver for existing pooled box or lazily created box.
        **kwargs : Any
            Extra reconnect keyword arguments.

        Returns
        -------
        Quel1Box
            Linked box object.
        """
        check_box_availability(box_name)
        box = get_existing_or_create_box(box_name, False)
        if noise_threshold is None:
            noise_threshold = relaxed_noise_threshold
        if not all(box.link_status().values()):
            raise ConnectionError(f"Box {box_name} has down links before linkup.")
        box.reconnect(background_noise_threshold=noise_threshold, **kwargs)
        status = box.link_status()
        if not all(status.values()):
            logger.warning(f"Failed to linkup box {box_name}. Status: {status}")
        return box

    def linkup_boxes(
        self,
        *,
        box_list: list[str],
        noise_threshold: int | None,
        parallel: bool | None,
        default_parallel_mode: bool,
        max_parallel_workers: int,
        linkup_box: Callable[[str, int | None], Quel1Box | None],
    ) -> dict[str, Quel1Box]:
        """
        Linkup all requested boxes.

        Parameters
        ----------
        box_list : list[str]
            Box names.
        noise_threshold : int | None
            Noise threshold override.
        parallel : bool | None
            Parallel execution mode override.
        default_parallel_mode : bool
            Default parallel mode used when `parallel` is None.
        max_parallel_workers : int
            Maximum worker count for parallel linkup.
        linkup_box : Callable[[str, int | None], Quel1Box | None]
            Single-box linkup function.

        Returns
        -------
        dict[str, Quel1Box]
            Successfully linked boxes keyed by box name.
        """
        unique_box_list = list(dict.fromkeys(box_list))
        if parallel is None:
            parallel = default_parallel_mode
        if not parallel:
            boxes: dict[str, Quel1Box] = {}
            for box_name in unique_box_list:
                linked_box = self._safe_linkup_box(
                    box_name=box_name,
                    noise_threshold=noise_threshold,
                    linkup_box=linkup_box,
                )
                if linked_box is not None:
                    boxes[box_name] = linked_box
            return boxes

        results = run_parallel_map(
            unique_box_list,
            lambda box_name: linkup_box(box_name, noise_threshold),
            key=lambda box_name: box_name,
            max_workers=max_parallel_workers,
            on_error=self._fallback_linkup_box_result,
        )
        boxes: dict[str, Quel1Box] = {}
        for box_name, linked_box in results.items():
            if linked_box is None:
                continue
            boxes[box_name] = linked_box
            logger.info(f"{box_name:5} : Linked up")
        return boxes

    def relinkup(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
        relaxed_noise_threshold: int,
        check_box_availability: Callable[[str], None],
        get_existing_or_create_box: Callable[[str, bool], Quel1Box],
        resolve_config_options: Callable[[str, str], list[Quel1ConfigOption] | None],
    ) -> None:
        """
        Relink one box.

        Parameters
        ----------
        box_name : str
            Box name.
        noise_threshold : int | None
            Noise threshold override.
        relaxed_noise_threshold : int
            Default threshold when `noise_threshold` is None.
        check_box_availability : Callable[[str], None]
            Box availability validator.
        get_existing_or_create_box : Callable[[str, bool], Quel1Box]
            Resolver for existing pooled box or lazily created box.
        resolve_config_options : Callable[[str, str], list[Quel1ConfigOption] | None]
            Per-box config option resolver.
        """
        check_box_availability(box_name)
        if noise_threshold is None:
            noise_threshold = relaxed_noise_threshold
        box = get_existing_or_create_box(box_name, False)
        config_options = resolve_config_options(box_name, box.boxtype)
        box.relinkup(
            use_204b=False,
            background_noise_threshold=noise_threshold,
            config_options=config_options,
        )
        box.reconnect(background_noise_threshold=noise_threshold)

    def relinkup_boxes(
        self,
        *,
        box_list: list[str],
        noise_threshold: int | None,
        parallel: bool | None,
        default_parallel_mode: bool,
        max_parallel_workers: int,
        relinkup_box: Callable[[str, int | None], None],
    ) -> None:
        """
        Relink all requested boxes.

        Parameters
        ----------
        box_list : list[str]
            Box names.
        noise_threshold : int | None
            Noise threshold override.
        parallel : bool | None
            Parallel execution mode override.
        default_parallel_mode : bool
            Default parallel mode used when `parallel` is None.
        max_parallel_workers : int
            Maximum worker count for parallel relinkup.
        relinkup_box : Callable[[str, int | None], None]
            Single-box relinkup function.
        """
        unique_box_list = list(dict.fromkeys(box_list))
        if parallel is None:
            parallel = default_parallel_mode
        if not parallel:
            for box_name in unique_box_list:
                relinkup_box(box_name, noise_threshold)
            return
        run_parallel_each(
            unique_box_list,
            lambda box_name: relinkup_box(box_name, noise_threshold),
            max_workers=max_parallel_workers,
            on_error=self._log_relinkup_error,
        )

    def _safe_linkup_box(
        self,
        *,
        box_name: str,
        noise_threshold: int | None,
        linkup_box: Callable[[str, int | None], Quel1Box | None],
    ) -> Quel1Box | None:
        """Link up one box and log failures without raising."""
        try:
            linked_box = linkup_box(box_name, noise_threshold)
            logger.info(f"{box_name:5} : Linked up")
        except Exception as exc:
            logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
            return None
        else:
            return linked_box

    @staticmethod
    def _fallback_linkup_box_result(
        box_name: str, exc: BaseException
    ) -> Quel1Box | None:
        """Log a linkup error and return no box for the failed item."""
        logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
        return None

    @staticmethod
    def _log_relinkup_error(box_name: str, exc: BaseException) -> None:
        """Log a relinkup error for one box."""
        logger.exception(f"{box_name:5} : Error during relinkup", exc_info=exc)
