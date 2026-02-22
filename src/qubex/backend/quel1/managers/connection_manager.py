"""Connection and lifecycle manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import Any, Literal

from qubex.backend.parallel_box_executor import run_parallel_each, run_parallel_map

logger = logging.getLogger(__name__)


class Quel1ConnectionManager:
    """Handle connect/disconnect and box link-maintenance flows for QuEL-1."""

    def connect(
        self,
        *,
        is_connected: bool,
        box_names: str | list[str] | None,
        available_boxes: Callable[[], Collection[str]],
        parallel: bool | None,
        default_parallel_mode: bool,
        create_boxpool: Callable[[list[str], bool], Any],
        create_quel1system_from_boxpool: Callable[[list[str]], Any],
        create_resource_map: Callable[[Literal["cap", "gen"]], dict[str, dict]],
    ) -> tuple[Any, Any, dict[str, dict], dict[str, dict]] | None:
        """
        Resolve and create connected runtime state for requested boxes.

        Parameters
        ----------
        is_connected : bool
            Whether backend resources are already connected.
        box_names : str | list[str] | None
            Target boxes. If None, all available boxes are selected.
        available_boxes : Callable[[], Collection[str]]
            Available box resolver used when `box_names` is None.
        parallel : bool | None
            Parallel creation/reconnect mode override.
        default_parallel_mode : bool
            Default parallel mode used when `parallel` is None.
        create_boxpool : Callable[[list[str], bool], Any]
            Factory to create a connected boxpool.
        create_quel1system_from_boxpool : Callable[[list[str]], Any]
            Factory to create a Quel1System from connected boxes.
        create_resource_map : Callable[[str], dict[str, dict]]
            Factory for resource maps (`"cap"` and `"gen"`).

        Returns
        -------
        tuple[Any, Any, dict[str, dict], dict[str, dict]] | None
            Connected state tuple `(boxpool, quel1system, cap_map, gen_map)`,
            or `None` when already connected.
        """
        if parallel is None:
            parallel = default_parallel_mode
        if is_connected:
            logger.info("Already connected. Skipping backend reconnect.")
            return None
        resolved_box_names: list[str]
        if box_names is None:
            resolved_box_names = list(available_boxes())
        elif isinstance(box_names, str):
            resolved_box_names = [box_names]
        else:
            resolved_box_names = list(box_names)

        boxpool = create_boxpool(resolved_box_names, parallel)
        quel1system = create_quel1system_from_boxpool(resolved_box_names)
        cap_resource_map = create_resource_map("cap")
        gen_resource_map = create_resource_map("gen")
        return boxpool, quel1system, cap_resource_map, gen_resource_map

    def disconnect(
        self,
        *,
        collect_held_resources: Callable[[], list[Any]],
        disconnect_resource_safely: Callable[[Any], None],
    ) -> None:
        """
        Disconnect all currently held resources.

        Parameters
        ----------
        collect_held_resources : Callable[[], list[Any]]
            Resource enumerator.
        disconnect_resource_safely : Callable[[Any], None]
            Safe resource disconnect function.
        """
        for resource in collect_held_resources():
            disconnect_resource_safely(resource)

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
        get_existing_or_create_box: Callable[[str, bool], Any],
        **kwargs: Any,
    ) -> Any:
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
        get_existing_or_create_box : Callable[[str, bool], Any]
            Resolver for existing pooled box or lazily created box.
        **kwargs : Any
            Extra reconnect keyword arguments.

        Returns
        -------
        Any
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
        linkup_box: Callable[[str, int | None], Any],
    ) -> dict[str, Any]:
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
        linkup_box : Callable[[str, int | None], Any]
            Single-box linkup function.

        Returns
        -------
        dict[str, Any]
            Successfully linked boxes keyed by box name.
        """
        unique_box_list = list(dict.fromkeys(box_list))
        if parallel is None:
            parallel = default_parallel_mode
        if not parallel:
            boxes: dict[str, Any] = {}
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
        boxes: dict[str, Any] = {}
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
        get_existing_or_create_box: Callable[[str, bool], Any],
        resolve_config_options: Callable[[str, str], list[Any] | None],
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
        get_existing_or_create_box : Callable[[str, bool], Any]
            Resolver for existing pooled box or lazily created box.
        resolve_config_options : Callable[[str, str], list[Any] | None]
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
        linkup_box: Callable[[str, int | None], Any],
    ) -> Any | None:
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
    def _fallback_linkup_box_result(box_name: str, exc: BaseException) -> Any | None:
        """Log a linkup error and return no box for the failed item."""
        logger.exception(f"{box_name:5} : Error during linkup", exc_info=exc)
        return None

    @staticmethod
    def _log_relinkup_error(box_name: str, exc: BaseException) -> None:
        """Log a relinkup error for one box."""
        logger.exception(f"{box_name:5} : Error during relinkup", exc_info=exc)
