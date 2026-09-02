"""
Run the warm-up characterization campaign for the selected muxes.

Start this script as soon as the fridge warm-up begins (or shortly before).
See ``PLAN.ja.md`` next to this file for the experiment plan. Check the
stored calibrations offline with ``--preflight`` and validate the full
measurement chain on hardware with ``--dry-run`` while the fridge is still
cold.

Muxes listed in ``--forbidden-muxes`` (default: mux 8, which hosts a
concurrent experiment) are never touched: the script refuses to start when
a forbidden mux is selected, when a selected qubit belongs to one, or when
a selected control box is shared with one. Clock re-synchronization on
connect is off unless ``--sync-clocks`` is given.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MUXES = [7, 9]
DEFAULT_FORBIDDEN_MUXES = [8]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the warm-up campaign."""
    parser = argparse.ArgumentParser(
        description="Warm-up characterization campaign for the selected muxes.",
    )
    parser.add_argument(
        "--muxes",
        nargs="+",
        type=int,
        default=DEFAULT_MUXES,
        help=f"Mux indices to use (default: {DEFAULT_MUXES}).",
    )
    parser.add_argument(
        "--forbidden-muxes",
        nargs="*",
        type=int,
        default=DEFAULT_FORBIDDEN_MUXES,
        help=(
            "Mux indices that must never be touched, including via a shared "
            f"control box (default: {DEFAULT_FORBIDDEN_MUXES})."
        ),
    )
    parser.add_argument("--system-id", default=None, help="System ID to load.")
    parser.add_argument("--config-dir", default=None, help="Configuration directory.")
    parser.add_argument("--params-dir", default=None, help="Parameters directory.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Qubit labels to include (default: every qubit in the muxes).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: warmup_data/mux<indices>_<UTC timestamp>).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=12.0,
        help="Campaign duration budget in hours (default: 12).",
    )
    parser.add_argument(
        "--cycle-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between cycle starts (default: 0, back-to-back).",
    )
    parser.add_argument(
        "--thermal-shots",
        type=int,
        default=2**16,
        help="Shots per thermal excitation sequence (default: 65536).",
    )
    parser.add_argument(
        "--electrical-delay",
        type=float,
        default=None,
        help="Fixed electrical delay in ns for reflection scans (default: measure).",
    )
    parser.add_argument(
        "--reflection-width",
        type=float,
        default=None,
        help="Reflection sweep width in GHz (default: library default, 50 MHz).",
    )
    parser.add_argument(
        "--reflection-df",
        type=float,
        default=None,
        help="Reflection sweep step in GHz (default: library default, 0.5 MHz).",
    )
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        default=None,
        help="Warm-up steps to skip (e.g. thermal single_shot).",
    )
    parser.add_argument(
        "--sync-clocks",
        action="store_true",
        help="Re-synchronize clocks of the selected boxes on connect (default: off).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only check mux isolation and stored calibrations offline, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single fast cycle to validate the chain, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the experiment for the selected muxes and run the warm-up campaign."""
    args = parse_args()

    forbidden_muxes = list(args.forbidden_muxes or [])
    selected_forbidden = sorted(set(args.muxes) & set(forbidden_muxes))
    if selected_forbidden:
        raise SystemExit(
            f"ERROR: muxes {selected_forbidden} are forbidden and must not be selected."
        )

    import qubex as qx
    from qubex.contrib import (
        check_mux_isolation,
        plot_warmup_log,
        preflight_check,
        warmup_campaign,
    )
    from qubex.contrib.experiment.warmup_characterization import WARMUP_STEPS

    experiment_kwargs: dict[str, Any] = {"muxes": list(args.muxes)}
    if args.system_id is not None:
        experiment_kwargs["system_id"] = args.system_id
    if args.config_dir is not None:
        experiment_kwargs["config_dir"] = args.config_dir
    if args.params_dir is not None:
        experiment_kwargs["params_dir"] = args.params_dir

    exp = qx.Experiment(**experiment_kwargs)
    print(f"muxes {args.muxes} qubits: {exp.qubit_labels}")
    print(f"selected boxes: {exp.box_ids}")

    if forbidden_muxes:
        isolation = check_mux_isolation(exp, forbidden_muxes)
        if not isolation.data["isolated"]:
            raise SystemExit(
                "ERROR: the selected muxes share qubits "
                f"{isolation.data['shared_qubits']} or boxes "
                f"{isolation.data['shared_boxes']} with forbidden muxes "
                f"{forbidden_muxes}. Refusing to touch hardware."
            )

    if args.targets:
        unknown = [target for target in args.targets if target not in exp.qubit_labels]
        if unknown:
            raise SystemExit(
                f"ERROR: targets {unknown} are not part of muxes {args.muxes}."
            )

    preflight = preflight_check(exp, targets=args.targets)
    if args.preflight:
        return
    if not preflight.data["all_ready"]:
        print(
            "WARNING: some targets lack calibrations for some steps; "
            "those steps will be logged as failed and the campaign continues."
        )

    exp.connect(sync_clocks=args.sync_clocks)
    exp.check_status()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mux_tag = "-".join(str(mux) for mux in args.muxes)
        output_dir = Path("warmup_data") / f"mux{mux_tag}_{stamp}"

    skip = set(args.skip_steps or [])
    steps = [step for step in WARMUP_STEPS if step not in skip]

    campaign_kwargs: dict[str, Any] = {
        "targets": args.targets,
        "output_dir": output_dir,
        "max_duration": args.max_hours * 3600.0,
        "cycle_interval": args.cycle_interval,
        "steps": steps,
        "thermal_shots": args.thermal_shots,
        "electrical_delay": args.electrical_delay,
        "reflection_frequency_width": args.reflection_width,
        "reflection_df": args.reflection_df,
    }
    if args.dry_run:
        campaign_kwargs["max_cycles"] = 1
        campaign_kwargs["thermal_shots"] = min(args.thermal_shots, 4096)

    result = warmup_campaign(exp, **campaign_kwargs)

    print("")
    print(f"stop reason : {result.data['stop_reason']}")
    print(f"cycles      : {result.data['n_cycles']}")
    print(f"output dir  : {result.data['output_dir']}")

    figures = plot_warmup_log(
        output_dir,
        save_dir=output_dir / "figures",
        plot=False,
    )
    print(f"figures     : {sorted(figures)} -> {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
