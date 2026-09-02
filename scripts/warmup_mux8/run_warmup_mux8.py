"""
Run the warm-up characterization campaign for mux 8.

Start this script as soon as the fridge warm-up begins (or shortly before).
See ``PLAN.ja.md`` next to this file for the experiment plan. Validate the
full measurement chain in advance with ``--dry-run`` while the fridge is
still cold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MUX = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the mux 8 warm-up campaign."""
    parser = argparse.ArgumentParser(
        description=f"Warm-up characterization campaign for mux {MUX}.",
    )
    parser.add_argument("--system-id", default=None, help="System ID to load.")
    parser.add_argument("--config-dir", default=None, help="Configuration directory.")
    parser.add_argument("--params-dir", default=None, help="Parameters directory.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Qubit labels to include (default: every qubit in the mux).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: warmup_data/mux8_<UTC timestamp>).",
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
        "--skip-steps",
        nargs="*",
        default=None,
        help="Warm-up steps to skip (e.g. thermal single_shot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single fast cycle to validate the chain, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the experiment for mux 8 and run the warm-up campaign."""
    args = parse_args()

    import qubex as qx
    from qubex.contrib import plot_warmup_log, warmup_campaign
    from qubex.contrib.experiment.warmup_characterization import WARMUP_STEPS

    experiment_kwargs: dict[str, Any] = {"muxes": [MUX]}
    if args.system_id is not None:
        experiment_kwargs["system_id"] = args.system_id
    if args.config_dir is not None:
        experiment_kwargs["config_dir"] = args.config_dir
    if args.params_dir is not None:
        experiment_kwargs["params_dir"] = args.params_dir

    exp = qx.Experiment(**experiment_kwargs)
    exp.connect()
    print(f"mux {MUX} qubits: {exp.qubit_labels}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("warmup_data") / f"mux{MUX}_{stamp}"

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
