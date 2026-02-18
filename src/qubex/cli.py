"""Command-line interface for qubex utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _build_template_payloads(chip_id: str, n_qubits: int) -> dict[Path, dict]:
    """
    Build YAML payloads for an initial qubex configuration tree.

    Parameters
    ----------
    chip_id : str
        Chip identifier used as the top-level YAML key.
    n_qubits : int
        Number of qubits written into `chip.yaml`.

    Returns
    -------
    dict[Path, dict]
        Mapping from relative destination path to YAML payload.
    """
    return {
        Path("config/chip.yaml"): {
            chip_id: {
                "name": "...",
                "n_qubits": n_qubits,
                "backend": "quel1",
            }
        },
        Path("config/box.yaml"): {
            "Q2A": {
                "name": "...",
                "type": "quel1-a",
                "address": "10.0.x.x",
                "adapter": "...",
            }
        },
        Path("config/wiring.yaml"): {
            chip_id: [
                {
                    "mux": 0,
                    "ctrl": ["Q2A-2", "Q2A-4", "Q2A-9", "Q2A-11"],
                    "read_out": "Q2A-1",
                    "read_in": "Q2A-0",
                    "pump": "Q2A-3",
                }
            ]
        },
        Path("config/skew.yaml"): {
            "box_setting": {"Q2A": {"slot": 3, "wait": 0}},
            "monitor_port": "Q2A-5",
            "reference_port": "Q2A-4",
            "scale": {"Q2A-4": 0.125},
            "target_port": {"Q2A-4": None},
            "time_to_start": 4,
            "trigger_nport": 3,
        },
        Path("params/props.yaml"): {
            chip_id: {
                "resonator_frequency": {"Q00": None},
                "qubit_frequency": {"Q00": None},
                "control_frequency": {"Q00": None},
                "qubit_anharmonicity": {"Q00": None},
            }
        },
        Path("params/params.yaml"): {
            chip_id: {
                "control_amplitude": {"Q00": 0.03},
                "readout_amplitude": {"Q00": 0.05},
            }
        },
    }


def _create_config_skeleton(
    *,
    chip_id: str,
    n_qubits: int,
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> int:
    """
    Create an initial YAML configuration skeleton for qubex.

    Parameters
    ----------
    chip_id : str
        Chip identifier used for directory and YAML top-level key.
    n_qubits : int
        Number of qubits for `chip.yaml`.
    output_dir : Path
        Root directory under which `<chip_id>/` is created.
    force : bool
        Whether existing files should be overwritten.
    dry_run : bool
        Whether to only print the files to be generated.

    Returns
    -------
    int
        Process exit code. `0` on success, `1` on error.
    """
    chip_root = output_dir / chip_id
    payloads = _build_template_payloads(chip_id=chip_id, n_qubits=n_qubits)
    destinations = [chip_root / relative_path for relative_path in payloads]
    existing_files = [path for path in destinations if path.exists()]

    if existing_files and not force:
        print(
            "Refusing to overwrite existing files. Use --force to overwrite:",
            file=sys.stderr,
        )
        for existing_path in existing_files:
            print(f"  - {existing_path}", file=sys.stderr)
        return 1

    for relative_path, payload in payloads.items():
        destination = chip_root / relative_path
        if dry_run:
            print(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        destination.write_text(serialized, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for `qubex` CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured root parser.
    """
    parser = argparse.ArgumentParser(prog="qubex")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init-config",
        help="Create YAML template files for qubex configuration.",
    )
    init_parser.add_argument(
        "--chip-id",
        default="64Qv2",
        help="Chip identifier used in directory and YAML keys.",
    )
    init_parser.add_argument(
        "--n-qubits",
        type=int,
        default=64,
        help="Number of qubits written to config/chip.yaml.",
    )
    init_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Root directory where <chip-id>/ is created.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files when present.",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be created without writing them.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run the `qubex` command-line entrypoint.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument vector for testing. If omitted, `sys.argv` is used.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-config":
        return _create_config_skeleton(
            chip_id=args.chip_id,
            n_qubits=args.n_qubits,
            output_dir=args.output_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
