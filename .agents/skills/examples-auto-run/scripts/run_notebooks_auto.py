"""Run notebooks non-interactively with per-notebook logs and rerun support."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class NotebookRunResult:
    """Store one notebook execution result."""

    notebook_path: Path
    log_path: Path
    command: list[str]
    returncode: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute one or more notebooks with nbconvert, preserve per-notebook "
            "logs, and optionally write a rerun shell script for failures."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Notebook files or directories to scan recursively for .ipynb files.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory for per-notebook logs and executed notebook outputs.",
    )
    parser.add_argument(
        "--main-log",
        type=Path,
        default=None,
        help="Optional main summary log path. Defaults to <logs-dir>/main.log.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Notebook execution timeout in seconds.",
    )
    parser.add_argument(
        "--write-rerun",
        action="store_true",
        help="Write a rerun shell script for failed notebooks.",
    )
    parser.add_argument(
        "--rerun-file",
        type=Path,
        default=None,
        help="Optional rerun shell script path. Defaults to <logs-dir>/rerun_failed.sh.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Notebook path to skip. May be passed multiple times.",
    )
    return parser.parse_args()


def discover_notebooks(paths: list[str], skipped: set[Path]) -> list[Path]:
    """Return sorted notebook paths from file and directory inputs."""
    notebooks: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in skipped:
            continue
        if path.is_dir():
            for notebook in path.rglob("*.ipynb"):
                resolved = notebook.resolve()
                if resolved not in skipped:
                    notebooks.add(resolved)
            continue
        if path.suffix == ".ipynb" and path.exists():
            notebooks.add(path)
    return sorted(notebooks)


def build_log_name(notebook_path: Path) -> str:
    """Return a stable log file name for one notebook."""
    sanitized = "__".join(notebook_path.parts[-4:])
    return f"{sanitized}.log"


def run_notebook(
    notebook_path: Path,
    logs_dir: Path,
    timeout: int,
) -> NotebookRunResult:
    """Execute one notebook and write its log file."""
    outputs_dir = logs_dir / "executed"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / build_log_name(notebook_path)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output-dir",
        str(outputs_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    process = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=os.environ.copy(),
        check=False,
    )
    log_text = "\n".join(
        [
            f"NOTEBOOK: {notebook_path}",
            f"COMMAND: {shlex.join(command)}",
            f"RETURN_CODE: {process.returncode}",
            "",
            "STDOUT:",
            process.stdout,
            "",
            "STDERR:",
            process.stderr,
            "",
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")
    return NotebookRunResult(
        notebook_path=notebook_path,
        log_path=log_path,
        command=command,
        returncode=process.returncode,
    )


def write_main_log(
    results: list[NotebookRunResult],
    skipped: set[Path],
    main_log_path: Path,
) -> None:
    """Write the summary log."""
    lines = []
    if skipped:
        lines.append("SKIPPED:")
        lines.extend(f"- {path}" for path in sorted(skipped))
        lines.append("")
    lines.append("RESULTS:")
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        lines.append(f"- {status} {result.notebook_path} -> {result.log_path}")
    lines.append("")
    failures = [result for result in results if result.returncode != 0]
    lines.append(f"TOTAL={len(results)} FAILURES={len(failures)}")
    main_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rerun_script(
    failures: list[NotebookRunResult],
    logs_dir: Path,
    rerun_path: Path,
    timeout: int,
) -> None:
    """Write a rerun shell script for failed notebooks."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    script_path = Path(__file__).resolve()
    for failure in failures:
        command = [
            "uv",
            "run",
            "python",
            str(script_path),
            str(failure.notebook_path),
            "--logs-dir",
            str(logs_dir),
            "--timeout",
            str(timeout),
        ]
        lines.append(shlex.join(command))
    rerun_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rerun_path.chmod(0o755)


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    logs_dir = args.logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = (
        args.main_log.resolve() if args.main_log is not None else logs_dir / "main.log"
    )
    rerun_path = (
        args.rerun_file.resolve()
        if args.rerun_file is not None
        else logs_dir / "rerun_failed.sh"
    )
    skipped = {Path(path).resolve() for path in args.skip}
    notebooks = discover_notebooks(args.paths, skipped)
    if not notebooks:
        raise SystemExit("No notebooks found for the given inputs.")

    results = [
        run_notebook(path, logs_dir=logs_dir, timeout=args.timeout)
        for path in notebooks
    ]
    write_main_log(results, skipped=skipped, main_log_path=main_log_path)

    failures = [result for result in results if result.returncode != 0]
    if args.write_rerun and failures:
        write_rerun_script(
            failures,
            logs_dir=logs_dir,
            rerun_path=rerun_path,
            timeout=args.timeout,
        )

    print(f"Main log: {main_log_path}")
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{status} {result.notebook_path} -> {result.log_path}")
    if args.write_rerun and failures:
        print(f"Rerun file: {rerun_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
