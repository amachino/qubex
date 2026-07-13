"""Collect the release diff against the previous tag."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the previous release tag and print the compare range, "
            "commit log, and diff stat for release review."
        )
    )
    parser.add_argument(
        "--target-ref",
        default="HEAD",
        help="Git ref to compare against the previous release tag.",
    )
    parser.add_argument(
        "--current-tag",
        default=None,
        help="Optional current release tag. If provided, the script resolves the previous tag after it.",
    )
    parser.add_argument(
        "--base-tag",
        default=None,
        help="Optional explicit base tag. Overrides automatic tag resolution.",
    )
    return parser.parse_args()


def run_git(*args: str) -> str:
    """Return stdout for one git command."""
    git_binary = shutil.which("git")
    if git_binary is None:
        raise FileNotFoundError("git executable not found in PATH.")
    process = subprocess.run(  # noqa: S603
        [git_binary, *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    return process.stdout.strip()


def list_tags() -> list[str]:
    """Return tags sorted by creation date descending."""
    output = run_git("tag", "--sort=-creatordate")
    return [line for line in output.splitlines() if line]


def resolve_exact_head_tag(target_ref: str) -> str | None:
    """Return the exact tag on one ref, if any."""
    git_binary = shutil.which("git")
    if git_binary is None:
        raise FileNotFoundError("git executable not found in PATH.")
    process = subprocess.run(  # noqa: S603
        [git_binary, "describe", "--tags", "--exact-match", target_ref],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip() or None


def resolve_base_tag(
    *,
    tags: list[str],
    base_tag: str | None,
    current_tag: str | None,
    target_ref: str,
) -> str:
    """Resolve the previous tag to compare against."""
    if base_tag is not None:
        return base_tag
    if len(tags) == 0:
        raise ValueError("No git tags found.")

    resolved_current_tag = current_tag or resolve_exact_head_tag(target_ref)
    if resolved_current_tag is None:
        return tags[0]
    try:
        current_index = tags.index(resolved_current_tag)
    except ValueError:
        return tags[0]
    previous_index = current_index + 1
    if previous_index >= len(tags):
        raise ValueError(f"No previous tag found after {resolved_current_tag}.")
    return tags[previous_index]


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    tags = list_tags()
    base_tag = resolve_base_tag(
        tags=tags,
        base_tag=args.base_tag,
        current_tag=args.current_tag,
        target_ref=args.target_ref,
    )
    compare_range = f"{base_tag}..{args.target_ref}"
    log_output = run_git("log", "--oneline", compare_range)
    diff_stat_output = run_git("diff", "--stat", compare_range)
    print(f"BASE_TAG={base_tag}")
    print(f"TARGET_REF={args.target_ref}")
    print(f"COMPARE_RANGE={compare_range}")
    print()
    print("LOG:")
    print(log_output or "(no commits)")
    print()
    print("DIFF_STAT:")
    print(diff_stat_output or "(no diff)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
