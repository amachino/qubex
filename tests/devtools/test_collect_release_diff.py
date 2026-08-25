"""Tests for the final release review helper."""

from pathlib import Path
from runpy import run_path
from typing import Any

import pytest


@pytest.fixture
def release_diff_module() -> dict[str, Any]:
    """Release helper should be loadable without running its CLI."""
    script_path = (
        Path(__file__).parents[2]
        / ".agents"
        / "skills"
        / "final-release-review"
        / "scripts"
        / "collect_release_diff.py"
    )
    return run_path(str(script_path))


def test_release_tags_use_version_order(
    release_diff_module: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release tags should select v1.4.8 as the predecessor of v1.4.9."""
    # Arrange
    list_tags = release_diff_module["list_tags"]
    resolve_base_tag = release_diff_module["resolve_base_tag"]

    def fake_run_git(*args: str) -> str:
        if "--sort=-version:refname" in args:
            return "\n".join(
                [
                    "v1.5.0rc2",
                    "v1.5.0rc1",
                    "v1.5.0b4",
                    "v1.4.9",
                    "v1.4.8",
                ]
            )
        return "\n".join(
            [
                "v1.5.0rc2",
                "v1.5.0rc1",
                "v1.4.9",
                "v1.5.0b4",
                "v1.4.8",
            ]
        )

    monkeypatch.setitem(list_tags.__globals__, "run_git", fake_run_git)

    # Act
    base_tag = resolve_base_tag(
        tags=list_tags(),
        merged_tags=["v1.4.9", "v1.4.8"],
        base_tag=None,
        current_tag="v1.4.9",
        target_ref="v1.4.9",
    )

    # Assert
    assert base_tag == "v1.4.8"


def test_untagged_target_uses_latest_merged_release(
    release_diff_module: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untagged target should compare from its latest merged release tag."""
    # Arrange
    resolve_base_tag = release_diff_module["resolve_base_tag"]
    monkeypatch.setitem(
        resolve_base_tag.__globals__,
        "resolve_exact_head_tag",
        lambda target_ref: None,
    )

    # Act
    base_tag = resolve_base_tag(
        tags=["v1.5.0rc2", "v1.5.0rc1", "v1.4.9", "v1.4.8"],
        merged_tags=["v1.4.9", "v1.4.8"],
        base_tag=None,
        current_tag=None,
        target_ref="maintenance-1.4",
    )

    # Assert
    assert base_tag == "v1.4.9"
