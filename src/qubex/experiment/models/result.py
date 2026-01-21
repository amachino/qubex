from __future__ import annotations

from collections import UserDict
from datetime import datetime


class Result(UserDict):
    def __init__(
        self,
        data: dict | None = None,
    ) -> None:
        """General result container for experiment calls."""
        super().__init__(data)
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __repr__(self) -> str:
        return f"<Result created_at={self.created_at} data={{...}}>"
