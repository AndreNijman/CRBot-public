from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Dict


class TrainingStatus:
    """Thread-safe container shared between training loop and UI."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[str, Any] = {
            "episode": 0,
            "step": 0,
            "epsilon": 1.0,
            "hand": [],
            "enemy": [],
            "elixir": None,
            "win": False,
            "play_again": False,
            "last_action": None,
            "tower_ocr": {
                "ally": {"princess_left": [], "king": [], "princess_right": []},
                "enemy": {"princess_left": [], "king": [], "princess_right": []},
            },
        }

    def set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                self._data[key] = deepcopy(value)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)
