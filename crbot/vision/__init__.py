from __future__ import annotations

from .detection import (
    HAS_TESSERACT,
    assert_tesseract_ready,
    find_play_again_center,
    find_start_battle_center,
    has_winner_text,
    screenshot_region,
)

__all__ = [
    "HAS_TESSERACT",
    "assert_tesseract_ready",
    "find_play_again_center",
    "find_start_battle_center",
    "has_winner_text",
    "screenshot_region",
]
