from __future__ import annotations

import time
from typing import Tuple

import pyautogui as pag
import pygetwindow as gw

WIN_TITLE = "pyclashbot-96"
TARGET_W = 720  # Fallback width if a window reports 0


def set_window_title(title: str) -> None:
    """Update the cached window title used for lookups."""
    global WIN_TITLE
    if title:
        WIN_TITLE = title


def _find_window(title: str | None = None):
    target = title or WIN_TITLE
    titles = [t for t in gw.getAllTitles() if target.lower() in t.lower()]
    if not titles:
        raise RuntimeError(f"Window containing '{target}' not found")
    return gw.getWindowsWithTitle(titles[0])[0]


def align_and_get_bbox(title: str | None = None) -> Tuple[int, int, int, int]:
    """Align the Clash Royale window to the right edge and return its bounding box."""
    screen_w, screen_h = pag.size()
    win = _find_window(title)

    if win.isMinimized:
        win.restore()
        time.sleep(0.25)

    win.activate()
    time.sleep(0.2)

    width = max(getattr(win, "width", TARGET_W), 1)
    y = 0
    x = max(0, screen_w - width)

    win.moveTo(x, y)
    time.sleep(0.15)

    return win.left, win.top, win.width, win.height
