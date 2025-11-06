from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Sequence

CLEAR = "\x1b[2J\x1b[H"


def _fmt_list(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return ", ".join(map(str, value))
    return str(value)


class StatusUI:
    """Minimal terminal status widget used during training."""

    def __init__(self, min_interval: float = 0.05) -> None:
        self.last_render = 0.0
        self.min_interval = float(min_interval)
        self.hand: Sequence[str] | None = None
        self.enemy: Sequence[str] | None = None
        self.win = False
        self.play_again = False

    def update(self, *, hand=None, enemy=None, win=False, play_again=False) -> None:
        self.hand = hand if hand is not None else []
        self.enemy = enemy if enemy is not None else []
        self.win = bool(win)
        self.play_again = bool(play_again)
        self.render()

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < self.min_interval:
            return
        self.last_render = now

        hand_str = _fmt_list(self.hand)
        enemy_str = _fmt_list(self.enemy)
        win_str = "yes" if self.win else "none"
        play_again_str = "yes" if self.play_again else "none"

        output = [
            CLEAR,
            "CRBot status",
            "---------------------------",
            f"player hand: {hand_str}",
            f"enemy troops: {enemy_str}",
            f"win screen: {win_str}",
            f"play again btn: {play_again_str}",
            "---------------------------",
        ]
        print("\n".join(output), end="", flush=True)
