# ui_simple.py
import os
import time

CLEAR = "\x1b[2J\x1b[H"

def _fmt_list(x):
    if not x:
        return "none"
    if isinstance(x, (list, tuple)):
        return ", ".join(map(str, x)) if x else "none"
    return str(x)

class StatusUI:
    def __init__(self):
        self.last_render = 0.0
        self.min_interval = 0.05  # seconds

        self.hand = None
        self.enemy = None
        self.win = False
        self.play_again = False

    def update(self, *, hand=None, enemy=None, win=False, play_again=False):
        self.hand = hand if hand is not None else []
        self.enemy = enemy if enemy is not None else []
        self.win = bool(win)
        self.play_again = bool(play_again)
        self.render()

    def render(self, force=False):
        now = time.time()
        if not force and now - self.last_render < self.min_interval:
            return
        self.last_render = now

        hand_str = _fmt_list(self.hand)
        enemy_str = _fmt_list(self.enemy)
        win_str = "yes" if self.win else "none"
        play_again_str = "yes" if self.play_again else "none"

        out = [
            CLEAR,
            "CRBot status",
            "---------------------------",
            f"player hand: {hand_str}",
            f"enemy troops: {enemy_str}",
            f"win screen: {win_str}",
            f"play again btn: {play_again_str}",
            "---------------------------",
        ]
        # single print call
        print("\n".join(out), end="", flush=True)
