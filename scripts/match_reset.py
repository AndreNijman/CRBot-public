from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import pyautogui as pag

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crbot.monitoring import write_match
from crbot.utils import align_and_get_bbox
from crbot.vision import (
    find_play_again_center,
    find_start_battle_center,
    has_winner_text,
    screenshot_region,
)

pag.FAILSAFE = True

CHECK_INTERVAL = 0.35
PLAY_AGAIN_RETRIES = 8
START_BUTTON_RETRIES = 8
RETRY_DELAY = 0.4
START_BUTTON_DELAY = 5.0


def wait_for_match_end(bbox) -> bool:
    """Poll for 'WINNER' anywhere on screen. Returns True when detected."""
    while True:
        img = screenshot_region(bbox)
        if has_winner_text(img):
            return True
        time.sleep(CHECK_INTERVAL)


def click_play_again(bbox) -> bool:
    pos = find_play_again_center(bbox)
    if not pos:
        return False
    pag.moveTo(*pos, duration=0.15)
    pag.click()
    return True


def click_start_next_battle(bbox) -> bool:
    pos = find_start_battle_center(bbox)
    if not pos:
        return False
    pag.moveTo(*pos, duration=0.15)
    pag.click()
    return True


def end_episode_cleanly(bbox) -> bool:
    pag.mouseUp()
    time.sleep(0.5)
    for _ in range(PLAY_AGAIN_RETRIES):
        if click_play_again(bbox):
            break
        time.sleep(RETRY_DELAY)
    else:
        return False

    time.sleep(START_BUTTON_DELAY)
    for _ in range(START_BUTTON_RETRIES):
        if click_start_next_battle(bbox):
            return True
        time.sleep(RETRY_DELAY)
    return False


def run_one_episode(verbose: bool = True) -> None:
    bbox = align_and_get_bbox()
    start_ts = time.time()

    if verbose:
        print("Waiting for WINNER...")
    wait_for_match_end(bbox)

    if verbose:
        print("Ending episode. Clicking Play Again...")
    clicked = end_episode_cleanly(bbox)

    end_ts = time.time()
    write_match(start_ts, end_ts)

    if verbose:
        print(f"Logged match. Duration {end_ts - start_ts:.1f}s. RestartSequenceComplete={clicked}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor for match completion and requeue automatically.")
    parser.add_argument("--loop", action="store_true", help="Run until interrupted.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.loop:
        while True:
            try:
                run_one_episode()
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print("Error:", exc)
                time.sleep(1.0)
    else:
        run_one_episode()


if __name__ == "__main__":
    main()
