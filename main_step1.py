# main_step1.py
import time
import pyautogui as pag

from window_helper import align_and_get_bbox
from vision import (
    screenshot_region,
    has_winner_text,
    find_play_again_center,
    find_start_battle_center,
)
from logger import write_match

pag.FAILSAFE = True

CHECK_INTERVAL = 0.35  # seconds
PLAY_AGAIN_RETRIES = 8
START_BUTTON_RETRIES = 8
RETRY_DELAY = 0.4
START_BUTTON_DELAY = 5.0

def wait_for_match_end(bbox) -> bool:
    """Poll for 'WINNER' anywhere. Returns True when detected."""
    while True:
        img = screenshot_region(bbox)
        if has_winner_text(img):
            return True
        time.sleep(CHECK_INTERVAL)

def click_play_again(bbox) -> bool:
    """Find play again color and click once. Returns True if clicked."""
    pos = find_play_again_center(bbox)
    if not pos:
        return False
    pag.moveTo(*pos, duration=0.15)
    pag.click()
    return True


def click_start_next_battle(bbox) -> bool:
    """Find the yellow confirmation button and click it."""
    pos = find_start_battle_center(bbox)
    if not pos:
        return False
    pag.moveTo(*pos, duration=0.15)
    pag.click()
    return True

def end_episode_cleanly(bbox):
    # Stop all actions for a short beat
    pag.mouseUp()  # if held
    time.sleep(0.5)
    # Click Play Again (blue button)
    for _ in range(PLAY_AGAIN_RETRIES):
        if click_play_again(bbox):
            break
        time.sleep(RETRY_DELAY)
    else:
        return False

    # Allow queue screen to render before searching for the yellow button
    time.sleep(START_BUTTON_DELAY)

    for _ in range(START_BUTTON_RETRIES):
        if click_start_next_battle(bbox):
            return True
        time.sleep(RETRY_DELAY)
    return False

def run_one_episode():
    bbox = align_and_get_bbox()
    start_ts = time.time()

    # At this point you’ve already queued a battle via your existing flow.
    # We just wait for the game to finish.
    print("Waiting for WINNER...")
    wait_for_match_end(bbox)

    # End episode cleanly
    print("Ending episode. Clicking Play Again...")
    clicked = end_episode_cleanly(bbox)

    end_ts = time.time()
    write_match(start_ts, end_ts)

    print(f"Logged match. Duration {end_ts - start_ts:.1f}s. RestartSequenceComplete={clicked}")

if __name__ == "__main__":
    while True:
        try:
            run_one_episode()
            # small cooldown so the next queue screen appears
            time.sleep(1.0)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(1.0)
