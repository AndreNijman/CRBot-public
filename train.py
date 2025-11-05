import os
import time
import glob
import json
import torch
import threading
from datetime import datetime

from env import ClashRoyaleEnv
from dqn_agent import DQNAgent

from window_helper import align_and_get_bbox
from vision import (
    screenshot_region,
    has_winner_text,
    find_play_again_center,
    assert_tesseract_ready,
)
from logger import write_match
from ui_simple import StatusUI

EPISODES = 10000
BATCH_SIZE = 32
CHECK_INTERVAL = 0.35
PLAY_AGAIN_RETRIES = 8
PLAY_AGAIN_RETRY_DELAY = 0.4
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_latest_model_path(models_dir=MODELS_DIR):
    files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not files:
        return None
    files.sort()
    return files[-1]

def load_latest(agent):
    latest = get_latest_model_path()
    if latest:
        agent.load(os.path.basename(latest))
        meta = latest.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                d = json.load(f)
                agent.epsilon = d.get("epsilon", 1.0)

def save_checkpoint(agent):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"model_{ts}.pth")
    torch.save(agent.model.state_dict(), path)
    with open(os.path.join(MODELS_DIR, f"meta_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({"epsilon": agent.epsilon}, f)

def start_endgame_watcher(bbox, ui: StatusUI):
    stop_evt = threading.Event()
    finished_evt = threading.Event()

    def _watch():
        while not stop_evt.is_set():
            try:
                img = screenshot_region(bbox)
                win = has_winner_text(img)
                if win:
                    finished_evt.set()
                # keep UI showing win flag live
                ui.update(
                    hand=None,  # unchanged here
                    enemy=None,  # unchanged here
                    win=win,
                    play_again=False,  # filled by main loop when scanned
                )
            except Exception:
                # swallow
                pass
            time.sleep(CHECK_INTERVAL)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t, stop_evt, finished_evt

def click_play_again(bbox):
    pos = find_play_again_center(bbox)
    if not pos:
        return False
    try:
        import pyautogui as pag
        pag.moveTo(*pos, duration=0.10)
        pag.click()
        return True
    except Exception:
        return False

def end_episode_cleanly(bbox):
    try:
        import pyautogui as pag
        pag.mouseUp()
    except Exception:
        pass
    for _ in range(PLAY_AGAIN_RETRIES):
        if click_play_again(bbox):
            return True
        time.sleep(PLAY_AGAIN_RETRY_DELAY)
    return False

def _safe_get_hand(env):
    # Try common hooks. Fallback to []
    for name in ("get_current_hand", "get_hand", "current_hand"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return fn() or []
            except Exception:
                return []
    return []

def _safe_get_enemy(env):
    for name in ("get_enemy_detections", "get_enemy_units", "enemy_units"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return fn() or []
            except Exception:
                return []
    return []

def train():
    assert_tesseract_ready()

    ui = StatusUI()
    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    load_latest(agent)

    for ep in range(EPISODES):
        bbox = align_and_get_bbox()
        state = env.reset()
        total_reward = 0.0
        done = False

        watcher, stop_watch, finished = start_endgame_watcher(bbox, ui)
        start_ts = time.time()

        try:
            while not done:
                # pull current detections for UI
                hand = _safe_get_hand(env)
                enemy = _safe_get_enemy(env)

                # scan play-again presence without clicking yet
                play_again_detected = find_play_again_center(bbox) is not None

                # update UI (only)
                ui.update(
                    hand=hand if hand else [],
                    enemy=enemy if enemy else [],
                    win=finished.is_set(),
                    play_again=play_again_detected,
                )

                if finished.is_set():
                    done = True
                    break

                action = agent.act(state)
                next_state, reward, step_done = env.step(action)
                agent.remember(state, action, reward, next_state, step_done)
                agent.replay(BATCH_SIZE)

                state = next_state
                total_reward += reward
                done = step_done
        finally:
            stop_watch.set()
            watcher.join(timeout=1.0)

        # click play again after win screen detected
        end_episode_cleanly(bbox)
        end_ts = time.time()
        write_match(start_ts, end_ts)

        if ep % 10 == 0:
            agent.update_target_model()
            save_checkpoint(agent)

        time.sleep(1.0)

if __name__ == "__main__":
    train()
