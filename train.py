import os
import time
import glob
import json
import torch
import threading
from datetime import datetime

from env import ClashRoyaleEnv
from dqn_agent import DQNAgent

# Autonomy bits
from window_helper import align_and_get_bbox
from vision import (
    screenshot_region,
    has_winner_text,
    find_play_again_center,
    assert_tesseract_ready,
)
from logger import write_match

# ---- config
EPISODES = 10000
BATCH_SIZE = 32
CHECK_INTERVAL = 0.35
PLAY_AGAIN_RETRIES = 8
PLAY_AGAIN_RETRY_DELAY = 0.4
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- model io
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
        print(f"Loaded {os.path.basename(latest)} | epsilon={agent.epsilon}")

def save_checkpoint(agent):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"model_{ts}.pth")
    torch.save(agent.model.state_dict(), path)
    with open(os.path.join(MODELS_DIR, f"meta_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({"epsilon": agent.epsilon}, f)
    print(f"Saved checkpoint -> {path}")

# ---- end-of-game watcher (WINNER OCR)
def start_endgame_watcher(bbox):
    stop_evt = threading.Event()
    finished_evt = threading.Event()

    def _watch():
        while not stop_evt.is_set():
            try:
                img = screenshot_region(bbox)
                if has_winner_text(img):
                    finished_evt.set()
                    return
            except Exception as e:
                # Never kill the thread on OCR issues
                print(f"OCR watch error: {e}")
            time.sleep(CHECK_INTERVAL)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t, stop_evt, finished_evt

# ---- play again click
def click_play_again(bbox):
    pos = find_play_again_center(bbox)
    if not pos:
        return False
    try:
        import pyautogui as pag
        pag.moveTo(*pos, duration=0.15)
        pag.click()
        return True
    except Exception as e:
        print(f"Play Again click error: {e}")
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

# ---- main loop
def train():
    # Fail fast if Tesseract isn’t reachable
    assert_tesseract_ready()

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    load_latest(agent)

    for ep in range(EPISODES):
        bbox = align_and_get_bbox()  # right-edge, full-height (from window_helper)
        state = env.reset()
        total_reward = 0.0
        done = False

        print(f"Episode {ep + 1} | epsilon={agent.epsilon:.3f}")

        watcher, stop_watch, finished = start_endgame_watcher(bbox)
        start_ts = time.time()

        try:
            while not done:
                if finished.is_set():  # game finished by OCR
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

        clicked = end_episode_cleanly(bbox)
        end_ts = time.time()
        write_match(start_ts, end_ts)  # extend later with crowns/hp

        print(f"Episode {ep + 1}: reward={total_reward:.2f} clicked_play_again={clicked}")

        if ep % 10 == 0:
            agent.update_target_model()
            save_checkpoint(agent)

        time.sleep(1.0)

if __name__ == "__main__":
    train()
