from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

from crbot.config import MODELS_DIR
from crbot.environment import ClashRoyaleEnv
from crbot.monitoring import write_match
from crbot.reinforcement import DQNAgent
from crbot.training.status import TrainingStatus
from crbot.ui.web import create_status_app
from crbot.utils import align_and_get_bbox
from crbot.vision import (
    assert_tesseract_ready,
    find_play_again_center,
    find_start_battle_center,
    has_winner_text,
    screenshot_region,
)


def _start_status_web(status: TrainingStatus, host: str, port: int) -> threading.Thread:
    def _server():
        from logging import getLogger

        getLogger("werkzeug").setLevel("ERROR")
        app = create_status_app(status)
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_server, daemon=True)
    thread.start()
    return thread


def _latest_model_path(models_dir: Path = MODELS_DIR) -> Path | None:
    candidates = sorted(models_dir.glob("model_*.pth"))
    return candidates[-1] if candidates else None


def _load_latest(agent: DQNAgent) -> None:
    latest = _latest_model_path()
    if not latest:
        return
    agent.load(latest)
    meta = latest.with_name(latest.name.replace("model_", "meta_").replace(".pth", ".json"))
    if meta.exists():
        with meta.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))


def _save_checkpoint(agent: DQNAgent) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_path = MODELS_DIR / f"model_{timestamp}.pth"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.model.state_dict(), weight_path)
    with (MODELS_DIR / f"meta_{timestamp}.json").open("w", encoding="utf-8") as handle:
        json.dump({"epsilon": agent.epsilon}, handle)


def _set_env_input_lock(env, locked: bool) -> None:
    actions = getattr(env, "actions", None)
    setter = getattr(actions, "set_input_lock", None)
    if callable(setter):
        try:
            setter(locked)
        except Exception:
            pass


def _safe_call(env, names: Iterable[str]):
    for name in names:
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
    return None


def _start_endgame_watcher(bbox, status: TrainingStatus):
    stop_evt = threading.Event()
    finished_evt = threading.Event()

    def _watch():
        while not stop_evt.is_set():
            try:
                img = screenshot_region(bbox)
                win = has_winner_text(img)
                if win:
                    finished_evt.set()
                current = status.get()
                status.set(win=win, play_again=current.get("play_again", False))
            except Exception:
                pass
            time.sleep(0.2)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    return thread, stop_evt, finished_evt


def _click_play_again(bbox) -> bool:
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


def _click_start_next_battle(bbox) -> bool:
    pos = find_start_battle_center(bbox)
    if not pos:
        return False
    try:
        import pyautogui as pag

        pag.moveTo(*pos, duration=0.10)
        pag.click()
        return True
    except Exception:
        return False


def _end_episode_cleanly(bbox) -> bool:
    try:
        import pyautogui as pag

        pag.mouseUp()
    except Exception:
        pass

    for _ in range(8):
        if _click_play_again(bbox):
            break
        time.sleep(0.35)
    else:
        return False

    time.sleep(5.0)

    for _ in range(8):
        if _click_start_next_battle(bbox):
            return True
        time.sleep(0.35)
    return False


def train_agent(
    episodes: int = 10_000,
    batch_size: int = 32,
    status: TrainingStatus | None = None,
    *,
    web_host: str = "127.0.0.1",
    web_port: int = 5000,
    start_web: bool = True,
) -> None:
    """Entry point mirroring the previous train.py script."""
    assert_tesseract_ready()
    status = status or TrainingStatus()

    web_thread = None
    if start_web:
        web_thread = _start_status_web(status, web_host, web_port)

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    _load_latest(agent)

    for episode in range(episodes):
        bbox = align_and_get_bbox()
        state = env.reset()
        done = False
        step = 0

        status.set(episode=episode + 1, epsilon=getattr(agent, "epsilon", 1.0), step=0)

        watcher, stop_watch, finished = _start_endgame_watcher(bbox, status)
        start_ts = time.time()
        input_locked = False

        try:
            while not done:
                step += 1

                hand = _safe_call(env, ("get_current_hand",))
                enemy = _safe_call(env, ("get_enemy_detections",))
                elixir = _safe_call(env, ("get_elixir",))
                tower_ocr = env.get_tower_ocr_debug()
                play_again_detected = find_play_again_center(bbox) is not None
                outcome = None
                get_outcome = getattr(env, "get_match_outcome", None)
                if callable(get_outcome):
                    outcome = get_outcome()
                win_flag = finished.is_set()
                if outcome:
                    win_flag = outcome == "victory"

                status.set(
                    hand=hand or [],
                    enemy=enemy or [],
                    elixir=elixir,
                    tower_ocr=tower_ocr or status.get().get("tower_ocr"),
                    win=win_flag,
                    play_again=play_again_detected,
                    step=step,
                    epsilon=getattr(agent, "epsilon", 1.0),
                )

                if finished.is_set():
                    if not input_locked:
                        _set_env_input_lock(env, True)
                        input_locked = True
                    done = True
                    break

                action = agent.act(state)
                status.set(last_action=str(action))

                next_state, reward, step_done = env.step(action)
                agent.remember(state, action, reward, next_state, step_done)
                agent.replay(batch_size)

                state = next_state
                done = step_done
        finally:
            stop_watch.set()
            watcher.join(timeout=1.0)

        restart_complete = _end_episode_cleanly(bbox)
        if input_locked:
            if restart_complete:
                _set_env_input_lock(env, False)
            else:
                print("Warning: Restart sequence incomplete; inputs remain locked.")
        end_ts = time.time()
        write_match(start_ts, end_ts)

        if episode % 10 == 0:
            agent.update_target_model()
            _save_checkpoint(agent)

        time.sleep(1.0)

    if web_thread:
        web_thread.join(timeout=0.1)
