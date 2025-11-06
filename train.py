# train.py (web UI + quiet loop + big tower OCR lists)
import os
import time
import glob
import json
import torch
import threading
from datetime import datetime
from flask import Flask, jsonify, Response

from env import ClashRoyaleEnv
from dqn_agent import DQNAgent

from window_helper import align_and_get_bbox
from vision import (
    screenshot_region,
    has_winner_text,
    find_play_again_center,
    find_start_battle_center,
    assert_tesseract_ready,
)
from logger import write_match

# ---------- config ----------
EPISODES = 10000
BATCH_SIZE = 32
CHECK_INTERVAL = 0.20
PLAY_AGAIN_RETRIES = 8
PLAY_AGAIN_RETRY_DELAY = 0.35
START_BUTTON_RETRIES = 8
START_BUTTON_DELAY = 5.0
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- status ----------
class _Status:
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._data = {
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
                "ally":  {"princess_left": [], "king": [], "princess_right": []},
                "enemy": {"princess_left": [], "king": [], "princess_right": []},
            },
        }

    def set(self, **fields):
        from copy import deepcopy
        with self._lock:
            for k, v in fields.items():
                self._data[k] = deepcopy(v)

    def get(self):
        from copy import deepcopy
        with self._lock:
            return deepcopy(self._data)

STATUS = _Status()

# ---------- web ----------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CRBot status</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    html,body{background:#0b0b0b;color:#eaeaea;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0}
    .wrap{max-width:980px;margin:36px auto;padding:0 16px}
    h1{font-size:24px;margin:0 0 12px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
    .card{background:#151515;border:1px solid #222;border-radius:12px;padding:14px}
    .k{opacity:.7;font-size:12px;margin-bottom:6px}
    .v{font-size:15px;white-space:pre-wrap;word-break:break-word}
    .yes{color:#90ee90} .none{color:#ffadad}
    table{width:100%;border-collapse:collapse;margin-top:6px}
    th,td{border:1px solid #2a2a2a;padding:6px 8px;text-align:center;font-size:14px}
    th{background:#101010}
    footer{opacity:.6;font-size:12px;margin-top:12px}
    code{background:#0f0f0f;border:1px solid #222;border-radius:6px;padding:2px 6px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CRBot status</h1>

    <div class="grid">
      <div class="card"><div class="k">episode</div><div id="episode" class="v">0</div></div>
      <div class="card"><div class="k">step</div><div id="step" class="v">0</div></div>
      <div class="card"><div class="k">epsilon</div><div id="epsilon" class="v">1.0</div></div>
      <div class="card"><div class="k">elixir</div><div id="elixir" class="v">none</div></div>
      <div class="card"><div class="k">win screen</div><div id="win" class="v none">none</div></div>
      <div class="card"><div class="k">play again btn</div><div id="play" class="v none">none</div></div>
      <div class="card" style="grid-column:1/-1"><div class="k">player hand</div><div id="hand" class="v">none</div></div>
      <div class="card" style="grid-column:1/-1"><div class="k">enemy troops</div><div id="enemy" class="v">none</div></div>

      <div class="card" style="grid-column:1/-1">
        <div class="k">tower OCR (all numbers found in big boxes)</div>
        <table>
          <thead>
            <tr><th></th><th>Princess L</th><th>King</th><th>Princess R</th></tr>
          </thead>
          <tbody>
            <tr><th>Enemy</th><td id="e_pl">[]</td><td id="e_k">[]</td><td id="e_pr">[]</td></tr>
            <tr><th>Ally</th><td id="a_pl">[]</td><td id="a_k">[]</td><td id="a_pr">[]</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card" style="grid-column:1/-1"><div class="k">last action</div><div id="last_action" class="v">none</div></div>
    </div>

    <footer>Auto-refreshing. Raw JSON at <code>/status</code>. Crops saved to <code>debug/</code>.</footer>
  </div>

  <script>
    const $ = id => document.getElementById(id);
    const fmtList = x => (!x || x.length===0) ? "[]" : "[" + x.join(", ") + "]";
    const fmt = x => (x===null || x===undefined || x==="" ? "none" : String(x));
    async function tick(){
      try{
        const res = await fetch("/status", {cache:"no-store"});
        if(res.ok){
          const j = await res.json();
          $("episode").textContent = fmt(j.episode);
          $("step").textContent = fmt(j.step);
          $("epsilon").textContent = fmt(j.epsilon);
          $("elixir").textContent = fmt(j.elixir);
          $("hand").textContent = (j.hand && j.hand.length) ? j.hand.join(", ") : "none";
          $("enemy").textContent = (j.enemy && j.enemy.length) ? j.enemy.join(", ") : "none";
          $("win").textContent = j.win ? "yes" : "none";
          $("win").className = "v " + (j.win ? "yes" : "none");
          $("play").textContent = j.play_again ? "yes" : "none";
          $("play").className = "v " + (j.play_again ? "yes" : "none");
          $("last_action").textContent = fmt(j.last_action);

          const t = j.tower_ocr || {ally:{},enemy:{}};
          $("e_pl").textContent = fmtList(t.enemy?.princess_left || []);
          $("e_k").textContent  = fmtList(t.enemy?.king || []);
          $("e_pr").textContent = fmtList(t.enemy?.princess_right || []);
          $("a_pl").textContent = fmtList(t.ally?.princess_left || []);
          $("a_k").textContent  = fmtList(t.ally?.king || []);
          $("a_pr").textContent = fmtList(t.ally?.princess_right || []);
        }
      }catch(e){}
      setTimeout(tick, 60);
    }
    tick();
  </script>
</body>
</html>"""

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.get("/status")
def status():
    return jsonify(STATUS.get())

def _run_web():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# ---------- model io ----------
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

# ---------- watcher ----------
def start_endgame_watcher(bbox):
    stop_evt = threading.Event()
    finished_evt = threading.Event()
    def _watch():
        while not stop_evt.is_set():
            try:
                img = screenshot_region(bbox)
                win = has_winner_text(img)
                if win:
                    finished_evt.set()
                cur = STATUS.get()
                STATUS.set(win=win, play_again=cur.get("play_again", False))
            except Exception:
                pass
            time.sleep(CHECK_INTERVAL)
    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t, stop_evt, finished_evt

# ---------- play again ----------
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


def click_start_next_battle(bbox):
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


def end_episode_cleanly(bbox):
    try:
        import pyautogui as pag
        pag.mouseUp()
    except Exception:
        pass
    for _ in range(PLAY_AGAIN_RETRIES):
        if click_play_again(bbox):
            break
        time.sleep(PLAY_AGAIN_RETRY_DELAY)
    else:
        return False

    time.sleep(START_BUTTON_DELAY)

    for _ in range(START_BUTTON_RETRIES):
        if click_start_next_battle(bbox):
            return True
        time.sleep(PLAY_AGAIN_RETRY_DELAY)
    return False


def _set_env_input_lock(env, locked: bool) -> None:
    actions = getattr(env, "actions", None)
    setter = getattr(actions, "set_input_lock", None)
    if callable(setter):
        try:
            setter(locked)
        except Exception:
            pass

# ---------- safe getter ----------
def _safe(env, names):
    for n in names:
        fn = getattr(env, n, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
    return None

# ---------- main ----------
def train():
    assert_tesseract_ready()

    web_thread = threading.Thread(target=_run_web, daemon=True)
    web_thread.start()

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    load_latest(agent)

    for ep in range(EPISODES):
        bbox = align_and_get_bbox()
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        STATUS.set(episode=ep + 1, epsilon=getattr(agent, "epsilon", 1.0), step=0)

        watcher, stop_watch, finished = start_endgame_watcher(bbox)
        start_ts = time.time()
        endgame_input_locked = False

        try:
            while not done:
                step += 1

                hand   = _safe(env, ("get_current_hand",))
                enemy  = _safe(env, ("get_enemy_detections",))
                elixir = _safe(env, ("get_elixir",))

                # big OCR numbers from tower areas
                tower_ocr = env.get_tower_ocr_debug()

                play_again_detected = find_play_again_center(bbox) is not None

                STATUS.set(
                    hand=hand or [],
                    enemy=enemy or [],
                    elixir=elixir,
                    tower_ocr=tower_ocr or STATUS.get().get("tower_ocr"),
                    win=finished.is_set(),
                    play_again=play_again_detected,
                    step=step,
                    epsilon=getattr(agent, "epsilon", 1.0),
                )

                if finished.is_set():
                    if not endgame_input_locked:
                        _set_env_input_lock(env, True)
                        endgame_input_locked = True
                    done = True
                    break

                action = agent.act(state)
                STATUS.set(last_action=str(action))

                next_state, reward, step_done = env.step(action)
                agent.remember(state, action, reward, next_state, step_done)
                agent.replay(BATCH_SIZE)

                state = next_state
                total_reward += reward
                done = step_done
        finally:
            stop_watch.set()
            watcher.join(timeout=1.0)

        restart_complete = end_episode_cleanly(bbox)
        if endgame_input_locked:
            if restart_complete:
                _set_env_input_lock(env, False)
            else:
                print("Warning: Restart sequence incomplete; inputs remain locked.")
        end_ts = time.time()
        write_match(start_ts, end_ts)

        if ep % 10 == 0:
            agent.update_target_model()
            save_checkpoint(agent)

        time.sleep(1.0)

if __name__ == "__main__":
    train()
