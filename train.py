# train.py (web UI + quiet training loop)
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
    assert_tesseract_ready,
)
from logger import write_match

# ---------- config ----------
EPISODES = 10000
BATCH_SIZE = 32
CHECK_INTERVAL = 0.20
PLAY_AGAIN_RETRIES = 8
PLAY_AGAIN_RETRY_DELAY = 0.35
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- in-process status store ----------
class _Status:
    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._data = {
            "hand": [],
            "enemy": [],
            "win": False,
            "play_again": False,
        }

    def set(self, **fields):
        with self._lock:
            self._data.update(fields)

    def get(self):
        with self._lock:
            return dict(self._data)

STATUS = _Status()

# ---------- tiny web ui (Flask) ----------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CRBot status</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    html,body{background:#0b0b0b;color:#eaeaea;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0}
    .wrap{max-width:780px;margin:40px auto;padding:0 16px}
    h1{font-size:24px;margin:0 0 10px}
    .card{background:#151515;border:1px solid #222;border-radius:12px;padding:16px;margin-top:12px}
    .row{display:flex;gap:12px;flex-wrap:wrap}
    .chip{flex:1 1 220px;background:#111;border:1px solid #242424;border-radius:10px;padding:12px}
    .k{opacity:.7;font-size:12px;margin-bottom:6px}
    .v{font-size:16px;white-space:pre-wrap;word-break:break-word}
    .yes{color:#90ee90} .none{color:#ffadad}
    footer{opacity:.6;font-size:12px;margin-top:14px}
    code{background:#0f0f0f;border:1px solid #222;border-radius:6px;padding:2px 6px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CRBot status</h1>

    <div class="card">
      <div class="row">
        <div class="chip">
          <div class="k">player hand</div>
          <div id="hand" class="v">none</div>
        </div>
        <div class="chip">
          <div class="k">enemy troops</div>
          <div id="enemy" class="v">none</div>
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <div class="chip">
          <div class="k">win screen</div>
          <div id="win" class="v none">none</div>
        </div>
        <div class="chip">
          <div class="k">play again btn</div>
          <div id="play" class="v none">none</div>
        </div>
      </div>
      <footer>Auto-refreshing. Open <code>/status</code> for raw JSON.</footer>
    </div>
  </div>

  <script>
    const $ = id => document.getElementById(id);
    function fmt(x){
      if(!x || x.length===0) return "none";
      if(Array.isArray(x)) return x.join(", ");
      return String(x);
    }
    async function tick(){
      try{
        const res = await fetch("/status", {cache:"no-store"});
        if(!res.ok) throw new Error("bad");
        const j = await res.json();
        $("hand").textContent = fmt(j.hand);
        $("enemy").textContent = fmt(j.enemy);

        $("win").textContent = j.win ? "yes" : "none";
        $("win").className = "v " + (j.win ? "yes" : "none");

        $("play").textContent = j.play_again ? "yes" : "none";
        $("play").className = "v " + (j.play_again ? "yes" : "none");
      }catch(e){}
      setTimeout(tick, 200);
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
    # quiet server
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
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
                # keep UI updated
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

# ---------- helpers for pulling detections from env ----------
def _safe_get_hand(env):
    for name in ("get_current_hand", "get_hand", "current_hand"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                v = fn() or []
                return v
            except Exception:
                return []
    return []

def _safe_get_enemy(env):
    for name in ("get_enemy_detections", "get_enemy_units", "enemy_units", "get_visible_enemies"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                v = fn() or []
                return v
            except Exception:
                return []
    return []

# ---------- main ----------
def train():
    assert_tesseract_ready()

    # start web ui
    web_thread = threading.Thread(target=_run_web, daemon=True)
    web_thread.start()

    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    load_latest(agent)

    for _ in range(EPISODES):
        bbox = align_and_get_bbox()
        state = env.reset()
        total_reward = 0.0
        done = False

        watcher, stop_watch, finished = start_endgame_watcher(bbox)
        start_ts = time.time()

        try:
            while not done:
                # poll env to feed UI
                hand = _safe_get_hand(env)
                enemy = _safe_get_enemy(env)

                # detect playagain (don’t click yet)
                play_again_detected = find_play_again_center(bbox) is not None
                STATUS.set(
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

        end_episode_cleanly(bbox)
        end_ts = time.time()
        write_match(start_ts, end_ts)

        if _ % 10 == 0:
            agent.update_target_model()
            save_checkpoint(agent)

        time.sleep(1.0)

if __name__ == "__main__":
    train()
