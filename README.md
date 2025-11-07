# CRBot – Clash Royale Reinforcement Learning Bot

CRBot is a Windows-first automation and reinforcement-learning stack that teaches a Clash Royale agent to play directly against live opponents. The project captures the BlueStacks client, performs real-time vision with local Ultralytics models (no Roboflow dependency), and drives actions via PyAutoGUI while a DQN agent learns from dense rewards.

---

## Highlights

- **Local YOLO vision** – separate weights for arena (troops/towers) and card bar detection, shared between the bot and the standalone live-vision viewer.
- **Integrated live HUD** – the training preview window mirrors the manual viewer and overlays detections, reward, tower state, last action, and match outcome.
- **Reward system for real matches** – bonuses for destroying towers, defensive cleanup, and final win/loss signals inferred from the “WINNER” banner.
- **Headless-friendly automations** – automatic match restarts, BlueStacks alignment, and a lightweight Flask status dashboard.
- **Pluggable datasets** – drop in your own YOLO datasets/weights for cards or arena detections without touching the training loop.

---

## Repository Layout

```
crbot/
  automation/           # BlueStacks/pyautogui control layer
  environment/          # ClashRoyaleEnv: RL-friendly wrapper + reward logic
  reinforcement/        # DQN agent and replay buffer
  training/             # Episode loop + status web server
  vision/               # OCR utilities (winner screen, buttons)
  ui/                   # Flask status page
scripts/
  train_bot.py          # CLI entry point for RL training
  live_vision_viewer.py # Standalone vision debugger (same models as env)
yolo_models/            # Checked-in base weights (cards.pt, etc.)
documents/              # Reference material (reward design PDF)
.gitignore              # Not tracked: datasets/, runs/, models/, logs/, screenshots/, debug/, .env
```

> **Heads-up:** Anything listed in `.gitignore` stays local by design. That includes datasets, trained weights, runs, models, logs, debug screenshots, and your `.env`. Plan your own backup strategy for those folders.

---

## Prerequisites

- Windows 10/11 (BlueStacks automation is Windows-only)
- Python 3.12 (with `pip`)
- [BlueStacks Pie 64-bit](https://www.bluestacks.com/bluestacks-5.html) with Clash Royale installed
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (optional but recommended)
- GPU support (CUDA) is helpful but not required; the YOLO models default to device 0
- `pip install -r requirements.txt`

Optional:

- `mss` and OpenCV are already listed in requirements; ensure GPU drivers are up-to-date.
- If you want the legacy Roboflow workflows, keep your `.env`, but the default build no longer depends on them.

---

## Configuration Checklist

1. **Clone the repository**
   ```powershell
   git clone https://github.com/AndreNijman/CRBot-public.git
   cd CRBot-public
   ```
2. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Optional .env**
   - Copy `.env.example` → `.env` if you still use Roboflow or other private keys.
   - Otherwise the modern pipeline runs without any API keys.
4. **Tesseract**
   - Add `TESSERACT_EXE` to your environment if it is not in `PATH`.
5. **BlueStacks setup**
   - Create/launch a Pie 64-bit instance, install Clash Royale, position the window on the right edge, and ensure the title contains `pyclashbot-96` (default expected string).

---

## Vision Models & Datasets

Two weight files drive all detection:

| Purpose | Default Path | Source |
|---------|--------------|--------|
| Cards (deck bar) | `yolo_models/cards.pt` | Any YOLOv8/11 detection model trained on cropped card slots |
| Arena (troops + towers) | `runs/arena/train_full_s1280/weights/best.pt` | YOLO detection model trained on the full battlefield |

### Dataset Structure

1. Place each dataset inside `datasets/` (this folder is ignored by git):
   ```
   datasets/
     cards_custom/
       data.yaml
       images/
         train/
         val/
       labels/
         train/
         val/
     arena_custom/
       data.yaml
       images/...
   ```
2. Author `data.yaml` like any YOLO dataset, e.g.:
   ```yaml
   path: datasets/arena_custom
   train: images/train
   val: images/val
   names:
     0: goblin
     1: king-tower
     ...
   ```
   - For arena models, include tower classes (`king-tower`, `queen-tower`, etc.) so the reward logic can infer tower status.
   - For cards, list every card name exactly as you want it to appear in telemetry.

### Training New Weights

Use Ultralytics (already installed) to train in place:

```powershell
# Arena example
yolo detect train `
  model=yolov8s.pt `
  data=datasets/arena_custom/data.yaml `
  imgsz=1280 `
  epochs=80 `
  batch=4 `
  project=runs/arena `
  name=train_full_s1280 `
  cache=True `
  device=0

# Cards example
yolo detect train `
  model=yolov8s.pt `
  data=datasets/cards_custom/data.yaml `
  imgsz=960 `
  epochs=60 `
  batch=8 `
  project=runs/cards `
  name=deck_v1
```

After training:

1. Copy or point the bot to the `best.pt` you care about.
   - Arena path is configured in `crbot/environment/game_env.py` via `ARENA_MODEL_PATH`.
   - Cards path is `yolo_models/cards.pt`. Replace the file or update the constant.
2. Optionally keep the `runs/` directory under version control in a separate repo; this project ignores it by default.

---

## Live Vision Viewer

Use the same models outside of training to verify detections:

```powershell
python scripts\live_vision_viewer.py
```

- Requires Clash Royale to be visible.
- Press `Q` to exit.
- The script annotates cards (green), troops (orange), and towers (cyan) exactly as the bot sees them, with the same class filters (`clock`, `elixir`, and UI text are suppressed).

---

## Training the Bot

1. Launch BlueStacks and queue into the main battle screen (no pop-ups).
2. Run the trainer:
   ```powershell
   python scripts\train_bot.py --episodes 200 --batch-size 32
   ```
   Common flags:
   - `--episodes N` – number of matches to play (default 10,000).
   - `--batch-size B` – replay batch size (default 32).
   - `--no-web` – disable the status page.
   - `--web-host` / `--web-port` – bind address for the status dashboard (defaults to `127.0.0.1:5000`).
3. Watch the integrated preview window (“CR Vision Tester”) for real-time overlays.
4. Monitor `http://127.0.0.1:5000/` for aggregated stats (episode, epsilon, hand, tower OCR, etc.).

### Outputs (Ignored by Git)

- `logs/` – JSONL match summaries.
- `models/` – DQN checkpoints (`model_*.pth` + `meta_*.json`).
- `runs/` – YOLO training artifacts.
- `screenshots/`, `debug/` – captured frames for OCR / troubleshooting.

Make backups if you need to keep any of these artifacts.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `No compatible parameters found in checkpoint ...` | DQN architecture changed since the checkpoint was created | Delete/rename old files in `models/` or retrain to generate new weights |
| Preview window blank | Clash Royale window title mismatch or BlueStacks minimized | Update `DEFAULT_GAME_TITLE` (in `crbot/utils/window.py`) or bring BlueStacks to foreground |
| `ValueError: Coordinate 'right' is less than 'left'` in tower OCR | Mismatched detections vs. screenshot | Regenerate arena weights with tower classes, ensure BlueStacks resolution matches the default layout |
| Live viewer shows ticks instead of detections | Wrong weights or missing files | Confirm `runs/arena/.../best.pt` and `yolo_models/cards.pt` exist; rerun YOLO training if not |
| Tesseract errors | Missing installation | Install Tesseract and set `TESSERACT_EXE` |

---

## Contributing

1. Fork and create a feature branch (`git checkout -b feature/my-change`).
2. Keep datasets, runs, and local checkpoints out of commits (they are ignored intentionally).
3. Submit a pull request describing your change. When relevant, describe what datasets/weights you used so others can reproduce your setup.

---

## License & Credits

CRBot is MIT-licensed. Clash Royale and associated assets belong to Supercell; use this project at your own risk and respect the game’s Terms of Service.
