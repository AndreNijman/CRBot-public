from __future__ import annotations

import os
import re
import time
import threading
import random
from pathlib import Path

import numpy as np
import pyautogui
from dotenv import load_dotenv
import mss
from PIL import Image, ImageOps, ImageFilter
from ultralytics import YOLO

from crbot.automation.controller import ActionController
from crbot.config import DEBUG_DIR, SCREENSHOTS_DIR
from crbot.vision import has_winner_text, screenshot_region

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

try:
    from crbot.utils.window import WIN_TITLE as DEFAULT_GAME_TITLE
except Exception:
    DEFAULT_GAME_TITLE = "pyclashbot-96"

import pytesseract

# Optional: set Tesseract path via env if not on PATH
TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE and os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES  = 10
SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]
MODEL_DIR = Path("yolo_models")
CARDS_MODEL_PATH = (MODEL_DIR / "cards.pt").resolve()
ARENA_MODEL_PATH = Path("runs/arena/train_full_s1280/weights/best.pt").resolve()
IGNORED_MODEL_CLASSES = {"elixir", "clock", "text"}
DECK_HEIGHT_FRACTION = 0.24
WINDOW_NAME = "CR Vision Tester"
KING_TOWER_LABELS = {"king-tower"}
PRINCESS_TOWER_LABELS = {
    "queen-tower",
    "cannoneer-tower",
    "dagger-duchess-tower",
}
ALLY_REGION_MIN = 0.55
ENEMY_REGION_MAX = 0.45
ARENA_IMGSZ = 960
CARDS_IMGSZ = 960
ARENA_CONF = 0.25
CARDS_CONF = 0.25

PRINCESS_TOWER_REWARD = 50.0
PRINCESS_TOWER_PENALTY = 50.0
KING_TOWER_REWARD = 150.0
KING_TOWER_PENALTY = 150.0
TOWER_EVENT_COOLDOWN_MS = 1500
DEFENSE_REWARD_PER_UNIT = 8.0

ENV_DEBUG_DIR = str((DEBUG_DIR / "env").resolve())
os.makedirs(ENV_DEBUG_DIR, exist_ok=True)

def _nc(s):
    return s.strip().lower() if isinstance(s, str) else ""

def _now_ms():
    return int(time.time() * 1000)

class ClashRoyaleEnv:
    def __init__(self):
        self.actions = ActionController()
        self.arena_model = self._load_yolo_model(ARENA_MODEL_PATH, "arena/troops+towers")
        self.card_model = self._load_yolo_model(CARDS_MODEL_PATH, "cards")
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        screenshot_dir = SCREENSHOTS_DIR / "env"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_path = str((screenshot_dir / "current.png").resolve())

        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_enemy_troops = None

        self.match_over_detected = False

        # UI caches
        self._last_hand   = []
        self._last_enemy  = []
        self._last_elixir = None

        # cache preds + image size for tower OCR anchoring
        self._last_predictions = []
        self._last_img_size = None  # (W, H)

        self._emote_interval_range = (15.0, 45.0)
        self._next_emote_time = None
        self._last_reward = 0.0
        self._tower_status = None
        self._tower_events = {}
        self._tower_event_times = {"ally": {"king": None, "princess": None}, "enemy": {"king": None, "princess": None}}
        self._tower_pending = {"ally": {"king": 0, "princess": 0}, "enemy": {"king": 0, "princess": 0}}
        self._current_enemy_troops = 0
        self._last_action_desc = "n/a"
        self._match_outcome = None
        self._last_window_bbox = None
        self._detector_frame_origin = (self.actions.TOP_LEFT_X, self.actions.TOP_LEFT_Y)

        # Game window preview (for debugging/duplication)
        self._gw_warned = False
        self._window_bbox_warned = False
        self._window_preview_name = WINDOW_NAME
        self._window_preview_created = False
        self._window_preview_stop = threading.Event()
        self._window_preview_thread = None
        self._window_preview_size = (0, 0)

        if cv2 is not None:
            self._start_window_preview_thread()

    # ---------------- Roboflow setups ----------------
    def _load_yolo_model(self, path: Path, description: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{description} model not found at {path}")
        try:
            return YOLO(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load {description} model from {path}: {exc}") from exc

    # ---------------- Lifecycle ----------------
    def reset(self):
        setter = getattr(self.actions, "set_input_lock", None)
        if callable(setter):
            try:
                setter(False)
            except Exception:
                pass
        time.sleep(3)
        self.game_over_flag = None
        self._match_outcome = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()

        self.prev_enemy_troops = None
        self.match_over_detected = False

        self._last_hand = []
        self._last_enemy = []
        self._last_elixir = None

        self._last_predictions = []
        self._last_img_size = None

        self._last_reward = 0.0
        self._tower_status = None
        self._tower_events = {}
        self._tower_event_times = {"ally": {"king": None, "princess": None}, "enemy": {"king": None, "princess": None}}
        self._tower_pending = {"ally": {"king": 0, "princess": 0}, "enemy": {"king": 0, "princess": 0}}
        self._current_enemy_troops = 0

        self._schedule_next_emote()

        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()
        if self._window_preview_thread is not None:
            self._window_preview_stop.set()
            self._window_preview_thread.join(timeout=1.0)
            self._window_preview_thread = None
        if cv2 is not None and getattr(self, "_window_preview_created", False):
            try:
                cv2.destroyWindow(self._window_preview_name)
            except Exception:
                pass
            self._window_preview_created = False

    def _get_game_window_bbox(self):
        title = os.getenv("GAME_WINDOW_TITLE", DEFAULT_GAME_TITLE)
        if not title:
            return None
        if gw is None:
            if not self._gw_warned:
                print("[ClashRoyaleEnv] Install pygetwindow to locate game window; falling back to full screen.")
                self._gw_warned = True
            return None
        handles = []
        try:
            handles = gw.getWindowsWithTitle(title) or []
        except Exception:
            handles = []
        if not handles:
            try:
                candidates = [
                    t for t in gw.getAllTitles()
                    if t and title.lower() in t.lower()
                ]
            except Exception:
                candidates = []
            for candidate in candidates:
                try:
                    handles = gw.getWindowsWithTitle(candidate) or []
                except Exception:
                    handles = []
                if handles:
                    break
        if not handles:
            if not self._window_bbox_warned:
                print(f"[ClashRoyaleEnv] Game window containing '{title}' not found; falling back to full screen.")
                self._window_bbox_warned = True
            return None
        win = handles[0]
        if getattr(win, "isMinimized", False):
            try:
                win.restore()
                time.sleep(0.05)
            except Exception:
                pass
        left = max(0, getattr(win, "left", 0))
        top = max(0, getattr(win, "top", 0))
        width = max(0, getattr(win, "width", 0))
        height = max(0, getattr(win, "height", 0))
        if width <= 0 or height <= 0:
            if not self._window_bbox_warned:
                print("[ClashRoyaleEnv] Game window reported zero size; falling back to full screen.")
                self._window_bbox_warned = True
            return None
        self._window_bbox_warned = False
        return (left, top, width, height)

    def _capture_region(self, left, top, width, height):
        if width <= 0 or height <= 0:
            return None
        left = int(left)
        top = int(top)
        width = int(width)
        height = int(height)

        try:
            return pyautogui.screenshot(region=(left, top, width, height))
        except Exception:
            return None

    def _capture_window_frame(self):
        bbox = self._get_game_window_bbox()
        if not bbox:
            return None, None
        left, top, width, height = bbox
        if width <= 0 or height <= 0:
            return None, None
        try:
            with mss.mss() as sct:
                shot = sct.grab({"left": int(left), "top": int(top), "width": int(width), "height": int(height)})
            frame = np.array(shot)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame, bbox
        except Exception:
            pass
        capture = self._capture_region(left, top, width, height)
        if capture is None:
            return None, None
        return cv2.cvtColor(np.array(capture.convert("RGB")), cv2.COLOR_RGB2BGR), bbox

    def _extract_field_from_frame(self, frame_bgr, bbox):
        if frame_bgr is None or bbox is None:
            return None
        left, top, width, height = bbox
        fx1 = int(self.actions.TOP_LEFT_X - left)
        fy1 = int(self.actions.TOP_LEFT_Y - top)
        fx2 = fx1 + int(self.actions.WIDTH)
        fy2 = fy1 + int(self.actions.HEIGHT)
        h, w = frame_bgr.shape[:2]
        fx1 = max(0, min(w, fx1))
        fx2 = max(0, min(w, fx2))
        fy1 = max(0, min(h, fy1))
        fy2 = max(0, min(h, fy2))
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        return frame_bgr[fy1:fy2, fx1:fx2].copy()

    def _start_window_preview_thread(self):
        if cv2 is None or self._window_preview_thread is not None:
            return
        self._window_preview_stop.clear()
        thread = threading.Thread(target=self._window_preview_loop, daemon=True)
        thread.start()
        self._window_preview_thread = thread

    # ---------------- Vision helpers ----------------
    def _capture_and_process_scene(self, *, record_events: bool = True):
        frame_bgr, bbox = self._capture_window_frame()
        origin = (self.actions.TOP_LEFT_X, self.actions.TOP_LEFT_Y)
        frame_for_detector = None
        if frame_bgr is not None and bbox is not None:
            self._last_window_bbox = bbox
            origin = (bbox[0], bbox[1])
            field_crop = self._extract_field_from_frame(frame_bgr, bbox)
            try:
                target = field_crop if field_crop is not None else frame_bgr
                Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB)).save(self.screenshot_path)
            except Exception:
                pass
            frame_for_detector = frame_bgr
        else:
            self._last_window_bbox = None
            try:
                self.actions.capture_area(self.screenshot_path)
                with Image.open(self.screenshot_path).convert("RGB") as frame:
                    frame_for_detector = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            except Exception:
                return [], (self.actions.WIDTH, self.actions.HEIGHT)
            origin = (self.actions.TOP_LEFT_X, self.actions.TOP_LEFT_Y)
            bbox = (origin[0], origin[1], self.actions.WIDTH, self.actions.HEIGHT)
        if frame_for_detector is None:
            return [], (self.actions.WIDTH, self.actions.HEIGHT)
        self._detector_frame_origin = origin
        frame_rgb = cv2.cvtColor(frame_for_detector, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        detections = self._predict_arena(pil_frame, origin)
        self._last_predictions = detections
        self._last_img_size = pil_frame.size
        self._update_tower_status(detections, record_events=record_events)
        return detections, pil_frame.size

    def _predict_arena(self, frame: Image.Image, origin):
        arr = np.array(frame)
        width, height = frame.size
        detections = []
        try:
            results = self.arena_model.predict(
                arr,
                conf=ARENA_CONF,
                imgsz=ARENA_IMGSZ,
                verbose=False,
            )
        except Exception:
            return detections
        r = results[0]
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", {})
        if boxes is None:
            return detections
        for b in boxes:
            cls_id = int(b.cls[0])
            raw_label = names.get(cls_id, str(cls_id))
            raw_lower = raw_label.lower()
            if raw_lower in IGNORED_MODEL_CLASSES:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            x_norm = cx / max(width, 1)
            y_norm = cy / max(height, 1)
            side = self._infer_side(y_norm)
            tower_role = self._tower_role_from_label(raw_lower)
            cls_label = raw_lower
            if tower_role and side in ("ally", "enemy"):
                cls_label = f"{side} {tower_role} tower"
            abs_cx = cx + origin[0]
            abs_cy = cy + origin[1]
            abs_x1 = x1 + origin[0]
            abs_y1 = y1 + origin[1]
            abs_x2 = x2 + origin[0]
            abs_y2 = y2 + origin[1]
            bbox_frame = (x1, y1, x2, y2)
            bbox_abs = (abs_x1, abs_y1, abs_x2, abs_y2)
            width_px = max(1.0, x2 - x1)
            height_px = max(1.0, y2 - y1)
            detections.append(
                {
                    "class": cls_label,
                    "raw_class": raw_lower,
                    "x": cx,
                    "y": cy,
                    "abs_x": abs_cx,
                    "abs_y": abs_cy,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                     "width": width_px,
                     "height": height_px,
                     "bbox_frame": bbox_frame,
                     "bbox_abs": bbox_abs,
                    "score": float(b.conf[0]),
                    "side": side,
                    "tower_role": tower_role,
                }
            )
        return detections

    def _infer_side(self, y_norm: float) -> str | None:
        if y_norm <= ENEMY_REGION_MAX:
            return "enemy"
        if y_norm >= ALLY_REGION_MIN:
            return "ally"
        return None

    def _tower_role_from_label(self, label: str) -> str | None:
        if label in KING_TOWER_LABELS:
            return "king"
        if label in PRINCESS_TOWER_LABELS:
            return "princess"
        return None

    def _update_tower_status(self, detections, *, record_events: bool):
        counts = {
            "ally": {"king": 0, "princess": 0},
            "enemy": {"king": 0, "princess": 0},
        }
        princess_scores = {"ally": [], "enemy": []}
        king_scores = {"ally": [], "enemy": []}
        for det in detections or []:
            role = det.get("tower_role")
            side = det.get("side")
            if role is None or side not in ("ally", "enemy"):
                continue
            if role == "king":
                king_scores[side].append(det)
            else:
                princess_scores[side].append(det)
        for side in ("ally", "enemy"):
            if king_scores[side]:
                counts[side]["king"] = 1
            counts[side]["princess"] = min(2, len(princess_scores[side]))
        if not record_events:
            self._tower_status = counts
            return
        if self._tower_status is None:
            self._tower_status = counts
            self._tower_events = {}
            self._tower_event_times = {"ally": {"king": None, "princess": None}, "enemy": {"king": None, "princess": None}}
            self._tower_pending = {"ally": {"king": 0, "princess": 0}, "enemy": {"king": 0, "princess": 0}}
            return
        prev = self._tower_status
        now = _now_ms()
        events = {
            "enemy_princess_destroyed": 0,
            "enemy_king_destroyed": 0,
            "ally_princess_lost": 0,
            "ally_king_lost": 0,
        }

        for side in ("ally", "enemy"):
            for role in ("king", "princess"):
                prev_count = prev[side][role]
                new_count = counts[side][role]
                pending = self._tower_pending[side][role]
                last_change = self._tower_event_times[side][role]

                if new_count < prev_count:
                    self._tower_pending[side][role] = prev_count - new_count
                    self._tower_event_times[side][role] = now
                elif new_count > prev_count:
                    self._tower_pending[side][role] = 0
                    self._tower_event_times[side][role] = None
                else:
                    if pending > 0 and last_change is not None:
                        if now - last_change >= TOWER_EVENT_COOLDOWN_MS:
                            if side == "enemy" and role == "princess":
                                events["enemy_princess_destroyed"] += pending
                            elif side == "enemy" and role == "king":
                                events["enemy_king_destroyed"] += pending
                            elif side == "ally" and role == "princess":
                                events["ally_princess_lost"] += pending
                            elif side == "ally" and role == "king":
                                events["ally_king_lost"] += pending
                            self._tower_pending[side][role] = 0
                            self._tower_event_times[side][role] = None

        self._tower_events = events
        self._tower_status = counts

    def _determine_outcome_from_towers(self) -> str:
        if self._tower_status is None:
            self._capture_and_process_scene(record_events=False)
        ally = self._tower_status or {"ally": {"king": 0, "princess": 0}, "enemy": {"king": 0, "princess": 0}}
        ally_total = ally["ally"]["king"] + ally["ally"]["princess"]
        enemy_total = ally["enemy"]["king"] + ally["enemy"]["princess"]
        if ally_total > enemy_total:
            return "victory"
        if ally_total < enemy_total:
            return "defeat"
        return "defeat"

    def get_match_outcome(self) -> str | None:
        return self._match_outcome

    def _window_preview_loop(self):
        if cv2 is None:
            return
        name = self._window_preview_name
        if not self._window_preview_created:
            try:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                self._window_preview_created = True
            except Exception:
                print("[ClashRoyaleEnv] Failed to create preview window; disabling.")
                return

        target_dt = 1.0 / 60.0
        max_height = 720
        max_width = 1280
        blank = np.zeros((360, 640, 3), dtype=np.uint8)

        while not self._window_preview_stop.is_set():
            start = time.time()
            frame, bbox = self._capture_window_frame()
            if frame is not None:
                frame = self._annotate_window_frame(frame, bbox)
            if frame is None:
                frame = blank.copy()
            h, w = frame.shape[:2]
            scale = 1.0
            if h > max_height:
                scale = min(scale, max_height / float(h))
            if w > max_width:
                scale = min(scale, max_width / float(w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            if (w, h) != self._window_preview_size:
                try:
                    cv2.resizeWindow(name, w, h)
                except Exception:
                    pass
                self._window_preview_size = (w, h)
            cv2.imshow(name, frame)
            cv2.waitKey(1)
            elapsed = time.time() - start
            sleep = target_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _annotate_window_frame(self, frame_bgr, bbox):
        if cv2 is None or frame_bgr is None or bbox is None:
            return frame_bgr

        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]
        overlay = annotated.copy()
        window_left, window_top, _, _ = bbox

        detections = self._last_predictions or []
        for det in detections:
            bbox_abs = det.get("bbox_abs")
            bbox_frame = det.get("bbox_frame")
            if bbox_abs:
                x1 = int(bbox_abs[0] - window_left)
                y1 = int(bbox_abs[1] - window_top)
                x2 = int(bbox_abs[2] - window_left)
                y2 = int(bbox_abs[3] - window_top)
            elif bbox_frame:
                x1, y1, x2, y2 = map(int, bbox_frame)
            else:
                continue
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cls = det.get("class", "unknown")
            label = det.get("raw_class") or cls
            score = det.get("score")
            if isinstance(score, (int, float)):
                label = f"{label} {score:.2f}"
            color = (60, 200, 200)
            side = det.get("side")
            if det.get("tower_role"):
                color = (0, 200, 255)
            elif side == "ally":
                color = (0, 255, 0)
            elif side == "enemy":
                color = (255, 200, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            self._draw_label_with_bg(annotated, label, (x1 + 4, max(16, y1 - 6)), color, scale=0.55)

        card_left = self.actions.CARD_BAR_X - window_left
        card_top = self.actions.CARD_BAR_Y - window_top
        card_w = self.actions.CARD_BAR_WIDTH
        card_h = self.actions.CARD_BAR_HEIGHT
        if card_w > 0 and card_h > 0:
            x0 = max(0, min(width - 1, int(card_left)))
            y0 = max(0, min(height - 1, int(card_top)))
            x1 = max(0, min(width - 1, int(card_left + card_w)))
            y1 = max(0, min(height - 1, int(card_top + card_h)))
            if x1 > x0 and y1 > y0:
                base_color = (0, 180, 255)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), base_color, 1)
                slot_width = card_w / max(1, self.num_cards)
                cards = self._last_hand or []
                for idx in range(self.num_cards):
                    sx0 = int(card_left + idx * slot_width)
                    sx1 = int(card_left + (idx + 1) * slot_width)
                    sx0 = max(0, min(width - 1, sx0))
                    sx1 = max(0, min(width - 1, sx1))
                    if sx1 <= sx0:
                        continue
                    cv2.rectangle(overlay, (sx0, y0), (sx1, y1), base_color, 1)
                    if idx < len(cards):
                        text = str(cards[idx])
                        text_pos = (sx0 + 4, min(height - 6, y1 - 6))
                        self._draw_label_with_bg(annotated, text, text_pos, base_color, scale=0.45)

        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        ally_towers = 0
        enemy_towers = 0
        if self._tower_status:
            ally = self._tower_status.get("ally", {})
            enemy = self._tower_status.get("enemy", {})
            ally_towers = ally.get("king", 0) + ally.get("princess", 0)
            enemy_towers = enemy.get("king", 0) + enemy.get("princess", 0)
        info_lines = [
            f"Reward: {self._last_reward:.2f}",
            f"Enemy troops: {self._current_enemy_troops}",
            f"Towers A:{ally_towers} E:{enemy_towers}",
            f"Last action: {self._last_action_desc}",
            f"Outcome: {self._match_outcome or 'playing'}",
            "Hand: " + (", ".join(self._last_hand) if self._last_hand else "unknown"),
        ]
        y = 20
        for line in info_lines:
            self._draw_label_with_bg(annotated, line, (10, y), (255, 255, 255))
            y += 22

        return annotated

    def _draw_label_with_bg(self, img, text, org, color, scale=0.55):
        if cv2 is None:
            return
        x, y = org
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        bg_color = (0, 0, 0)
        cv2.rectangle(
            img,
            (x - 4, y - text_size[1] - 6),
            (x + text_size[0] + 4, y + 6),
            bg_color,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness + 1,
            cv2.LINE_AA,
        )

    def _schedule_next_emote(self):
        low, high = self._emote_interval_range
        delay = random.uniform(low, high)
        self._next_emote_time = time.time() + delay

    def _maybe_trigger_emote(self):
        if self.match_over_detected or self.game_over_flag:
            return
        if self._next_emote_time is None:
            self._schedule_next_emote()
            return
        if time.time() < self._next_emote_time:
            return
        checker = getattr(self.actions, "is_input_locked", None)
        if callable(checker):
            try:
                if checker():
                    self._next_emote_time = time.time() + 1.0
                    return
            except Exception:
                pass
        emote = getattr(self.actions, "emote_burst", None)
        if callable(emote):
            try:
                emote()
            except Exception:
                pass
        self._schedule_next_emote()

    # ---------------- Main RL step ----------------
    def step(self, action_index):
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # no-op

        self._maybe_trigger_emote()

        if self.game_over_flag:
            final_state = self._get_state()
            reward = self._compute_reward(final_state)
            self.match_over_detected = False
            return final_state, reward, True

        self.current_cards = self.detect_cards_in_hand()

        if self.current_cards and all(card == "Unknown" for card in self.current_cards):
            locked = False
            checker = getattr(self.actions, "is_input_locked", None)
            if callable(checker):
                try:
                    locked = checker()
                except Exception:
                    locked = False
            if not locked:
                pyautogui.moveTo(1611, 831, duration=0.2)
                pyautogui.click()
            state = self._get_state()
            return state, 0.0, False

        card_index, x_frac, y_frac = self.available_actions[action_index]
        self._last_action_desc = self._describe_action(action_index, card_index, x_frac, y_frac)

        if card_index != -1 and card_index < len(self.current_cards):
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(1)

        next_state = self._get_state()
        reward = self._compute_reward(next_state)
        return next_state, reward, False

    def _describe_action(self, action_index, card_index, x_frac, y_frac):
        if card_index == -1:
            return f"{action_index}: no-op"
        card_name = "Unknown"
        if 0 <= card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
        return f"{action_index}: {card_name} -> ({x_frac:.2f}, {y_frac:.2f})"

    # ---------------- State construction ----------------
    def _get_state(self):
        # TODO: Re-enable elixir detection once the vision pipeline is reliable.
        # try:
        #     elixir = self.actions.count_elixir()
        # except Exception:
        #     elixir = None
        # self._last_elixir = int(elixir) if isinstance(elixir, (int, float)) else None
        self._last_elixir = None
        detections, img_size = self._capture_and_process_scene(record_events=True)
        return self._build_state_from_detections(detections, img_size)

    def _build_state_from_detections(self, detections, img_size):
        width, height = img_size
        allies, enemies = [], []
        enemy_names = []
        enemy_troops = 0
        towers_alias = {"ally king tower", "ally princess tower", "enemy king tower", "enemy princess tower"}
        field_left = self.actions.TOP_LEFT_X
        field_top = self.actions.TOP_LEFT_Y
        field_right = field_left + self.actions.WIDTH
        field_bottom = field_top + self.actions.HEIGHT
        for det in detections or []:
            cls = _nc(det.get("class"))
            if cls in towers_alias:
                continue
            abs_x = det.get("abs_x")
            abs_y = det.get("abs_y")
            if abs_x is None or abs_y is None:
                continue
            if not (field_left <= abs_x <= field_right and field_top <= abs_y <= field_bottom):
                continue
            x_norm = (abs_x - field_left) / float(self.actions.WIDTH)
            y_norm = (abs_y - field_top) / float(self.actions.HEIGHT)
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            side = det.get("side")
            if side == "ally":
                allies.append((x_norm, y_norm))
            else:
                enemies.append((x_norm, y_norm))
                enemy_names.append(det.get("raw_class", cls) or "enemy")
                if side in ("enemy", None):
                    enemy_troops += 1
        self._last_enemy = enemy_names
        self._current_enemy_troops = enemy_troops

        def pad(units, limit):
            units = units[:limit]
            if len(units) < limit:
                units = units + [(0.0, 0.0)] * (limit - len(units))
            return units

        ally_flat = [coord for pair in pad(allies, MAX_ALLIES) for coord in pair]
        enemy_flat = [coord for pair in pad(enemies, MAX_ENEMIES) for coord in pair]
        return np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)

    # ---------------- Big OCR boxes anchored on tower detections ----------------
    def _get_tower_ocr_debug(self):
        """
        Returns all numbers OCR'd from large boxes around each tower area, using
        Roboflow tower detections to anchor boxes when available.

        {
          "ally":  {"princess_left":[...], "king":[...], "princess_right":[...]},
          "enemy": {"princess_left":[...], "king":[...], "princess_right":[...]},
        }
        Saves crops under debug/ for inspection.
        """
        if not os.path.isfile(self.screenshot_path):
            return {"ally": {}, "enemy": {}}

        try:
            img = Image.open(self.screenshot_path).convert("RGB")
        except Exception:
            return {"ally": {}, "enemy": {}}

        out = {"ally": {}, "enemy": {}}

        rois_pred = self._tower_rois_from_predictions(self._last_predictions, img.size)
        rois_fb_enemy = self._fallback_rois(img.size, "enemy")
        rois_fb_ally  = self._fallback_rois(img.size, "ally")

        rois_enemy = rois_pred.get("enemy") or rois_fb_enemy
        rois_ally  = rois_pred.get("ally")  or rois_fb_ally

        for side, rois in (("enemy", rois_enemy), ("ally", rois_ally)):
            for slot, box in rois.items():
                local = self._field_local_box(box, img.size)
                if not local:
                    continue
                crop = img.crop(local)
                # save crop for debugging
                stamp = int(time.time() * 1000)
                crop_path = os.path.join(ENV_DEBUG_DIR, f"{side}_{slot}_{stamp}.png")
                try:
                    crop.save(crop_path)
                except Exception:
                    pass
                nums = self._ocr_all_numbers(crop)
                out[side][slot] = nums

        return out

    def _tower_rois_from_predictions(self, predictions, img_size):
        """
        Build OCR boxes using Roboflow tower detections if present.
        Expect:
          ally/enemy princess tower (two each), ally/enemy king tower (one each).
        """
        W, H = img_size
        L = self.actions.TOP_LEFT_X
        T = self.actions.TOP_LEFT_Y
        FW = self.actions.WIDTH
        FH = self.actions.HEIGHT

        ally_princess, enemy_princess = [], []
        ally_king, enemy_king = [], []

        for p in predictions or []:
            if not isinstance(p, dict):
                continue
            cls = _nc(p.get("class",""))
            x = p.get("abs_x", p.get("x"))
            y = p.get("abs_y", p.get("y"))
            if x is None or y is None:
                continue
            if cls == "ally princess tower":
                ally_princess.append((x, y))
            elif cls == "enemy princess tower":
                enemy_princess.append((x, y))
            elif cls == "ally king tower":
                ally_king.append((x, y))
            elif cls == "enemy king tower":
                enemy_king.append((x, y))

        def box_around(cx, cy, w_frac=0.22, h_frac=0.12, y_shift_px=0):
            bw = max(16, int(w_frac * FW))
            bh = max(14, int(h_frac * FH))
            x1 = int(cx - bw // 2)
            y1 = int(cy - bh // 2 + y_shift_px)
            x2 = x1 + bw
            y2 = y1 + bh
            # clamp to field bbox
            x1 = max(L, x1); y1 = max(T, y1)
            x2 = min(L + FW, x2); y2 = min(T + FH, y2)
            return (x1, y1, x2, y2)

        rois = {"ally": {}, "enemy": {}}

        # Enemy numbers are above bars
        if enemy_king:
            cx, cy = enemy_king[0]
            rois["enemy"]["king"] = box_around(cx, cy, y_shift_px=-int(0.03 * FH))
        if len(enemy_princess) >= 2:
            enemy_princess.sort(key=lambda t: t[0])
            (l_cx, l_cy), (r_cx, r_cy) = enemy_princess[0], enemy_princess[-1]
            rois["enemy"]["princess_left"]  = box_around(l_cx, l_cy, y_shift_px=-int(0.03 * FH))
            rois["enemy"]["princess_right"] = box_around(r_cx, r_cy, y_shift_px=-int(0.03 * FH))

        # Ally numbers are under bars
        if ally_king:
            cx, cy = ally_king[0]
            rois["ally"]["king"] = box_around(cx, cy, y_shift_px=+int(0.03 * FH))
        if len(ally_princess) >= 2:
            ally_princess.sort(key=lambda t: t[0])
            (l_cx, l_cy), (r_cx, r_cy) = ally_princess[0], ally_princess[-1]
            rois["ally"]["princess_left"]  = box_around(l_cx, l_cy, y_shift_px=+int(0.03 * FH))
            rois["ally"]["princess_right"] = box_around(r_cx, r_cy, y_shift_px=+int(0.03 * FH))

        if not rois["ally"]:
            del rois["ally"]
        if not rois["enemy"]:
            del rois["enemy"]
        return rois

    def _fallback_rois(self, img_size, side):
        """
        Wide bands fallback when predictions missing.
        """
        L = self.actions.TOP_LEFT_X
        T = self.actions.TOP_LEFT_Y
        W = self.actions.WIDTH
        H = self.actions.HEIGHT

        left_x1  = L + int(0.08 * W)
        left_x2  = L + int(0.38 * W)
        mid_x1   = L + int(0.38 * W)
        mid_x2   = L + int(0.62 * W)
        right_x1 = L + int(0.62 * W)
        right_x2 = L + int(0.92 * W)

        if side == "enemy":
            y1 = T + int(0.02 * H)
            y2 = T + int(0.24 * H)
        else:
            y1 = T + int(0.76 * H)
            y2 = T + int(0.98 * H)

        return {
            "princess_left":  (left_x1,  y1, left_x2,  y2),
            "king":           (mid_x1,   y1, mid_x2,   y2),
            "princess_right": (right_x1, y1, right_x2, y2),
        }

    def _field_local_box(self, box, img_size):
        if not box:
            return None
        W, H = img_size
        L = self.actions.TOP_LEFT_X
        T = self.actions.TOP_LEFT_Y
        x1, y1, x2, y2 = box
        x1 -= L
        x2 -= L
        y1 -= T
        y2 -= T
        x1 = max(0, min(W, int(x1)))
        x2 = max(0, min(W, int(x2)))
        y1 = max(0, min(H, int(y1)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _ocr_all_numbers(self, crop_img):
        """
        Return every integer detected in the crop using two passes (binarized + gray),
        and two PSMs (7, 13). Upscales first.
        """
        up = crop_img.resize((int(crop_img.width * 2.5), int(crop_img.height * 2.5)), Image.BICUBIC)
        g  = ImageOps.grayscale(up)
        g  = ImageOps.autocontrast(g, cutoff=2)
        g  = g.filter(ImageFilter.MedianFilter(3))

        bin_img = g.point(lambda p: 255 if p > 140 else 0)

        nums = set()
        for src in (bin_img, g):
            for psm in (7, 13):
                try:
                    data = pytesseract.image_to_data(
                        src,
                        config=f"--psm {psm} -l eng -c tessedit_char_whitelist=0123456789",
                        output_type=pytesseract.Output.DICT
                    )
                except Exception:
                    continue
                texts = data.get("text", [])
                confs = data.get("conf", [])
                for t, c in zip(texts, confs):
                    if not t:
                        continue
                    for s in re.findall(r"\d+", t):
                        try:
                            cf = float(c) if not isinstance(c, str) else float(c) if c.strip() else -1.0
                        except:
                            cf = -1.0
                        if cf >= 0:
                            nums.add(int(s))
        return sorted(nums)

    # ---------------- Reward ----------------
    def _compute_reward(self, state):
        if state is None:
            return 0.0
        events = self._tower_events or {}
        reward = 0.0
        reward += events.get("enemy_princess_destroyed", 0) * PRINCESS_TOWER_REWARD
        reward += events.get("enemy_king_destroyed", 0) * KING_TOWER_REWARD
        reward -= events.get("ally_princess_lost", 0) * PRINCESS_TOWER_PENALTY
        reward -= events.get("ally_king_lost", 0) * KING_TOWER_PENALTY

        defense_bonus = 0.0
        if self.prev_enemy_troops is not None and self._current_enemy_troops is not None:
            diff = self.prev_enemy_troops - self._current_enemy_troops
            if diff > 0:
                defense_bonus = diff * DEFENSE_REWARD_PER_UNIT
        reward += defense_bonus

        self.prev_enemy_troops = self._current_enemy_troops
        self._tower_events = {}
        self._last_reward = reward
        return reward

    # ---------------- Hand detection ----------------
    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
        except Exception:
            return []
        detected = []
        for path in card_paths:
            best_label = "Unknown"
            best_conf = -1.0
            try:
                results = self.card_model.predict(
                    str(path),
                    conf=CARDS_CONF,
                    imgsz=CARDS_IMGSZ,
                    verbose=False,
                )
            except Exception:
                detected.append(best_label)
                continue
            r = results[0]
            boxes = getattr(r, "boxes", None)
            names = getattr(r, "names", {})
            if boxes is None:
                detected.append(best_label)
                continue
            for b in boxes:
                cls_id = int(b.cls[0])
                label = names.get(cls_id, str(cls_id))
                if label.lower() in IGNORED_MODEL_CLASSES:
                    continue
                conf = float(b.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_label = label
            detected.append(best_label)
        self._last_hand = list(detected)
        return detected

    # ---------------- Action space ----------------
    def get_available_actions(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op
        return actions

    # ---------------- Endgame watcher ----------------
    def _endgame_watcher(self):
        """Continuously scan for the WINNER banner and infer outcome from tower counts."""
        bbox = None
        while not self._endgame_thread_stop.is_set():
            if bbox is None:
                bbox = self._get_game_window_bbox()
                if bbox is None:
                    time.sleep(0.5)
                    continue
            try:
                shot = screenshot_region(bbox)
            except Exception:
                bbox = None
                time.sleep(0.5)
                continue
            if shot and has_winner_text(shot):
                outcome = self._determine_outcome_from_towers()
                self.game_over_flag = outcome
                self._match_outcome = outcome
                break
            time.sleep(0.3)

    # ---------------- Getters for web UI ----------------
    def get_current_hand(self):
        return list(self._last_hand)

    def get_enemy_detections(self):
        return list(self._last_enemy)

    def get_elixir(self):
        return self._last_elixir

    # tower OCR debug (public)
    def get_tower_ocr_debug(self):
        return self._get_tower_ocr_debug()
