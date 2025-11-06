import os
import re
import time
import threading
import random

import numpy as np
import pyautogui
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageOps, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

try:
    from window_helper import WIN_TITLE as DEFAULT_GAME_TITLE
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

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

def _nc(s):
    return s.strip().lower() if isinstance(s, str) else ""

class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', "current.png")
        os.makedirs(os.path.dirname(self.screenshot_path), exist_ok=True)

        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = None

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

        # Game window preview (for debugging/duplication)
        self._gw_warned = False
        self._window_bbox_warned = False
        self._window_preview_name = "Game Window Preview"
        self._window_preview_created = False
        self._window_preview_stop = threading.Event()
        self._window_preview_thread = None
        self._window_preview_size = (0, 0)

        if cv2 is not None:
            self._start_window_preview_thread()

    # ---------------- Roboflow setups ----------------
    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        return InferenceHTTPClient(api_url="http://localhost:9001", api_key=api_key)

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        return InferenceHTTPClient(api_url="http://localhost:9001", api_key=api_key)

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
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected = False

        self._last_hand = []
        self._last_enemy = []
        self._last_elixir = None

        self._last_predictions = []
        self._last_img_size = None

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

    def _start_window_preview_thread(self):
        if cv2 is None or self._window_preview_thread is not None:
            return
        self._window_preview_stop.clear()
        thread = threading.Thread(target=self._window_preview_loop, daemon=True)
        thread.start()
        self._window_preview_thread = thread

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
        blank = np.zeros((240, 320, 3), dtype=np.uint8)

        while not self._window_preview_stop.is_set():
            start = time.time()
            bbox = self._get_game_window_bbox()
            frame = None
            if bbox:
                left, top, width, height = bbox
                capture = self._capture_region(left, top, width, height)
                if capture is not None:
                    frame = cv2.cvtColor(np.array(capture.convert("RGB")), cv2.COLOR_RGB2BGR)
                    frame = self._annotate_window_frame(frame, bbox)
                    if self._window_preview_size != (width, height):
                        try:
                            cv2.resizeWindow(name, width, height)
                        except Exception:
                            pass
                        self._window_preview_size = (width, height)
            if frame is None:
                frame = blank
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
        left, top, _, _ = bbox

        # Draw detections
        field_offset_x = self.actions.TOP_LEFT_X - left
        field_offset_y = self.actions.TOP_LEFT_Y - top
        predictions = self._last_predictions or []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            try:
                cx = float(pred.get("x", 0))
                cy = float(pred.get("y", 0))
                w = float(pred.get("width", 0))
                h = float(pred.get("height", 0))
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            x0 = int(field_offset_x + cx - w / 2)
            y0 = int(field_offset_y + cy - h / 2)
            x1 = int(field_offset_x + cx + w / 2)
            y1 = int(field_offset_y + cy + h / 2)
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            cls = pred.get("class", "unknown")
            score = pred.get("confidence", pred.get("confidence_score", ""))
            label = cls
            if isinstance(score, (int, float)):
                label = f"{cls} ({score:.2f})"
            cls_lower = cls.lower() if isinstance(cls, str) else ""
            if "enemy" in cls_lower:
                color = (48, 48, 230)  # Red-ish (BGR)
            elif "ally" in cls_lower:
                color = (48, 180, 48)  # Green-ish
            else:
                color = (200, 200, 48)  # Yellow-ish
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            text_origin = (x0, max(12, y0 - 6))
            cv2.putText(annotated, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw card bar annotations
        card_left = self.actions.CARD_BAR_X - left
        card_top = self.actions.CARD_BAR_Y - top
        card_w = self.actions.CARD_BAR_WIDTH
        card_h = self.actions.CARD_BAR_HEIGHT
        if card_w > 0 and card_h > 0:
            x0 = max(0, min(width - 1, int(card_left)))
            y0 = max(0, min(height - 1, int(card_top)))
            x1 = max(0, min(width - 1, int(card_left + card_w)))
            y1 = max(0, min(height - 1, int(card_top + card_h)))
            if x1 > x0 and y1 > y0:
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 140, 0), 2)
                slot_width = card_w / max(1, self.num_cards)
                cards = self._last_hand or []
                for idx, card_name in enumerate(cards):
                    sx0 = int(card_left + idx * slot_width)
                    sx1 = int(card_left + (idx + 1) * slot_width)
                    sx0 = max(0, min(width - 1, sx0))
                    sx1 = max(0, min(width - 1, sx1))
                    if sx1 <= sx0:
                        continue
                    cv2.rectangle(annotated, (sx0, y0), (sx1, y1), (255, 200, 0), 1)
                    text = str(card_name)
                    text_x = sx0 + 4
                    text_y = min(height - 5, y1 + 18)
                    cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        info_lines = [
            f"Enemies detected: {len(predictions)}",
            "Hand: " + (", ".join(self._last_hand) if self._last_hand else "unknown"),
        ]
        y = 20
        for line in info_lines:
            cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22

        return annotated

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
            done = True
            reward = self._compute_reward(self._get_state())
            if self.game_over_flag == "victory":
                reward += 100
            elif self.game_over_flag == "defeat":
                reward -= 100
            self.match_over_detected = False
            return self._get_state(), reward, done

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
            next_state = self._get_state()
            return next_state, 0, False

        card_index, x_frac, y_frac = self.available_actions[action_index]

        if card_index != -1 and card_index < len(self.current_cards):
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(1)

        current_enemy_princess_towers = self._count_enemy_princess_towers()
        princess_tower_reward = 0
        if self.prev_enemy_princess_towers is not None and current_enemy_princess_towers < self.prev_enemy_princess_towers:
            princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        reward = self._compute_reward(self._get_state()) + princess_tower_reward
        next_state = self._get_state()
        return next_state, reward, done

    # ---------------- State construction ----------------
    def _get_state(self):
        self.actions.capture_area(self.screenshot_path)
        self._last_elixir = None

        ws = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not ws:
            raise ValueError("WORKSPACE_TROOP_DETECTION env var is not set.")

        results = self.rf_model.run_workflow(
            workspace_name=ws,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )

        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]

        # cache for tower OCR
        self._last_predictions = predictions or []
        try:
            with Image.open(self.screenshot_path) as im:
                self._last_img_size = im.size  # (W, H)
        except Exception:
            self._last_img_size = (self.actions.WIDTH, self.actions.HEIGHT)

        # Build state vectors
        if not predictions:
            ally_flat = [0.0] * (2 * MAX_ALLIES)
            enemy_flat = [0.0] * (2 * MAX_ENEMIES)
            return np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)

        TOWERS = {"ally king tower","ally princess tower","enemy king tower","enemy princess tower"}

        allies, enemies, enemy_names = [], [], []
        for p in predictions:
            if not isinstance(p, dict): 
                continue
            cls_raw = p.get("class",""); cls = _nc(cls_raw)
            if cls in TOWERS: 
                continue
            x = p.get("x"); y = p.get("y")
            if x is None or y is None: 
                continue
            if cls.startswith("ally"):
                allies.append((x,y))
            elif cls.startswith("enemy"):
                enemies.append((x,y))
                enemy_names.append(cls.replace("enemy ","",1) if cls.startswith("enemy ") else (cls_raw or "enemy"))
            else:
                enemies.append((x,y))
                enemy_names.append(cls_raw if cls_raw else "unknown")

        self._last_enemy = enemy_names

        def norm(units): 
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x,y in units]
        def pad(units, n):
            units = norm(units)
            if len(units) < n: 
                units += [(0.0,0.0)] * (n - len(units))
            return units[:n]

        ally_flat  = [c for pos in pad(allies, MAX_ALLIES)  for c in pos]
        enemy_flat = [c for pos in pad(enemies, MAX_ENEMIES) for c in pos]
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
                crop = img.crop(box)
                # save crop for debugging
                stamp = int(time.time() * 1000)
                crop_path = os.path.join(DEBUG_DIR, f"{side}_{slot}_{stamp}.png")
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
            x = p.get("x"); y = p.get("y")
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
            return 0
        elixir = state[0] * 10
        enemy_positions = state[1 + 2 * MAX_ALLIES:]
        enemy_presence = sum(enemy_positions[1::2])
        reward = -enemy_presence
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)
        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence
        return reward

    # ---------------- Hand detection ----------------
    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
            cards = []
            ws = os.getenv('WORKSPACE_CARD_DETECTION')
            if not ws:
                raise ValueError("WORKSPACE_CARD_DETECTION env var is not set.")
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=ws,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                cards.append(predictions[0].get("class", "Unknown") if predictions else "Unknown")
            self._last_hand = list(cards)
            return cards
        except Exception:
            return []

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
        """Safe stub: don't crash if detect_game_end is missing."""
        while not self._endgame_thread_stop.is_set():
            try:
                fn = getattr(self.actions, "detect_game_end", None)
                result = fn() if callable(fn) else None
                if result:
                    self.game_over_flag = result
                    break
            except Exception:
                pass
            time.sleep(0.1)

    # ---------------- Princess towers ----------------
    def _count_enemy_princess_towers(self):
        self.actions.capture_area(self.screenshot_path)
        ws = os.getenv('WORKSPACE_TROOP_DETECTION')
        results = self.rf_model.run_workflow(
            workspace_name=ws,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]
        return sum(1 for p in predictions if isinstance(p, dict) and p.get("class") == "enemy princess tower")

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
