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
        self._next_elixir_debug = 0.0
        self._elixir_window_name = "Elixir OCR"
        self._elixir_window_created = False
        self._elixir_warned = False
        self._gw_warned = False
        self._elixir_bbox_warned = False
        if cv2 is None:
            print("[ClashRoyaleEnv] OpenCV not available; elixir debug window disabled.")
            self._elixir_warned = True

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
        if cv2 is not None and getattr(self, "_elixir_window_created", False):
            try:
                cv2.destroyWindow(self._elixir_window_name)
            except Exception:
                pass
            self._elixir_window_created = False

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
            if not self._elixir_bbox_warned:
                print(f"[ClashRoyaleEnv] Game window containing '{title}' not found; falling back to full screen.")
                self._elixir_bbox_warned = True
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
            if not self._elixir_bbox_warned:
                print("[ClashRoyaleEnv] Game window reported zero size; falling back to full screen.")
                self._elixir_bbox_warned = True
            return None
        self._elixir_bbox_warned = False
        return (left, top, width, height)

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

    def _read_elixir_via_ocr(self):
        bbox = self._get_game_window_bbox()
        if bbox:
            left, top_origin, width, height_full = bbox
        else:
            try:
                screen_width, screen_height = pyautogui.size()
            except Exception:
                screen_width = max(0, getattr(self.actions, "BOTTOM_RIGHT_X", 0))
                screen_height = max(0, getattr(self.actions, "BOTTOM_RIGHT_Y", 0))
            if screen_width <= 0 or screen_height <= 0:
                return None
            left, top_origin, width, height_full = 0, 0, screen_width, screen_height

        capture_height = max(60, int(height_full * 0.25))
        capture_height = min(capture_height, height_full)
        capture_top = top_origin + max(0, height_full - capture_height)

        if width <= 0 or capture_height <= 0:
            return None

        region_left = int(left)
        region_top = int(capture_top)
        region_width = int(width)
        region_height = int(capture_height)

        try:
            grab = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
        except Exception:
            return None

        capture_origin = (region_left, region_top)

        gray = grab.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = gray.filter(ImageFilter.MedianFilter(3))
        thresh = gray.point(lambda p: 255 if p > 140 else 0)

        rgb_np = np.array(grab)
        if rgb_np.ndim == 2:
            rgb_np = np.stack([rgb_np] * 3, axis=-1)
        elif rgb_np.shape[2] == 4:
            rgb_np = rgb_np[:, :, :3]

        if cv2 is not None:
            gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary = cv2.bitwise_not(binary)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            ocr_image = Image.fromarray(binary)
        else:
            gray = grab.convert("L")
            gray = ImageOps.autocontrast(gray, cutoff=1)
            gray = gray.filter(ImageFilter.MedianFilter(3))
            ocr_image = gray.point(lambda p: 255 if p > 135 else 0)

        try:
            data = pytesseract.image_to_data(
                ocr_image,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 -c tessedit_char_whitelist=0123456789",
            )
        except Exception:
            return None

        best = None
        candidates = []
        texts = data.get("text", [])
        tops = data.get("top", [])
        heights = data.get("height", [])
        lefts = data.get("left", [])
        widths = data.get("width", [])
        confs = data.get("conf", [])

        for idx, raw_text in enumerate(texts):
            text = raw_text.strip().replace("O", "0")
            if not text or not text.isdigit():
                continue
            try:
                value = int(text)
            except ValueError:
                continue
            if value < 0 or value > 10:
                continue
            top_y = tops[idx] if idx < len(tops) else 0
            height_val = heights[idx] if idx < len(heights) else 0
            left = lefts[idx] if idx < len(lefts) else 0
            width_val = widths[idx] if idx < len(widths) else 0
            conf_raw = confs[idx] if idx < len(confs) else ""
            try:
                conf_val = float(conf_raw)
            except (TypeError, ValueError):
                conf_val = -1.0
            if conf_val < 45.0:
                continue
            absolute_bottom = capture_origin[1] + top_y + height_val
            width_px = int(width_val)
            height_px = int(height_val)
            if width_px <= 6 or height_px <= 14:
                continue
            if width_px * height_px <= 220:
                continue
            cand = {
                "value": value,
                "left": int(left),
                "top": int(top_y),
                "width": int(width_val),
                "height": int(height_val),
                "abs_bottom": float(absolute_bottom),
                "confidence": str(conf_raw),
                "conf_score": conf_val,
            }
            cand["bottom"] = cand["top"] + max(cand["height"], 1)
            candidates.append(cand)

        if candidates:
            max_bottom = max(c["abs_bottom"] for c in candidates)
            bottom_band = [
                c for c in candidates if (max_bottom - c["abs_bottom"]) <= 40
            ]
            if bottom_band:
                best = max(bottom_band, key=lambda c: (c["conf_score"], c["abs_bottom"]))
            else:
                best = max(candidates, key=lambda c: c["conf_score"])

        try:
            self._show_elixir_debug(grab, candidates, best, capture_origin, (region_width, region_height))
        except Exception:
            pass

        return best["value"] if best else None

    def _show_elixir_debug(self, region_img, candidates, chosen, origin, size):
        if cv2 is None:
            if not getattr(self, "_elixir_warned", False):
                print("[ClashRoyaleEnv] Install opencv-python to enable elixir debug window.")
                self._elixir_warned = True
            return
        now = time.time()
        if now < getattr(self, "_next_elixir_debug", 0.0):
            return

        if not getattr(self, "_elixir_window_created", False):
            try:
                cv2.namedWindow(self._elixir_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._elixir_window_name, 420, 150)
                self._elixir_window_created = True
            except Exception:
                if not self._elixir_warned:
                    print("[ClashRoyaleEnv] Failed to create OpenCV debug window; disabling.")
                    self._elixir_warned = True
                return
        rgb = np.array(region_img.convert("RGB"), dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        h, w = frame_bgr.shape[:2]
        origin_x, origin_y = origin
        region_w, region_h = size
        info_lines = []
        if chosen:
            info_lines.append(
                f"chosen: {chosen['value']} (conf {chosen.get('conf_score', 0):.0f})"
            )
        info_lines.append(
            f"candidates: {len(candidates)}"
        )
        info_lines.append(f"region: {origin_x},{origin_y} {region_w}x{region_h}")

        for cand in candidates:
            left = max(0, cand["left"])
            top = max(0, cand["top"])
            right = min(w - 1, left + max(1, cand["width"]))
            bottom = min(h - 1, top + max(1, cand["height"]))
            if left >= w or top >= h:
                continue
            color = (230, 120, 60) if cand is chosen else (80, 80, 80)
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
            label = f"{cand['value']}"
            if cand.get("conf_score") is not None:
                label = f"{label} ({cand['conf_score']:.0f})"
            cv2.putText(
                frame_bgr,
                label,
                (left, max(12, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if not candidates:
            info_lines.append("no digits found")

        y0 = 18
        for line in info_lines:
            cv2.putText(
                frame_bgr,
                line,
                (8, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                line,
                (8, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y0 += 22

        cv2.imshow(self._elixir_window_name, frame_bgr)
        cv2.waitKey(1)
        self._next_elixir_debug = now + 0.016

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
        elixir = self.actions.count_elixir()
        if getattr(self.actions, "os_type", "") == "Windows":
            ocr_elixir = self._read_elixir_via_ocr()
            if ocr_elixir is not None:
                elixir = ocr_elixir
        self._last_elixir = int(elixir) if elixir is not None else None

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
