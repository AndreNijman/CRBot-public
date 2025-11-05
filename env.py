import re
import numpy as np
import time
import os
import pyautogui
import threading
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# Optional: set Tesseract exe via env if not on PATH
TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE and os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES  = 10
SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

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

        # Tower HP caches (current nums and max nums detected this match)
        self._tower_hp_curr = {
            "ally":  {"king": None, "princess_left": None, "princess_right": None},
            "enemy": {"king": None, "princess_left": None, "princess_right": None},
        }
        self._tower_hp_max = {
            "ally":  {"king": None, "princess_left": None, "princess_right": None},
            "enemy": {"king": None, "princess_left": None, "princess_right": None},
        }
        self._tower_max_initialized = False

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

        for side in ("ally","enemy"):
            for slot in ("king","princess_left","princess_right"):
                self._tower_hp_curr[side][slot] = None
                self._tower_hp_max[side][slot] = None
        self._tower_max_initialized = False

        # Try to capture initial max HPs at game start
        self._prime_tower_max_hp()
        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    # ---------------- Main RL step ----------------
    def step(self, action_index):
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # no-op

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
        # Always refresh frame and attempt OCR so UI keeps updating
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()
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

        # Update tower HP every state tick (uses preds or fallback ROIs)
        try:
            self._update_tower_hp_from_ocr(predictions)
        except Exception:
            pass

        if not predictions:
            ally_flat = [0.0] * (2 * MAX_ALLIES)
            enemy_flat = [0.0] * (2 * MAX_ENEMIES)
            return np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)

        def nc(s): return s.strip().lower() if isinstance(s, str) else ""
        TOWERS = {"ally king tower","ally princess tower","enemy king tower","enemy princess tower"}

        allies, enemies, enemy_names = [], [], []
        for p in predictions:
            if not isinstance(p, dict): continue
            cls_raw = p.get("class",""); cls = nc(cls_raw)
            if cls in TOWERS: continue
            x = p.get("x"); y = p.get("y")
            if x is None or y is None: continue
            if cls.startswith("ally"):
                allies.append((x,y))
            elif cls.startswith("enemy"):
                enemies.append((x,y))
                enemy_names.append(cls.replace("enemy ","",1) if cls.startswith("enemy ") else (cls_raw or "enemy"))
            else:
                enemies.append((x,y))
                enemy_names.append(cls_raw if cls_raw else "unknown")

        self._last_enemy = enemy_names

        def norm(units): return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x,y in units]
        def pad(units, n):
            units = norm(units)
            if len(units) < n: units += [(0.0,0.0)] * (n - len(units))
            return units[:n]

        ally_flat  = [c for pos in pad(allies, MAX_ALLIES)  for c in pos]
        enemy_flat = [c for pos in pad(enemies, MAX_ENEMIES) for c in pos]
        return np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)

    # ---------------- Tower OCR ----------------
    def _prime_tower_max_hp(self):
        for _ in range(4):
            try:
                self.actions.capture_area(self.screenshot_path)
                ws = os.getenv('WORKSPACE_TROOP_DETECTION')
                res = self.rf_model.run_workflow(
                    workspace_name=ws,
                    workflow_id="detect-count-and-visualize",
                    images={"image": self.screenshot_path}
                )
                preds = []
                if isinstance(res, dict) and "predictions" in res:
                    preds = res["predictions"]
                elif isinstance(res, list) and res and isinstance(res[0], dict) and "predictions" in res[0]:
                    preds = res[0]["predictions"]
                self._update_tower_hp_from_ocr(preds, init_only=True)
                if self._tower_hp_max["ally"]["king"] and self._tower_hp_max["enemy"]["king"]:
                    self._tower_max_initialized = True
                    return
            except Exception:
                pass
            time.sleep(0.35)
        self._tower_max_initialized = True

    def _update_tower_hp_from_ocr(self, predictions, init_only=False):
        if not os.path.isfile(self.screenshot_path): return
        img = Image.open(self.screenshot_path).convert("RGB")

        boxes = self._get_tower_boxes(predictions)
        for side in ("ally","enemy"):
            if not any(boxes[side].values()):
                boxes[side] = self._fallback_tower_rois(img.size, side)

        for side in ("ally","enemy"):
            for slot in ("princess_left","king","princess_right"):
                box = boxes[side].get(slot)
                if not box: continue
                crop = self._tower_text_crop(img, box, side)
                val = self._ocr_int(crop)
                if val is None: continue
                self._tower_hp_curr[side][slot] = val
                if init_only or self._tower_hp_max[side][slot] is None:
                    self._tower_hp_max[side][slot] = val
                elif val > (self._tower_hp_max[side][slot] or 0):
                    self._tower_hp_max[side][slot] = val

    def _get_tower_boxes(self, predictions):
        def nc(s): return s.strip().lower() if isinstance(s, str) else ""
        by = {"ally": [], "enemy": []}
        for p in predictions or []:
            if not isinstance(p, dict): continue
            cls = nc(p.get("class",""))
            if cls not in ("ally king tower","ally princess tower","enemy king tower","enemy princess tower"): continue
            x = p.get("x"); y = p.get("y")
            bw = int(p.get("width") or p.get("w") or 120)
            bh = int(p.get("height") or p.get("h") or 180)
            if x is None or y is None: continue
            side = "ally" if cls.startswith("ally") else "enemy"
            typ  = "king" if "king" in cls else "princess"
            by[side].append({"type": typ, "x": int(x), "y": int(y), "w": bw, "h": bh})

        def choose(arr):
            left = None; right = None; king = None
            for it in arr:
                if it["type"] == "king": king = it
                else:
                    if left is None or it["x"] < (left["x"] if left else 10**9):
                        left = it
                    if right is None or it["x"] > (right["x"] if right else -10**9):
                        right = it
            return {"princess_left": left, "king": king, "princess_right": right}

        return {"ally": choose(by["ally"]), "enemy": choose(by["enemy"])}

    def _fallback_tower_rois(self, img_size, side):
        L = self.actions.TOP_LEFT_X
        T = self.actions.TOP_LEFT_Y
        W = self.actions.WIDTH
        H = self.actions.HEIGHT

        xs = [L + int(0.25 * W), L + int(0.50 * W), L + int(0.75 * W)]
        if side == "enemy":
            ys = [T + int(0.17 * H), T + int(0.08 * H), T + int(0.17 * H)]
        else:
            ys = [T + int(0.86 * H), T + int(0.94 * H), T + int(0.86 * H)]

        bw = max(100, int(0.18 * W))
        bh = max(120, int(0.22 * H))

        return {
            "princess_left":  {"x": xs[0], "y": ys[0], "w": bw, "h": bh},
            "king":           {"x": xs[1], "y": ys[1], "w": bw, "h": bh},
            "princess_right": {"x": xs[2], "y": ys[2], "w": bw, "h": bh},
        }

    def _tower_text_crop(self, img, box, side):
        w_img, h_img = img.size
        x = box["x"]; y = box["y"]; w = box["w"]; h = box["h"]

        band_w = max(80, min(260, int(0.95 * w)))
        band_h = max(18, min(42, int(0.16 * h)))

        if side == "enemy":
            top = y - int(0.70 * h)   # above bar
        else:
            top = y - int(0.38 * h)   # under bar

        left = x - band_w // 2
        right = left + band_w
        bottom = top + band_h

        left = max(0, left); top = max(0, top)
        right = min(w_img, right); bottom = min(h_img, bottom)
        if right - left < 10 or bottom - top < 10:
            return None
        return img.crop((left, top, right, bottom))

    def _ocr_int(self, crop):
        if crop is None: return None
        g = ImageOps.grayscale(crop)
        g = ImageOps.autocontrast(g, cutoff=2)
        g = g.filter(ImageFilter.MedianFilter(3))
        g = g.point(lambda p: 255 if p > 140 else 0)
        txt = pytesseract.image_to_string(
            g, config="--psm 7 -l eng -c tessedit_char_whitelist=0123456789"
        )
        m = re.search(r"(\d{2,6})", txt)
        return int(m.group(1)) if m else None

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
        """Don’t crash if Actions has no detect_game_end()."""
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

    def get_tower_hp(self):
        """
        Returns percentages 0..100 (rounded client-side) or None if unknown.
        Keys: tower_hp = { ally:{princess_left, king, princess_right}, enemy:{...} }
        """
        out = {"ally": {"king": None, "princess_left": None, "princess_right": None},
               "enemy": {"king": None, "princess_left": None, "princess_right": None}}
        for side in ("ally","enemy"):
            for slot in ("princess_left","king","princess_right"):
                cur = self._tower_hp_curr[side][slot]
                mx  = self._tower_hp_max[side][slot]
                if cur is not None and mx and mx > 0:
                    out[side][slot] = 100.0 * (cur / mx)
        return out
