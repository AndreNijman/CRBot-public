import numpy as np
import time
import os
import pyautogui
import threading
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient

# Load environment variables from .env file
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

        # ---- UI-facing caches (read by train.py via getters) ----
        self._last_hand   = []      # list[str]
        self._last_enemy  = []      # list[str]
        self._last_elixir = None    # int 0..10

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
        # We assume "Play Again" was clicked by the controller layer.
        time.sleep(3)
        self.game_over_flag = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected = False
        # reset UI caches for new episode
        self._last_hand = []
        self._last_enemy = []
        self._last_elixir = None
        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    # ---------------- Main RL step ----------------
    def step(self, action_index):
        # If your Actions has a match-over image detector, gate actions:
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # force no-op

        if self.game_over_flag:
            done = True
            reward = self._compute_reward(self._get_state())
            result = self.game_over_flag
            if result == "victory":
                reward += 100
            elif result == "defeat":
                reward -= 100
            self.match_over_detected = False
            return self._get_state(), reward, done

        self.current_cards = self.detect_cards_in_hand()  # updates self._last_hand internally

        # If all cards Unknown, do a harmless click and skip move
        if self.current_cards and all(card == "Unknown" for card in self.current_cards):
            pyautogui.moveTo(1611, 831, duration=0.2)
            pyautogui.click()
            next_state = self._get_state()
            return next_state, 0, False

        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action

        spell_penalty = 0
        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(1)

            # Penalize spells dropped with no nearby enemy cluster
            if card_name in SPELL_CARDS:
                state = self._get_state()
                enemy_positions = []
                for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                    ex = state[i]
                    ey = state[i + 1]
                    if ex != 0.0 or ey != 0.0:
                        ex_px = int(ex * self.actions.WIDTH)
                        ey_px = int(ey * self.actions.HEIGHT)
                        enemy_positions.append((ex_px, ey_px))
                radius = 100
                found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < radius for ex, ey in enemy_positions)
                if not found_enemy:
                    spell_penalty = -5

        # Princess tower reward
        current_enemy_princess_towers = self._count_enemy_princess_towers()
        princess_tower_reward = 0
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        done = False
        reward = self._compute_reward(self._get_state()) + spell_penalty + princess_tower_reward
        next_state = self._get_state()
        return next_state, reward, done

    # ---------------- State construction ----------------
    def _get_state(self):
        # fresh frame
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()
        self._last_elixir = int(elixir) if elixir is not None else None

        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION env var is not set.")

        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )

        # unify predictions access
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]

        # No predictions -> return previous state shape if possible
        if not predictions:
            ally_flat = [0.0] * (2 * MAX_ALLIES)
            enemy_flat = [0.0] * (2 * MAX_ENEMIES)
            state = np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)
            # clear enemy cache since none seen
            self._last_enemy = []
            return state

        TOWER_CLASSES = {
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        }

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        # collect allies / enemies (positions) and enemy names for UI
        allies = []
        enemies = []
        enemy_names = []

        for p in predictions:
            if not isinstance(p, dict):
                continue
            cls_raw = p.get("class", "")
            cls = normalize_class(cls_raw)
            if cls in TOWER_CLASSES:
                continue
            x = p.get("x"); y = p.get("y")
            if x is None or y is None:
                continue

            if cls.startswith("ally"):
                allies.append((x, y))
                # we don't list ally names in UI
            elif cls.startswith("enemy"):
                enemies.append((x, y))
                name = cls.replace("enemy ", "", 1) if cls.startswith("enemy ") else (cls_raw or "enemy")
                enemy_names.append(name)
            else:
                # Unknown team label from model. Treat as enemy so UI shows something.
                enemies.append((x, y))
                enemy_names.append(cls_raw if cls_raw else "unknown")

        # Update enemy cache (names only)
        self._last_enemy = enemy_names

        # Normalize to field size
        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_positions  = pad_units(allies,  MAX_ALLIES)
        enemy_positions = pad_units(enemies, MAX_ENEMIES)

        ally_flat  = [coord for pos in ally_positions  for coord in pos]
        enemy_flat = [coord for pos in enemy_positions for coord in pos]

        state = np.array([(self._last_elixir or 0) / 10.0] + ally_flat + enemy_flat, dtype=np.float32)
        return state

    # ---------------- Reward ----------------
    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[0] * 10
        enemy_positions = state[1 + 2 * MAX_ALLIES:]  # x1,y1,x2,y2,....
        enemy_presence = sum(enemy_positions[1::2])   # sum y only

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

            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION env var is not set.")

            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )

                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])

                if predictions:
                    card_name = predictions[0].get("class", "Unknown")
                    cards.append(card_name)
                else:
                    cards.append("Unknown")

            # Update hand cache for UI
            self._last_hand = list(cards)
            return cards
        except Exception:
            # keep previous cache but return empty to avoid breaking logic
            return []

    # ---------------- Action space ----------------
    def get_available_actions(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op action
        return actions

    # ---------------- Endgame watcher ----------------
    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            time.sleep(0.1)  # faster polling

    # ---------------- Princess towers ----------------
    def _count_enemy_princess_towers(self):
        self.actions.capture_area(self.screenshot_path)

        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION env var is not set.")

        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
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
