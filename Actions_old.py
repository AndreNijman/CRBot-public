import os
import time
import platform
import random

import pyautogui

class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(self.script_dir, 'main_images')

        if self.os_type == "Darwin":
            self.TOP_LEFT_X = 1013
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1480
            self.BOTTOM_RIGHT_Y = 683
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

            self.CARD_BAR_X = self.TOP_LEFT_X
            self.CARD_BAR_Y = self.TOP_LEFT_Y + int(self.HEIGHT * 0.80)
            self.CARD_BAR_WIDTH = self.WIDTH
            self.CARD_BAR_HEIGHT = int(self.HEIGHT * 0.20)

        elif self.os_type == "Windows":
            self.TOP_LEFT_X = 1376
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1838
            self.BOTTOM_RIGHT_Y = 769
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

            self.CARD_BAR_X = 1450
            self.CARD_BAR_Y = 847
            self.CARD_BAR_WIDTH = 1862 - 1450
            self.CARD_BAR_HEIGHT = 971 - 847

        self.card_keys = {0: '1', 1: '2', 2: '3', 3: '4'}
        self.current_card_positions = {}
        self._input_locked = False

    def set_input_lock(self, locked: bool):
        self._input_locked = bool(locked)

    def is_input_locked(self) -> bool:
        return self._input_locked

    # --- SAFE STUB so env._endgame_watcher never crashes ---
    def detect_game_end(self):
        """Return 'victory' / 'defeat' / None. Stubbed to None; train.py handles OCR WINNER."""
        return None

    # ---------- Captures ----------
    def capture_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(save_path)

    def capture_card_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(self.CARD_BAR_X, self.CARD_BAR_Y, self.CARD_BAR_WIDTH, self.CARD_BAR_HEIGHT))
        screenshot.save(save_path)

    def capture_individual_cards(self):
        screenshot = pyautogui.screenshot(region=(self.CARD_BAR_X, self.CARD_BAR_Y, self.CARD_BAR_WIDTH, self.CARD_BAR_HEIGHT))
        card_width = self.CARD_BAR_WIDTH // 4
        cards = []
        for i in range(4):
            left = i * card_width
            card_img = screenshot.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            save_path = os.path.join(self.script_dir, 'screenshots', f"card_{i+1}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            card_img.save(save_path)
            cards.append(save_path)
        return cards

    # ---------- Elixir ----------
    def count_elixir(self):
        if self.os_type == "Darwin":
            for i in range(10, 0, -1):
                img = os.path.join(self.images_folder, f"{i}elixir.png")
                try:
                    if pyautogui.locateOnScreen(img, confidence=0.5, grayscale=True):
                        return i
                except Exception:
                    pass
            return 0
        elif self.os_type == "Windows":
            count = self._count_elixir_windows_bar()
            if count is None:
                count = self._count_elixir_windows_fallback()
            return max(0, min(10, count if count is not None else 0))
        else:
            return 0

    def _count_elixir_windows_bar(self):
        width = self.CARD_BAR_WIDTH
        if width <= 0:
            return None

        region_height = max(40, int(self.CARD_BAR_HEIGHT * 0.5))
        top = max(self.CARD_BAR_Y - region_height - 8, 0)

        try:
            grab = pyautogui.screenshot(
                region=(self.CARD_BAR_X, top, width, region_height)
            )
        except Exception:
            return None

        img = grab.convert("RGB")
        pixels = img.load()
        w, h = img.size
        if w <= 0 or h <= 0:
            return None

        segment_width = w / 10.0
        start_y = max(0, int(h * 0.35))
        end_y = min(h, int(h * 0.95))

        def is_purple(r, g, b):
            if r < 110 or b < 110:
                return False
            if r - g < 25 or b - g < 25:
                return False
            if abs(r - b) > 90:
                return False
            return True

        filled_segments = 0
        for idx in range(10):
            x0 = int(idx * segment_width)
            x1 = int((idx + 1) * segment_width)
            if x1 <= x0:
                x1 = min(w, x0 + 1)

            purple = 0
            total = 0
            for y in range(start_y, end_y, 2):
                for x in range(x0, min(x1, w), 2):
                    r, g, b = pixels[x, y]
                    total += 1
                    if is_purple(r, g, b):
                        purple += 1
            if total and (purple / total) >= 0.12:
                filled_segments += 1

        return filled_segments

    def _count_elixir_windows_fallback(self):
        target = (225, 128, 229)
        tolerance = 110
        offsets = [62 + 38 * i for i in range(10)]
        sample_y = int(self.CARD_BAR_Y + 142)

        count = 0
        for offset in offsets:
            x = int(self.CARD_BAR_X + offset)
            try:
                r, g, b = pyautogui.pixel(x, sample_y)
            except Exception:
                continue
            if (
                abs(r - target[0]) <= tolerance
                and abs(g - target[1]) <= tolerance
                and abs(b - target[2]) <= tolerance
            ):
                count += 1
        return count

    def update_card_positions(self, detections):
        sorted_cards = sorted(detections, key=lambda x: x['x'])
        self.current_card_positions = {card['class']: idx for idx, card in enumerate(sorted_cards)}

    # ---------- Plays ----------
    def card_play(self, x, y, card_index):
        if self._input_locked:
            return
        if card_index in self.card_keys:
            key = self.card_keys[card_index]
            pyautogui.press(key)
            time.sleep(0.2)
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()

    def click_battle_start(self):
        if self._input_locked:
            return False
        button_image = os.path.join(self.images_folder, "battlestartbutton.png")
        confidences = [0.8, 0.7, 0.6, 0.5]
        region = (1486, 755, 1730 - 1486, 900 - 755)
        while True:
            for c in confidences:
                try:
                    loc = pyautogui.locateOnScreen(button_image, confidence=c, region=region)
                    if loc:
                        x, y = pyautogui.center(loc)
                        pyautogui.moveTo(x, y, duration=0.2)
                        pyautogui.click()
                        return True
                except Exception:
                    pass
            pyautogui.moveTo(1705, 331, duration=0.2)
            pyautogui.click()
            time.sleep(1)

    def press_play_again_keyburst(self, repeats=1, delay=0.25, keys=('1', 'b', 'n')):
        if self._input_locked:
            return
        for _ in range(repeats):
            for key in keys:
                pyautogui.press(key)
                time.sleep(delay)

    def emote_burst(self):
        if self._input_locked:
            return
        pyautogui.press('e')
        time.sleep(0.5)
        pyautogui.press(random.choice(('5', '6', '7', '8', '9', '0')))