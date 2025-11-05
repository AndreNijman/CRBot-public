import pyautogui
import os
import time
import platform

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
            target = (225, 128, 229)
            tol = 80
            count = 0
            for x in range(1512, 1892, 38):
                r, g, b = pyautogui.pixel(x, 989)
                if abs(r - target[0]) <= tol and abs(g - target[1]) <= tol and abs(b - target[2]) <= tol:
                    count += 1
            return count
        else:
            return 0

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
