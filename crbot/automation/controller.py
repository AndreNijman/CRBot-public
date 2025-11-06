from __future__ import annotations

import random
import time
import platform
from pathlib import Path
from typing import Iterable, List

import pyautogui

from crbot.config import MAIN_IMAGES_DIR, SCREENSHOTS_DIR


class ActionController:
    """High-level wrapper around pyautogui that knows how to interact with the arena."""

    def __init__(self) -> None:
        self.os_type = platform.system()
        self.images_folder = MAIN_IMAGES_DIR
        self.screenshot_dir = SCREENSHOTS_DIR / "cards"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        if self.os_type == "Darwin":
            self.TOP_LEFT_X = 1013
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1480
            self.BOTTOM_RIGHT_Y = 683
        else:  # default to Windows coordinates
            self.TOP_LEFT_X = 1376
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1838
            self.BOTTOM_RIGHT_Y = 769

        self.FIELD_AREA = (
            self.TOP_LEFT_X,
            self.TOP_LEFT_Y,
            self.BOTTOM_RIGHT_X,
            self.BOTTOM_RIGHT_Y,
        )
        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

        if self.os_type == "Darwin":
            self.CARD_BAR_X = self.TOP_LEFT_X
            self.CARD_BAR_Y = self.TOP_LEFT_Y + int(self.HEIGHT * 0.80)
            self.CARD_BAR_WIDTH = self.WIDTH
            self.CARD_BAR_HEIGHT = int(self.HEIGHT * 0.20)
        else:
            self.CARD_BAR_X = 1450
            self.CARD_BAR_Y = 847
            self.CARD_BAR_WIDTH = 1862 - 1450
            self.CARD_BAR_HEIGHT = 971 - 847

        self.card_keys = {0: "1", 1: "2", 2: "3", 3: "4"}
        self.current_card_positions: dict[str, int] = {}
        self._input_locked = False

    # ---------- Input lock ----------
    def set_input_lock(self, locked: bool) -> None:
        self._input_locked = bool(locked)

    def is_input_locked(self) -> bool:
        return self._input_locked

    # --- SAFE STUB so the environment watcher never crashes ---
    def detect_game_end(self):
        """Return 'victory' / 'defeat' / None. Stubbed to None; OCR handles detection elsewhere."""
        return None

    # ---------- Captures ----------
    def capture_area(self, save_path: str | Path) -> None:
        screenshot = pyautogui.screenshot(
            region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT)
        )
        screenshot.save(str(save_path))

    def capture_card_area(self, save_path: str | Path) -> None:
        screenshot = pyautogui.screenshot(
            region=(
                self.CARD_BAR_X,
                self.CARD_BAR_Y,
                self.CARD_BAR_WIDTH,
                self.CARD_BAR_HEIGHT,
            )
        )
        screenshot.save(str(save_path))

    def capture_individual_cards(self) -> List[Path]:
        screenshot = pyautogui.screenshot(
            region=(
                self.CARD_BAR_X,
                self.CARD_BAR_Y,
                self.CARD_BAR_WIDTH,
                self.CARD_BAR_HEIGHT,
            )
        )
        card_width = self.CARD_BAR_WIDTH // 4
        cards: list[Path] = []
        for i in range(4):
            left = i * card_width
            card_img = screenshot.crop(
                (left, 0, left + card_width, self.CARD_BAR_HEIGHT)
            )
            save_path = self.screenshot_dir / f"card_{i + 1}.png"
            card_img.save(save_path)
            cards.append(save_path)
        return cards

    # ---------- Elixir ----------
    def count_elixir(self) -> int:
        if self.os_type == "Darwin":
            for i in range(10, 0, -1):
                img = self.images_folder / f"{i}elixir.png"
                try:
                    if pyautogui.locateOnScreen(
                        str(img), confidence=0.5, grayscale=True
                    ):
                        return i
                except Exception:
                    continue
            return 0
        if self.os_type == "Windows":
            count = self._count_elixir_windows_bar()
            if count is None:
                count = self._count_elixir_windows_fallback()
            return max(0, min(10, count or 0))
        return 0

    def _count_elixir_windows_bar(self) -> int | None:
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

        def is_purple(r: int, g: int, b: int) -> bool:
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

    def _count_elixir_windows_fallback(self) -> int:
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

    def update_card_positions(self, detections: Iterable[dict]) -> None:
        sorted_cards = sorted(detections, key=lambda x: x["x"])
        self.current_card_positions = {
            card["class"]: idx for idx, card in enumerate(sorted_cards)
        }

    # ---------- Plays ----------
    def card_play(self, x: int, y: int, card_index: int) -> None:
        if self._input_locked:
            return
        if card_index in self.card_keys:
            key = self.card_keys[card_index]
            pyautogui.press(key)
            time.sleep(0.2)
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()

    def click_battle_start(self) -> bool:
        if self._input_locked:
            return False
        button_image = self.images_folder / "battlestartbutton.png"
        confidences = [0.8, 0.7, 0.6, 0.5]
        region = (1486, 755, 1730 - 1486, 900 - 755)
        while True:
            for confidence in confidences:
                try:
                    loc = pyautogui.locateOnScreen(
                        str(button_image), confidence=confidence, region=region
                    )
                    if loc:
                        x, y = pyautogui.center(loc)
                        pyautogui.moveTo(x, y, duration=0.2)
                        pyautogui.click()
                        return True
                except Exception:
                    continue
            pyautogui.moveTo(1705, 331, duration=0.2)
            pyautogui.click()
            time.sleep(1)

    def press_play_again_keyburst(
        self, repeats: int = 1, delay: float = 0.25, keys: Iterable[str] = ("1", "b", "n")
    ) -> None:
        if self._input_locked:
            return
        for _ in range(repeats):
            for key in keys:
                pyautogui.press(key)
                time.sleep(delay)

    def emote_burst(self) -> None:
        if self._input_locked:
            return
        pyautogui.press("e")
        time.sleep(0.5)
        pyautogui.press(random.choice(("5", "6", "7", "8", "9", "0")))


# Backwards-compatible alias while callers migrate
Actions = ActionController
