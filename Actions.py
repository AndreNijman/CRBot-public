import pyautogui
import os
from datetime import datetime
import time
import platform

class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(self.script_dir, 'main_images')

        # Define screen regions based on OS
        if self.os_type == "Darwin":  # macOS
            self.TOP_LEFT_X = 1013
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1480
            self.BOTTOM_RIGHT_Y = 683
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)

            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

            # Safe defaults for card bar on mac (not used if you don't need it)
            self.CARD_BAR_X = self.TOP_LEFT_X
            self.CARD_BAR_Y = self.TOP_LEFT_Y + int(self.HEIGHT * 0.80)
            self.CARD_BAR_WIDTH = self.WIDTH
            self.CARD_BAR_HEIGHT = int(self.HEIGHT * 0.20)

        elif self.os_type == "Windows":  # windows
            self.TOP_LEFT_X = 1376
            self.TOP_LEFT_Y = 120
            self.BOTTOM_RIGHT_X = 1838
            self.BOTTOM_RIGHT_Y = 769
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)

            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

            # Card bar coordinates for Windows
            self.CARD_BAR_X = 1450
            self.CARD_BAR_Y = 847
            self.CARD_BAR_WIDTH = 1862 - 1450
            self.CARD_BAR_HEIGHT = 971 - 847

        # Card position to key mapping
        self.card_keys = {
            0: '1',
            1: '2',
            2: '3',
            3: '4'
        }

        # Card name to position mapping (will be updated during detection)
        self.current_card_positions = {}

    # -------------------------
    # Capture helpers
    # -------------------------
    def capture_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(save_path)

    def capture_card_area(self, save_path):
        """Capture screenshot of card area"""
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X,
            self.CARD_BAR_Y,
            self.CARD_BAR_WIDTH,
            self.CARD_BAR_HEIGHT
        ))
        screenshot.save(save_path)

    def capture_individual_cards(self):
        """Capture and split card bar into individual card images"""
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X,
            self.CARD_BAR_Y,
            self.CARD_BAR_WIDTH,
            self.CARD_BAR_HEIGHT
        ))

        # Calculate individual card widths
        card_width = self.CARD_BAR_WIDTH // 4
        cards = []

        # Split into 4 individual card images
        for i in range(4):
            left = i * card_width
            card_img = screenshot.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            save_path = os.path.join(self.script_dir, 'screenshots', f"card_{i+1}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            card_img.save(save_path)
            cards.append(save_path)

        return cards

    # -------------------------
    # Elixir + card positions
    # -------------------------
    def count_elixir(self):
        if self.os_type == "Darwin":
            for i in range(10, 0, -1):
                image_file = os.path.join(self.images_folder, f"{i}elixir.png")
                try:
                    location = pyautogui.locateOnScreen(image_file, confidence=0.5, grayscale=True)
                    if location:
                        return i
                except Exception as e:
                    print(f"Error locating {image_file}: {e}")
            return 0
        elif self.os_type == "Windows":
            target = (225, 128, 229)
            tolerance = 80
            count = 0
            for x in range(1512, 1892, 38):
                r, g, b = pyautogui.pixel(x, 989)
                if (abs(r - target[0]) <= tolerance) and (abs(g - target[1]) <= tolerance) and (abs(b - target[2]) <= tolerance):
                    count += 1
            return count
        else:
            return 0

    def update_card_positions(self, detections):
        """
        Update card positions based on detection results
        detections: list of dictionaries with 'class' and 'x' position
        """
        # Sort detections by x position (left to right)
        sorted_cards = sorted(detections, key=lambda x: x['x'])

        # Map cards to positions 0-3
        self.current_card_positions = {
            card['class']: idx
            for idx, card in enumerate(sorted_cards)
        }

    # -------------------------
    # Plays + clicks in battle
    # -------------------------
    def card_play(self, x, y, card_index):
        print(f"Playing card {card_index} at position ({x}, {y})")
        if card_index in self.card_keys:
            key = self.card_keys[card_index]
            print(f"Pressing key: {key}")
            pyautogui.press(key)
            time.sleep(0.2)
            print(f"Moving mouse to: ({x}, {y})")
            pyautogui.moveTo(x, y, duration=0.2)
            print("Clicking")
            pyautogui.click()
        else:
            print(f"Invalid card index: {card_index}")

    def click_battle_start(self):
        button_image = os.path.join(self.images_folder, "battlestartbutton.png")
        confidences = [0.8, 0.7, 0.6, 0.5]  # Try multiple confidence levels

        # Region for start button
        battle_button_region = (1486, 755, 1730 - 1486, 900 - 755)

        while True:
            for confidence in confidences:
                print(f"Looking for battle start button (confidence: {confidence})")
                try:
                    location = pyautogui.locateOnScreen(
                        button_image,
                        confidence=confidence,
                        region=battle_button_region
                    )
                    if location:
                        x, y = pyautogui.center(location)
                        print(f"Found battle start button at ({x}, {y})")
                        pyautogui.moveTo(x, y, duration=0.2)
                        pyautogui.click()
                        return True
                except Exception:
                    pass

            # clear overlays
            print("Button not found, clicking to clear screens...")
            pyautogui.moveTo(1705, 331, duration=0.2)
            pyautogui.click()
            time.sleep(1)

    # -------------------------
    # Post-battle helpers
    # -------------------------
    def _bottom_center_roi(self, w_ratio=0.60, y_from=0.65, y_to=0.95):
        """
        ROI inside the game window for bottom-center UI buttons like Play Again.
        Returns region=(left, top, width, height) in screen coords for locateOnScreen.
        """
        W = self.WIDTH
        H = self.HEIGHT

        roi_w = int(W * w_ratio)
        x0_rel = (W - roi_w) // 2
        x0 = self.TOP_LEFT_X + x0_rel
        y0 = self.TOP_LEFT_Y + int(H * y_from)
        roi_h = int(H * (y_to - y_from))
        return (x0, y0, roi_w, roi_h)

    def _click_center_bottom_fallback(self):
        """
        Safe fallback click near expected Play Again area for Trophy Road.
        """
        x = self.TOP_LEFT_X + self.WIDTH // 2
        y = self.TOP_LEFT_Y + int(self.HEIGHT * 0.86)
        print(f"Fallback click at ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.15)
        pyautogui.click()

    def click_play_again_trophyroad(self, timeout=3.0):
        """
        Robust Trophy Road Play Again click:
        - Searches for template(s) inside a bottom-center ROI of the game window
        - Falls back to a safe relative click if template not found
        """
        # You can provide both normal and alt templates if UI theme varies
        candidates = [
            os.path.join(self.images_folder, "play_again_trophyroad.png"),
            os.path.join(self.images_folder, "play_again.png"),  # optional backup if present
        ]
        # Filter to existing files
        templates = [p for p in candidates if os.path.isfile(p)]
        if not templates:
            print("No play-again templates found. Using fallback.")
            self._click_center_bottom_fallback()
            return True

        roi = self._bottom_center_roi(w_ratio=0.60, y_from=0.65, y_to=0.95)
        confidences = [0.88, 0.84, 0.80]

        t0 = time.time()
        while time.time() - t0 < timeout:
            for tpl in templates:
                for conf in confidences:
                    try:
                        loc = pyautogui.locateOnScreen(tpl, confidence=conf, region=roi)
                    except Exception as e:
                        print(f"locateOnScreen error: {e}")
                        loc = None
                    if loc:
                        x, y = pyautogui.center(loc)
                        print(f"Play Again matched at ({x}, {y}) with conf {conf} using {os.path.basename(tpl)}")
                        pyautogui.moveTo(x, y, duration=0.15)
                        pyautogui.click()
                        return True
            time.sleep(0.15)

        print("Play Again not found by template. Using fallback.")
        self._click_center_bottom_fallback()
        return True

    # -------------------------
    # End-of-battle detection
    # -------------------------
    def detect_game_end(self):
        """
        Detect end screen, decide victory/defeat, then click Play Again for Trophy Road.
        Replaced fixed coords with ROI template matching.
        """
        try:
            winner_img = os.path.join(self.images_folder, "Winner.png")
            confidences = [0.8, 0.7, 0.6]

            # Region where 'Winner' appears
            winner_region = (1510, 121, 1678 - 1510, 574 - 121)

            for confidence in confidences:
                print(f"\nTrying detection with confidence: {confidence}")
                winner_location = None

                # Try to find Winner in region
                try:
                    winner_location = pyautogui.locateOnScreen(
                        winner_img, confidence=confidence, grayscale=True, region=winner_region
                    )
                except Exception as e:
                    print(f"Error locating Winner: {str(e)}")

                if winner_location:
                    _, y = pyautogui.center(winner_location)
                    print(f"Found 'Winner' at y={y} with confidence {confidence}")
                    result = "victory" if y > 402 else "defeat"

                    # small pause for animation to settle
                    time.sleep(2.5)

                    # Try robust Trophy Road Play Again
                    self.click_play_again_trophyroad(timeout=3.0)
                    return result
        except Exception as e:
            print(f"Error in game end detection: {str(e)}")
        return None

    def detect_match_over(self):
        matchover_img = os.path.join(self.images_folder, "matchover.png")
        confidences = [0.8, 0.6, 0.4]
        # Define the region where the matchover image appears (adjust as needed)
        region = (1378, 335, 1808 - 1378, 411 - 335)
        for confidence in confidences:
            try:
                location = pyautogui.locateOnScreen(
                    matchover_img, confidence=confidence, grayscale=True, region=region
                )
                if location:
                    print("Match over detected!")
                    return True
            except Exception as e:
                print(f"Error locating matchover.png: {e}")
        return False
