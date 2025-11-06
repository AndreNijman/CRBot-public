from __future__ import annotations

import time
from pathlib import Path

from crbot.config import SCREENSHOTS_DIR
from crbot.environment import ClashRoyaleEnv


def test_card_detection() -> None:
    env = ClashRoyaleEnv()

    print("Testing card detection...")
    print("-" * 50)

    print("1. Capturing card area...")
    capture_dir = SCREENSHOTS_DIR / "tests"
    capture_dir.mkdir(parents=True, exist_ok=True)
    card_area_path = capture_dir / "cards.png"
    env.actions.capture_card_area(card_area_path)
    print(f"Card area screenshot saved to: {card_area_path}")

    print("\n2. Detecting cards...")
    cards = env.detect_cards_in_hand()
    print(f"Detected cards: {cards}")

    if not cards:
        print("No cards detected. Stopping test.")
        return

    print("\nCard positions in hand:")
    for card, position in env.actions.current_card_positions.items():
        key = env.actions.card_keys.get(position)
        print(f"Card {card} is in position {position} (Key: {key})")

    print("\nTesting card placement...")
    for card in cards:
        print(f"\nTrying to play card: {card}")
        x = env.actions.WIDTH // 2
        y = env.actions.HEIGHT * 3 // 4
        env.actions.card_play(x, y, card)
        time.sleep(2)


if __name__ == "__main__":
    test_card_detection()
