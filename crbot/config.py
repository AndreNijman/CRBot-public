from __future__ import annotations

from pathlib import Path

# Root of the repository (crbot/ sits one directory below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common asset and artifact locations
ASSETS_DIR = PROJECT_ROOT / "assets"
MAIN_IMAGES_DIR = ASSETS_DIR / "main_images"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
DEBUG_DIR = PROJECT_ROOT / "debug"
LOG_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure writable directories exist when the package is imported
for directory in (ASSETS_DIR, MAIN_IMAGES_DIR, SCREENSHOTS_DIR, DEBUG_DIR, LOG_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
