from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import pyautogui as pag
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pytesseract import TesseractError


def _configure_tesseract() -> bool:
    """Configure pytesseract if the binary path is supplied. Returns availability."""
    exe_from_env = os.getenv("TESSERACT_EXE")
    candidate_paths = []
    if exe_from_env:
        candidate_paths.append(exe_from_env)

    # Default Windows install location as a fallback
    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt":
        candidate_paths.append(default_win_path)

    for path in candidate_paths:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tessdata = os.path.join(os.path.dirname(path), "tessdata")
            if os.path.isdir(tessdata):
                os.environ.setdefault("TESSDATA_PREFIX", tessdata)
            break

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


HAS_TESSERACT = _configure_tesseract()

PLAY_AGAIN_RGB = (73, 164, 239)
PLAY_AGAIN_TOL = 18

SECOND_BUTTON_RGB = (239, 175, 0)
SECOND_BUTTON_TOL = 24


def assert_tesseract_ready() -> None:
    if not HAS_TESSERACT:
        raise RuntimeError(
            "Tesseract not reachable. Set the TESSERACT_EXE env var to your install path."
        )


def screenshot_region(bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Capture a screenshot of ``bbox`` (left, top, width, height)."""
    left, top, width, height = bbox
    return pag.screenshot(region=(left, top, width, height))


def _prep_for_ocr(img: Image.Image) -> Image.Image:
    """Lightly denoise and normalize an image to improve OCR stability."""
    gray = ImageOps.grayscale(img)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return ImageOps.autocontrast(gray, cutoff=2)


def has_winner_text(img: Image.Image) -> bool:
    try:
        proc = _prep_for_ocr(img).convert("L").copy()
        text = pytesseract.image_to_string(
            proc,
            config="--psm 6 -c preserve_interword_spaces=1",
            timeout=2,
        )
        return re.search(r"\bWINNER!?\b", text, re.IGNORECASE) is not None
    except TesseractError as exc:
        print(f"OCR timeout/crash: {exc}")
        return False
    except Exception as exc:
        print(f"OCR error: {exc}")
        return False


def _find_color_center(
    bbox: Tuple[int, int, int, int],
    target_rgb: Tuple[int, int, int],
    tolerance: int,
    vertical_slice: Tuple[float, float] = (0.5, 1.0),
) -> Optional[Tuple[int, int]]:
    """
    Locate the median pixel position of ``target_rgb`` within ``bbox``.
    ``vertical_slice`` constrains the scan to a percentage of the area height.
    """
    left, top, width, height = bbox
    start_frac, end_frac = vertical_slice
    start_frac = max(0.0, min(start_frac, 1.0))
    end_frac = max(start_frac, min(end_frac, 1.0))

    slice_top = int(round(height * start_frac))
    slice_bottom = int(round(height * end_frac))
    slice_height = max(1, slice_bottom - slice_top)

    img = pag.screenshot(region=(left, top + slice_top, width, slice_height))
    arr = np.asarray(img, dtype=np.int16)[:, :, :3]

    r, g, b = target_rgb
    tol = max(0, tolerance)
    mask = (
        (np.abs(arr[:, :, 0] - r) <= tol)
        & (np.abs(arr[:, :, 1] - g) <= tol)
        & (np.abs(arr[:, :, 2] - b) <= tol)
    )

    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask)
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    abs_x = left + cx
    abs_y = top + slice_top + cy
    return abs_x, abs_y


def find_play_again_center(bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Locate the Play Again button colour cluster within ``bbox``."""
    return _find_color_center(
        bbox,
        PLAY_AGAIN_RGB,
        PLAY_AGAIN_TOL,
        vertical_slice=(0.5, 1.0),
    )


def find_start_battle_center(bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Locate the yellow confirmation button used to queue the next match."""
    return _find_color_center(
        bbox,
        SECOND_BUTTON_RGB,
        SECOND_BUTTON_TOL,
        vertical_slice=(0.45, 1.0),
    )
