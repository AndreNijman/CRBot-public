# vision.py
import os
import re
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageOps, ImageFilter
import pyautogui as pag
import pytesseract
from pytesseract import TesseractError

# --- Tesseract setup (Windows) ---
# Change this if you installed elsewhere.
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    # Optional: point to tessdata if you moved it
    tessdata = os.path.join(os.path.dirname(TESSERACT_EXE), "tessdata")
    if os.path.isdir(tessdata):
        os.environ.setdefault("TESSDATA_PREFIX", tessdata)

PLAY_AGAIN_RGB = (73, 164, 239)  # previously (239, 178, 40)
PLAY_AGAIN_TOL = 18

SECOND_BUTTON_RGB = (239, 175, 0)
SECOND_BUTTON_TOL = 24

# -------- Helpers --------
def assert_tesseract_ready() -> None:
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract not reachable. Set pytesseract.pytesseract.tesseract_cmd "
            f"to your install path. Details: {e}"
        )

def screenshot_region(bbox: Tuple[int, int, int, int]) -> Image.Image:
    # bbox: (left, top, width, height)
    L, T, W, H = bbox
    return pag.screenshot(region=(L, T, W, H))

# Light denoise to help OCR on big colorful screens
def _prep_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # Slight contrast bump by autocontrast
    g = ImageOps.autocontrast(g, cutoff=2)
    return g

def has_winner_text(img: Image.Image) -> bool:
    try:
        # Fully materialize a copy so Tesseract can't trip on PIL internals
        proc = _prep_for_ocr(img).convert("L").copy()
        text = pytesseract.image_to_string(
            proc,
            config="--psm 6 -c preserve_interword_spaces=1",
            timeout=2,  # guard against hung child proc on Windows
        )
        return re.search(r"\bWINNER!?\b", text, re.IGNORECASE) is not None
    except TesseractError as e:
        # Timeout or crash; skip this tick
        print(f"OCR timeout/crash: {e}")
        return False
    except Exception as e:
        # Don’t kill the watcher on random OCR errors
        print(f"OCR error: {e}")
        return False

def _find_color_center(
    bbox: Tuple[int, int, int, int],
    target_rgb: Tuple[int, int, int],
    tolerance: int,
    vertical_slice: Tuple[float, float] = (0.5, 1.0),
) -> Optional[Tuple[int, int]]:
    """
    Find the median pixel location of a target color inside a vertical slice of the bbox.
    Returns absolute screen coordinates or None.
    """
    L, T, W, H = bbox
    start_frac, end_frac = vertical_slice
    start_frac = max(0.0, min(start_frac, 1.0))
    end_frac = max(start_frac, min(end_frac, 1.0))

    slice_top = int(round(H * start_frac))
    slice_bottom = int(round(H * end_frac))
    slice_height = max(1, slice_bottom - slice_top)

    img = pag.screenshot(region=(L, T + slice_top, W, slice_height))
    arr = np.asarray(img, dtype=np.int16)[:, :, :3]  # int16 avoids overflow

    r, g, b = target_rgb
    tol = max(0, tolerance)
    dr = np.abs(arr[:, :, 0] - r)
    dg = np.abs(arr[:, :, 1] - g)
    db = np.abs(arr[:, :, 2] - b)
    mask = (dr <= tol) & (dg <= tol) & (db <= tol)

    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask)
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    abs_x = L + cx
    abs_y = T + slice_top + cy
    return (abs_x, abs_y)


def find_play_again_center(bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Locate the Play Again button color in the lower half of the bbox."""
    return _find_color_center(
        bbox,
        PLAY_AGAIN_RGB,
        PLAY_AGAIN_TOL,
        vertical_slice=(0.5, 1.0),
    )


def find_start_battle_center(bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Locate the yellow confirmation button used to start the next match."""
    return _find_color_center(
        bbox,
        SECOND_BUTTON_RGB,
        SECOND_BUTTON_TOL,
        vertical_slice=(0.45, 1.0),
    )


