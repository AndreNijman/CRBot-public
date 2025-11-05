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

PLAY_AGAIN_RGB = (239, 178, 40)  # given color
COLOR_TOL = 18                   # bump to 24–28 if needed

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

def find_play_again_center(bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Scan bottom half for the target color. Returns absolute screen coords or None."""
    L, T, W, H = bbox
    img = pag.screenshot(region=(L, T + H // 2, W, H // 2))
    arr = np.asarray(img, dtype=np.int16)[:, :, :3]  # int16 avoids overflow

    r, g, b = PLAY_AGAIN_RGB
    dr = np.abs(arr[:, :, 0] - r)
    dg = np.abs(arr[:, :, 1] - g)
    db = np.abs(arr[:, :, 2] - b)
    mask = (dr <= COLOR_TOL) & (dg <= COLOR_TOL) & (db <= COLOR_TOL)

    if not mask.any():
        return None

    ys, xs = np.where(mask)
    # median reduces outliers
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    # absolute screen coords
    abs_x = L + cx
    abs_y = T + H // 2 + cy
    return (abs_x, abs_y)
