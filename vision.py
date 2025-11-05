# vision.py
import os
import re
import numpy as np
from PIL import Image
import pyautogui as pag
import pytesseract

# Set this if Tesseract isn't on PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PLAY_AGAIN_RGB = (239, 178, 40)  # given
COLOR_TOL = 18                    # tweak if needed

def screenshot_region(bbox):
    # bbox: (left, top, width, height)
    L, T, W, H = bbox
    return pag.screenshot(region=(L, T, W, H))

def has_winner_text(img: Image.Image) -> bool:
    # Light OCR clean
    text = pytesseract.image_to_string(
        img,
        config="--psm 6 -c preserve_interword_spaces=1"
    )
    # WINNER or WINNER! anywhere
    return re.search(r"\bWINNER!?\\b", text, re.IGNORECASE) is not None

def find_play_again_center(bbox) -> tuple | None:
    """Scan bottom half for the target color. Returns absolute screen coords or None."""
    L, T, W, H = bbox
    img = pag.screenshot(region=(L, T + H//2, W, H//2))
    arr = np.array(img)[:, :, :3]

    r, g, b = PLAY_AGAIN_RGB
    dr = np.abs(arr[:, :, 0] - r)
    dg = np.abs(arr[:, :, 1] - g)
    db = np.abs(arr[:, :, 2] - b)
    mask = (dr <= COLOR_TOL) & (dg <= COLOR_TOL) & (db <= COLOR_TOL)

    if not mask.any():
        return None

    ys, xs = np.where(mask)
    # take median to avoid outliers
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    # convert back to absolute screen coords
    abs_x = L + cx
    abs_y = T + H//2 + cy
    return (abs_x, abs_y)
