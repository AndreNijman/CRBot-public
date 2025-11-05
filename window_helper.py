# window_helper.py
import time
import pyautogui as pag
import pygetwindow as gw

WIN_TITLE = "pyclashbot-96"

# Narrow phone-ish width. Tweak if your templates expect a different width.
TARGET_W = 720  # try 540, 640, 720, or 900 depending on your setup

def _find_window():
    titles = [t for t in gw.getAllTitles() if WIN_TITLE.lower() in t.lower()]
    if not titles:
        raise RuntimeError(f"Window '{WIN_TITLE}' not found")
    return gw.getWindowsWithTitle(titles[0])[0]

def align_and_get_bbox():
    # Screen size for primary display
    screen_w, screen_h = pag.size()

    # Compute top-left so the window hugs the RIGHT edge and spans full height
    x = max(0, screen_w - TARGET_W)
    y = 0
    w = TARGET_W
    h = screen_h

    win = _find_window()
    if win.isMinimized:
        win.restore()
        time.sleep(0.25)

    # Activate, move, then resize. Small sleeps help avoid OS snap glitches.
    win.activate()
    time.sleep(0.2)
    win.moveTo(x, y)
    time.sleep(0.15)
    win.resizeTo(w, h)
    time.sleep(0.25)

    # Return stable bbox (left, top, width, height)
    return (win.left, win.top, win.width, win.height)
