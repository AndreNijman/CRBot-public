# window_helper.py
import time
import pyautogui as pag
import pygetwindow as gw

WIN_TITLE = "pyclashbot-96"

# Narrow phone-ish width. Tweak if your templates expect a different width.
TARGET_W = 720  # retained as fallback if window size can't be read

def _find_window():
    titles = [t for t in gw.getAllTitles() if WIN_TITLE.lower() in t.lower()]
    if not titles:
        raise RuntimeError(f"Window '{WIN_TITLE}' not found")
    return gw.getWindowsWithTitle(titles[0])[0]

def align_and_get_bbox():
    # Screen size for primary display
    screen_w, screen_h = pag.size()

    win = _find_window()
    if win.isMinimized:
        win.restore()
        time.sleep(0.25)

    # Activate then move without resizing. Small sleeps help avoid OS snap glitches.
    win.activate()
    time.sleep(0.2)

    width = max(getattr(win, "width", TARGET_W), 1)
    y = 0
    x = max(0, screen_w - width)

    win.moveTo(x, y)
    time.sleep(0.15)

    # Return stable bbox (left, top, width, height)
    return (win.left, win.top, win.width, win.height)
