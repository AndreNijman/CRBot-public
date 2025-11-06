# scripts/live_vision.py
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mss
import win32gui
import win32con
import win32api

TITLE_SUBSTR = "pyclashbot-96"   # exact window title or substring
MODEL_DIR = "yolo_models"
CARDS_PT = f"{MODEL_DIR}/cards.pt"
TROOPS_PT = f"{MODEL_DIR}/troops.pt"
TOWERS_PT = f"{MODEL_DIR}/towers.pt"

CONF_CARDS, CONF_TROOPS, CONF_TOWERS = 0.35, 0.30, 0.30
IMGSZ = 640
WINDOW_NAME = "CR Vision Tester"

# Load models once
cards_model  = YOLO(CARDS_PT)
troops_model = YOLO(TROOPS_PT)
towers_model = YOLO(TOWERS_PT)

def _find_window_rect(title_substr: str):
    """Return client-area bbox (left, top, right, bottom) of the first window whose title contains `title_substr`."""
    target = {"hwnd": None, "title": None}

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        t = win32gui.GetWindowText(hwnd)
        if t and title_substr.lower() in t.lower():
            target["hwnd"] = hwnd
            target["title"] = t

    win32gui.EnumWindows(enum_cb, None)
    hwnd = target["hwnd"]
    if not hwnd:
        return None, None

    # Bring to front
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    # Client rect in client coords
    l, t, r, b = win32gui.GetClientRect(hwnd)
    # Convert client origin to screen coords
    pt = win32gui.ClientToScreen(hwnd, (0, 0))
    left, top = pt
    right, bottom = left + (r - l), top + (b - t)
    return hwnd, (left, top, right, bottom)

def _grab_client_region(mss_obj: mss.mss, rect):
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    shot = mss_obj.grab({"left": left, "top": top, "width": w, "height": h})
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR).copy()  # BGRA -> BGR drop A
    return frame

def _predict(model, frame_bgr, conf):
    # feed BGR as PIL RGB to Ultralytics
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    r = model.predict(img, conf=conf, imgsz=IMGSZ, device=0, verbose=False)[0]
    dets = []
    names = r.names
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        score = float(b.conf[0])
        dets.append((names[cls], score, (x1, y1, x2, y2)))
    return dets

def _draw(frame, dets, color):
    for label, score, (x1, y1, x2, y2) in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    hwnd, rect = _find_window_rect(TITLE_SUBSTR)
    if not hwnd:
        print(f"Window not found containing title: {TITLE_SUBSTR}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)


    fps = 0.0
    with mss.mss() as sct:
        while True:
            t0 = time.time()
            frame = _grab_client_region(sct, rect)

            # run all three on the same frame
            det_cards  = _predict(cards_model,  frame, CONF_CARDS)
            det_troops = _predict(troops_model, frame, CONF_TROOPS)
            det_towers = _predict(towers_model, frame, CONF_TOWERS)

            # draw all overlays on one frame
            _draw(frame, det_cards,  (0, 255, 0))     # green  cards
            _draw(frame, det_troops, (255, 200, 0))   # orange troops
            _draw(frame, det_towers, (0, 200, 255))   # cyan   towers

            dt = max(time.time() - t0, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyWindow(WINDOW_NAME)

if __name__ == "__main__":
    main()
