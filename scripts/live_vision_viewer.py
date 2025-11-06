# scripts/live_vision_viewer.py
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mss
import win32gui, win32con

TITLE_SUBSTR = "pyclashbot-96"

MODEL_DIR = "yolo_models"
CARDS_PT  = f"{MODEL_DIR}/cards.pt"
TROOPS_PT = f"{MODEL_DIR}/troops.pt"
TOWERS_PT = f"{MODEL_DIR}/towers.pt"

# Keep cards/towers same as your working version.
CONF_CARDS, CONF_TROOPS, CONF_TOWERS = 0.25, 0.18, 0.22
IMGSZ_CARDS, IMGSZ_ARENA = 960, 960
DECK_HEIGHT_FRACTION = 0.24
ARENA_SCALE = 2.0  # ONLY used for troops
WINDOW_NAME = "CR Vision Tester"

cards_model  = YOLO(CARDS_PT)
troops_model = YOLO(TROOPS_PT)
towers_model = YOLO(TOWERS_PT)

def _find_window_rect(title_substr: str):
    target = {"hwnd": None}
    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        t = win32gui.GetWindowText(hwnd)
        if t and title_substr.lower() in t.lower():
            target["hwnd"] = hwnd
    win32gui.EnumWindows(enum_cb, None)
    hwnd = target["hwnd"]
    if not hwnd:
        return None, None
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    l, t, r, b = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = left + (r - l), top + (b - t)
    return hwnd, (left, top, right, bottom)

def _grab_client_region(mss_obj, rect):
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    shot = mss_obj.grab({"left": left, "top": top, "width": w, "height": h})
    return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR).copy()

def _predict(model, frame_bgr, conf, imgsz, augment=False):
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    r = model.predict(img, conf=conf, imgsz=imgsz, device=0, verbose=False, augment=augment)[0]
    out = []
    names = r.names
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return out
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0]); score = float(b.conf[0])
        out.append((names[cls], score, (x1, y1, x2, y2)))
    return out

def _draw(frame, dets, color):
    for label, score, (x1, y1, x2, y2) in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def _downscale(dets, sx, sy):
    invx, invy = 1.0/sx, 1.0/sy
    return [(lbl, sc, (int(x1*invx), int(y1*invy), int(x2*invx), int(y2*invy)))
            for (lbl, sc, (x1, y1, x2, y2)) in dets]

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
            H, W, _ = frame.shape

            # ---- Cards: bottom deck ROI (UNCHANGED) ----
            deck_h = int(DECK_HEIGHT_FRACTION * H)
            y0 = max(0, H - deck_h)
            deck_roi = frame[y0:H, 0:W]
            det_cards = _predict(cards_model, deck_roi, CONF_CARDS, IMGSZ_CARDS, augment=False)
            det_cards = [(lbl, sc, (x1, y1 + y0, x2, y2 + y0))
                         for (lbl, sc, (x1, y1, x2, y2)) in det_cards]

            # ---- Troops: arena-only crop, upscale, predict, downscale back ----
            arena = frame[0:y0, 0:W]
            arena_up = cv2.resize(arena, None, fx=ARENA_SCALE, fy=ARENA_SCALE, interpolation=cv2.INTER_LINEAR)
            det_troops_up = _predict(troops_model, arena_up, CONF_TROOPS, 1280, augment=True)
            det_troops = _downscale(det_troops_up, ARENA_SCALE, ARENA_SCALE)

            # ---- Towers: FULL FRAME (UNCHANGED) ----
            det_towers = _predict(towers_model, frame, CONF_TOWERS, IMGSZ_ARENA, augment=False)

            # draw
            _draw(frame, det_cards,  (0, 255, 0))     # cards
            _draw(frame, det_troops, (255, 200, 0))   # troops
            _draw(frame, det_towers, (0, 200, 255))   # towers

            # HUD
            dt = max(time.time() - t0, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS: {fps:.1f}  cards:{len(det_cards)} troops:{len(det_troops)} towers:{len(det_towers)}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cv2.destroyWindow(WINDOW_NAME)

if __name__ == "__main__":
    main()
