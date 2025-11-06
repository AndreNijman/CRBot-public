from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pyautogui as pag
from PIL import Image
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crbot.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "yolo_models"

MODES = ["cards", "troops", "towers", "arena"]  # arena = troops + towers overlay


def load_models(device: str | int) -> dict[str, YOLO]:
    return {
        "cards": YOLO(MODEL_DIR / "cards.pt"),
        "troops": YOLO(MODEL_DIR / "troops.pt"),
        "towers": YOLO(MODEL_DIR / "towers.pt"),
    }


def predict(model: YOLO, frame_bgr: np.ndarray, *, conf: float, imgsz: int, device: str | int):
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    result = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    dets = []
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        score = float(box.conf[0])
        dets.append((names[cls], score, (int(x1), int(y1), int(x2), int(y2))))
    return dets


def draw_detections(frame, detections, color):
    for label, score, (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def grab_frame() -> np.ndarray:
    shot = pag.screenshot()
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live object detection visualiser.")
    parser.add_argument("--device", default=0, help="YOLO device (GPU index or 'cpu').")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    models = load_models(args.device)
    mode_idx = 0
    fps = 0.0

    print("Controls: 1=cards  2=troops  3=towers  4=arena  q=quit")

    while True:
        frame = grab_frame()
        start = time.time()

        mode = MODES[mode_idx]
        if mode == "cards":
            dets = predict(models["cards"], frame, conf=0.35, imgsz=640, device=args.device)
            draw_detections(frame, dets, (0, 255, 0))
        elif mode == "troops":
            dets = predict(models["troops"], frame, conf=0.30, imgsz=640, device=args.device)
            draw_detections(frame, dets, (255, 200, 0))
        elif mode == "towers":
            dets = predict(models["towers"], frame, conf=0.30, imgsz=640, device=args.device)
            draw_detections(frame, dets, (0, 200, 255))
        else:
            d1 = predict(models["troops"], frame, conf=0.30, imgsz=640, device=args.device)
            d2 = predict(models["towers"], frame, conf=0.30, imgsz=640, device=args.device)
            draw_detections(frame, d1, (255, 200, 0))
            draw_detections(frame, d2, (0, 200, 255))

        elapsed = time.time() - start
        fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6))

        cv2.putText(
            frame,
            f"Mode: {mode}  FPS: {fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("CR Vision Tester", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            mode_idx = 0
        elif key == ord("2"):
            mode_idx = 1
        elif key == ord("3"):
            mode_idx = 2
        elif key == ord("4"):
            mode_idx = 3

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
