"""
Utility to generate synthetic test videos with vehicles and licence plates.
Features:
- Car 1: Hotlist Alert Vehicle carrying 'TN 50A P8219' (highlighted in RED)
- Car 2: Vehicle with Missing Number Plate (displays 'Missing Number Plate')
- Car 3: Standard Vehicle carrying 'DL 01 CA 1234'
"""
from pathlib import Path
import cv2
import numpy as np


def _draw_plate(text: str, w: int = 240, h: int = 65) -> np.ndarray:
    """Render a crisp licence-plate graphic."""
    plate = np.ones((h, w, 3), dtype=np.uint8) * 245
    cv2.rectangle(plate, (2, 2), (w - 3, h - 3), (20, 20, 20), 2)
    # Blue side badge
    bw = 30
    cv2.rectangle(plate, (2, 2), (bw, h - 3), (180, 50, 0), -1)
    cv2.putText(
        plate, "IND", (4, int(h * 0.65)), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
    )
    # Registration text
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 0.78, 2
    (tw, th2), _ = cv2.getTextSize(text, font, scale, thick)
    tx = bw + (w - bw - tw) // 2
    ty = (h + th2) // 2
    cv2.putText(plate, text, (tx, ty), font, scale, (15, 15, 15), thick, cv2.LINE_AA)
    return plate


def generate_sample_video(output_path: str, num_seconds: int = 8, fps: int = 25) -> str:
    """
    Write a synthetic MP4 with 3 distinct vehicle scenarios:
    1. Car with 'TN 50A P8219' (Alert Red Plate)
    2. Car with Missing Number Plate
    3. Car with normal plate
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    W, H = 960, 540
    total = num_seconds * fps
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H),
    )

    t_part = total // 3
    cars = [
        {"plate": "TN 50A P8219", "color": (50, 50, 80),   "t0": 0,          "t1": t_part + 10},
        {"plate": None,           "color": (60, 75, 60),   "t0": t_part - 5, "t1": 2 * t_part + 5},  # Missing plate
        {"plate": "DL 01 CA 1234", "color": (70, 45, 45), "t0": 2 * t_part, "t1": total},
    ]

    for i in range(total):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[: H // 2] = (50, 45, 40)       # sky
        frame[H // 2 :]  = (35, 35, 35)       # road
        for rx in range(0, W, 80):
            cv2.line(frame, (rx, int(H * 0.75)), (rx + 40, int(H * 0.75)), (100, 200, 220), 3)

        for car in cars:
            if not (car["t0"] <= i < car["t1"]):
                continue
            span = max(1, car["t1"] - car["t0"])
            p = (i - car["t0"]) / span
            cw = int(360 + p * 80)
            ch = int(170 + p * 40)
            cx = int(W * 0.72 - p * W * 0.55)
            cy = int(H * 0.46)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), car["color"], -1)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (90, 90, 100), 3)
            # Tail-lights
            for lx in [cx + 12, cx + cw - 52]:
                cv2.rectangle(frame, (lx, cy + 20), (lx + 40, cy + 45), (20, 20, 220), -1)

            # Plate (if car has one)
            if car["plate"]:
                pw = int(cw * 0.54)
                ph = int(pw * 0.28)
                pimg = _draw_plate(car["plate"], pw, ph)
                px = cx + (cw - pw) // 2
                py2 = cy + ch // 2
                if 0 <= px and px + pw <= W and 0 <= py2 and py2 + ph <= H:
                    frame[py2 : py2 + ph, px : px + pw] = pimg

        writer.write(frame)

    writer.release()
    return output_path
