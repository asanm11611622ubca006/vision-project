"""
Plate Detection module — uses OpenCV Haar Cascade with geometry filtering,
watermark zone exclusion, and vehicle-bounded detection.
"""
from typing import List, Tuple
import cv2
import numpy as np
from src import config


class PlateDetector:
    """Detects rectangular licence-plate regions in video frames."""

    def __init__(self, cascade_path: str = None):
        path = str(cascade_path or config.CASCADE_PATH)
        self.cascade = None
        if hasattr(cv2, "CascadeClassifier"):
            try:
                clf = cv2.CascadeClassifier(path)
                if clf and not clf.empty():
                    self.cascade = clf
            except Exception:
                self.cascade = None

    def detect_plates(
        self, frame: np.ndarray, exclude_watermarks: bool = True
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Detect licence plates in *frame*.

        Returns a list of ``(x, y, w, h, cropped_roi)`` tuples for every
        candidate that passes the area, aspect-ratio, and border filters.
        """
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        candidates = []

        # 1. Haar Cascade (close-up & standard plates)
        if self.cascade is not None:
            try:
                raw = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=config.SCALE_FACTOR,
                    minNeighbors=config.MIN_NEIGHBORS,
                    minSize=(25, 8),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for x, y, w, h in raw:
                    candidates.append([int(x), int(y), int(w), int(h)])
            except Exception:
                pass


        # 2. Morphological Edge Detection (captures blue, yellow, and distant plates)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad = cv2.convertScaleAbs(grad_x)
        blurred = cv2.GaussianBlur(abs_grad, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close horizontal gaps between characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = float(w) / float(h) if h > 0 else 0
            if (
                config.MIN_PLATE_AREA <= area <= config.MAX_PLATE_AREA
                and config.MIN_ASPECT_RATIO <= aspect <= config.MAX_ASPECT_RATIO
                and w >= 25
                and h >= 8
            ):
                roi_g = gray[y : y + h, x : x + w]
                if roi_g.size > 0 and np.std(roi_g) > 18:
                    candidates.append([int(x), int(y), int(w), int(h)])

        if not candidates:
            return []

        # 3. Non-Maximum Suppression (merge overlapping boxes)
        indices = cv2.dnn.NMSBoxes(
            candidates,
            [1.0] * len(candidates),
            score_threshold=0.5,
            nms_threshold=0.35,
        )

        results: List[Tuple[int, int, int, int, np.ndarray]] = []
        if len(indices) > 0:
            top_y_limit = int(frame_h * config.EXCLUDE_TOP_BORDER_PCT) if exclude_watermarks else 0
            bottom_y_limit = int(frame_h * (1.0 - config.EXCLUDE_BOTTOM_BORDER_PCT)) if exclude_watermarks else frame_h

            for idx in indices.flatten():
                x, y, w, h = candidates[idx]

                # Exclude CCTV timestamps / camera name banners at top or bottom borders
                if exclude_watermarks and (y < top_y_limit or (y + h) > bottom_y_limit):
                    continue

                area = w * h
                aspect = float(w) / float(h)

                if not (config.MIN_PLATE_AREA <= area <= config.MAX_PLATE_AREA):
                    continue
                if not (config.MIN_ASPECT_RATIO <= aspect <= config.MAX_ASPECT_RATIO):
                    continue

                # Generous padding so edge characters are not clipped
                pad_x = int(w * 0.08)
                pad_y = int(h * 0.12)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame_w, x + w + pad_x)
                y2 = min(frame_h, y + h + pad_y)

                roi = frame[y1:y2, x1:x2].copy()
                results.append((x, y, w, h, roi))

        return results

    def detect_in_vehicle_crop(
        self, frame: np.ndarray, vx1: int, vy1: int, vx2: int, vy2: int
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Run plate detection specifically on the lower bumper/grille of a vehicle.
        Coordinates are returned in original frame space.
        """
        vh = vy2 - vy1
        vw = vx2 - vx1
        if vw < 30 or vh < 25:
            return []

        # Target lower 75% of vehicle where plates reside
        by1 = max(0, vy1 + int(vh * 0.25))
        by2 = min(frame.shape[0], vy2)
        bx1 = max(0, vx1)
        bx2 = min(frame.shape[1], vx2)

        v_crop = frame[by1:by2, bx1:bx2]
        if v_crop.size == 0:
            return []

        # Detect candidate plates in the vehicle bumper crop
        local_cands = self.detect_plates(v_crop, exclude_watermarks=False)
        results = []
        for lx, ly, lw, lh, lroi in local_cands:
            gx = bx1 + lx
            gy = by1 + ly
            results.append((gx, gy, lw, lh, lroi))

        if not results:
            bw = int(vw * 0.85)
            bh = int(vh * 0.45)
            bx = bx1 + int((vw - bw) / 2)
            by = by1 + int((vh * 0.75 - bh) / 2)
            crop = frame[max(0, by):min(frame.shape[0], by + bh), max(0, bx):min(frame.shape[1], bx + bw)]
            if crop.size > 0:
                results.append((bx, by, bw, bh, crop))

        return results
