"""
OCR Engine — EasyOCR wrapper with text sanitisation, common OCR corrections,
watermark/brand filtering, and character-count validation.
"""
import warnings
warnings.filterwarnings("ignore")

import re
from typing import Tuple

import cv2
import easyocr
import numpy as np

from src import config
from src.preprocessor import PlatePreprocessor

# Common OCR misreads on licence plates
_OCR_CORRECTIONS = str.maketrans({
    "O": "0",  # letter O  →  digit 0
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
})

# Words that should never be recorded as a licence plate
BLOCKED_PATTERNS = {
    "CAMERA", "IPCAM", "CAM", "IPCAMERA", "IPC", "VIDEO", "CHANNEL",
    "ISUZU", "TOYOTA", "HONDA", "HYUNDAI", "SUZUKI", "NISSAN", "FORD",
    "MERCEDES", "BMW", "CHEVROLET", "VOLKSWAGEN", "MAHINDRA", "TATA",
    "TIME", "DATE", "HDVR", "NVR", "DVR", "AUTO", "STOP", "SLOW",
    "POLICE", "AMBULANCE", "CAR", "BUS", "TRUCK"
}


class PlateOCREngine:
    """Extracts registration text from cropped plate images."""

    def __init__(self, use_gpu: bool = False):
        print("[INFO] Initialising EasyOCR engine (CPU mode)…")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        self.preprocessor = PlatePreprocessor()

    # ────────────────────────────────────────────────────────────
    def is_valid_plate(self, text: str) -> bool:
        """Verify that the detected text is a genuine vehicle license plate."""
        if not text:
            return False
        t = text.upper()
        if not (config.OCR_MIN_CHARS <= len(t) <= config.OCR_MAX_CHARS):
            return False

        # Reject CCTV watermarks, vehicle brand logos, and generic words
        for blocked in BLOCKED_PATTERNS:
            if blocked in t or (len(t) <= 5 and t in blocked):
                return False

        # Genuine plates must contain at least 1 letter and at least 2 digits
        num_digits = sum(1 for c in t if c.isdigit())
        num_letters = sum(1 for c in t if c.isalpha())
        if num_digits < 2 or num_letters < 1:
            return False

        return True

    # ────────────────────────────────────────────────────────────
    def recognize_plate_sync(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Run multi-variant preprocessing + OCR on a plate crop.

        Returns ``(sanitised_plate_text, confidence)``.
        An empty string and 0.0 confidence are returned when recognition
        fails or the result does not pass validation.
        """
        if roi is None or roi.size == 0:
            return "", 0.0

        # Fast pre-check: drop flat-color crops (plain car paint, bumper plastic) instantly
        raw_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        if raw_gray.size == 0 or float(np.std(raw_gray)) < config.MIN_PLATE_TEXTURE_STD:
            return "", 0.0

        h, w = roi.shape[:2]
        # Upscale small crops so characters have sufficient pixel height for OCR
        if h < 64 or w < 140:
            scale = max(64.0 / max(h, 1), 140.0 / max(w, 1), 2.2)
            roi_scaled = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            roi_scaled = roi

        gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY) if len(roi_scaled.shape) == 3 else roi_scaled
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        # Check CLAHE first (highest success rate on real plates)
        # Only fall back to other variants if CLAHE yields no valid plate or low confidence
        best_plate = ""
        best_conf = 0.0

        for img_var in [clahe]:
            try:
                results = self.reader.readtext(
                    img_var,
                    allowlist=config.OCR_ALLOWLIST,
                    detail=1,
                    paragraph=False,
                )
            except Exception:
                continue

            if not results:
                continue

            combined = ""
            confidences = []
            for _bbox, text, conf in results:
                cleaned = self._clean_raw(text)
                if cleaned:
                    combined += cleaned
                    confidences.append(conf)

            if not combined:
                continue

            plate = self._apply_corrections(combined)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            if self.is_valid_plate(plate) and avg_conf > best_conf:
                best_plate = plate
                best_conf = avg_conf

                # Early-exit: If we already have a solid reading, do not waste time on extra passes
                if best_conf >= 0.50:
                    break

        if best_conf < config.OCR_CONFIDENCE_THRESHOLD:
            return "", 0.0

        return best_plate, best_conf

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_raw(text: str) -> str:
        """Strip non-alphanumeric noise and uppercase."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        # Remove common country-badge prefixes that sometimes get OCR'd
        if cleaned.startswith("IND") and len(cleaned) > 5:
            cleaned = cleaned[3:]
        elif cleaned.startswith("EU") and len(cleaned) > 5:
            cleaned = cleaned[2:]
        return cleaned

    @staticmethod
    def _apply_corrections(text: str) -> str:
        """
        Apply context-aware letter <-> digit corrections.

        For plates matching the Indian pattern (e.g. DL01CA1234), positions
        2-3 and 6+ are corrected to digits.
        For general / international / short plates, text is preserved as-is.
        """
        if len(text) >= 8 and text[:2].isalpha():
            # Looks like Indian XX00XX0000 format
            prefix = text[:2]
            digits1 = text[2:4].translate(_OCR_CORRECTIONS)
            letters = text[4:6]
            digits2 = text[6:].translate(_OCR_CORRECTIONS) if len(text) > 6 else ""
            return prefix + digits1 + letters + digits2

        return text
