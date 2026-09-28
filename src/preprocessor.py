"""
Image preprocessing pipeline for licence-plate crops before OCR.
Tries three enhancement variants and returns the sharpest result.
"""
import cv2
import numpy as np


class PlatePreprocessor:
    """Cleans plate ROI images to maximise OCR accuracy."""

    def preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Enhances plate ROI for OCR:
        1. Bicubic upscaling if the crop is small (CCTV / distant vehicles)
        2. Contrast enhancement & adaptive thresholding
        3. Returns the highest-contrast image for OCR
        """
        if roi is None or roi.size == 0:
            return roi

        h, w = roi.shape[:2]
        # Upscale small crops so characters have sufficient pixel height for OCR
        if h < 64 or w < 140:
            scale = max(64.0 / max(h, 1), 140.0 / max(w, 1), 2.0)
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Invert if plate has light text on dark background (blue/black plates)
        gh, gw = gray.shape
        border_mean = (np.mean(gray[:4, :]) + np.mean(gray[-4:, :]) + np.mean(gray[:, :4]) + np.mean(gray[:, -4:])) / 4.0
        center_mean = np.mean(gray[gh // 4 : 3 * gh // 4, gw // 4 : 3 * gw // 4])
        if border_mean < center_mean - 10:
            gray = 255 - gray

        variants = [
            self._variant_bilateral_otsu(gray),
            self._variant_clahe_binary(gray),
            self._variant_adaptive(gray),
            gray,
        ]

        # Pick the variant with the most edge pixels (sharpest text)
        best = max(variants, key=self._edge_density)
        return best

    # ── Variant A: Bilateral filter → Otsu threshold ──────────────
    @staticmethod
    def _variant_bilateral_otsu(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # ── Variant B: CLAHE → fixed binary threshold ─────────────────
    @staticmethod
    def _variant_clahe_binary(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        return thresh

    # ── Variant C: Gaussian blur → adaptive threshold ─────────────
    @staticmethod
    def _variant_adaptive(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ── Scoring helper ────────────────────────────────────────────
    @staticmethod
    def _edge_density(img: np.ndarray) -> float:
        """Returns the fraction of edge pixels (Canny) in the image."""
        edges = cv2.Canny(img, 100, 200)
        return float(np.count_nonzero(edges)) / max(edges.size, 1)
