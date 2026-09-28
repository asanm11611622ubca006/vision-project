"""
Central configuration for the Vehicle Number Plate Recognition System.
All tunable paths, thresholds, and constants live here.
"""
from pathlib import Path

# ──────────────────────── Paths ────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
PLATES_DIR = DATA_DIR / "plates"
EXCEL_PATH = DATA_DIR / "detections.xlsx"

# ──────────────────────── Plate Detection (Haar Cascade & Contours) ───────────
CASCADE_PATH = MODELS_DIR / "haarcascade_russian_plate_number.xml"
SCALE_FACTOR = 1.08
MIN_NEIGHBORS = 3
MIN_PLATE_AREA = 300        # minimum bounding-box area in pixels (allows distant / CCTV plates)
MAX_PLATE_AREA = 250000     # maximum bounding-box area
MIN_ASPECT_RATIO = 1.3      # width / height lower bound (supports 2-line & angled plates)
MAX_ASPECT_RATIO = 6.5      # width / height upper bound

# ──────────────────────── OCR ────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.20  # Balances catching distant plates with dropping pure noise
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_MIN_CHARS = 4                # Allows 4-10 character plates (e.g. AP09CP5546, TN50AP8219)
OCR_MAX_CHARS = 12

# ──────────────────────── Watermark & Border Exclusion ────────
EXCLUDE_TOP_BORDER_PCT = 0.08      # Ignore top 8% of frame (CCTV timestamp / camera name)
EXCLUDE_BOTTOM_BORDER_PCT = 0.04   # Ignore bottom 4% of frame

# ──────────────────────── Deduplication ────────────────────────
FUZZY_MATCH_MAX_EDITS = 2   # Plates within 2 edits on 8-10 char strings are considered the same vehicle

# ──────────────────────── Performance / Speed ────────────────────────
DEFAULT_FRAME_STRIDE = 6       # Adaptive sampling adjusts per video FPS
MAX_OCR_CANDIDATES = 2         # Only run deep OCR on top 2 most plate-like candidates per frame
MIN_PLATE_TEXTURE_STD = 14.0   # Captures soft/CCTV plates without dropping moving cars

# ──────────────────────── Special Alert Plates (Red Highlight) ────────
ALERT_PLATES = ["TN 50A P8219", "TN50AP8219"]

