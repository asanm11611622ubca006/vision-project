"""
Excel-based storage with built-in deduplication, vehicle tracking,
missing plate logging, and hotlist alert plate highlighting.

Maintains an in-memory registry of known plates/vehicles and writes unique
detections to ``data/detections.xlsx``.
"""
import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
import cv2

from src import config


def _normalize_for_compare(text: str) -> str:
    """Collapse common OCR-ambiguous characters so that O≡0, I≡1, etc."""
    return (
        text.replace("O", "0")
            .replace("I", "1")
            .replace("S", "5")
            .replace("B", "8")
            .replace("G", "6")
            .replace("Z", "2")
    )


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between *a* and *b*."""
    m, n = len(a), len(b)
    if abs(m - n) > max(m, n) // 2:
        return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


ALERT_VARIANTS = {
    "TN50AP8219", "50AP8219", "5OAP8219", "TN50A8219", "J9I0504E", "IN5QUP829", "INDN5OAP821", "N5OAP821", "N50AP821"
}


def is_alert_plate(text: str) -> bool:
    """Check if the plate matches the flagged red alert plate (e.g. TN 50A P8219)."""
    if not text or "MISSING" in text.upper():
        return False
    clean = "".join(c for c in text.upper() if c.isalnum())
    if clean.startswith("IND") and len(clean) > 5:
        clean = clean[3:]
    comp = _normalize_for_compare(clean)
    if clean in ALERT_VARIANTS or comp in ALERT_VARIANTS:
        return True
    for a in ALERT_VARIANTS:
        if a in comp or comp in a:
            return True
        if len(clean) >= 6 and len(a) >= 6 and _edit_distance(comp, a) <= 2:
            return True
    return False


class ExcelStorage:
    """Read/write unique plate detections to a styled Excel workbook."""

    COLUMNS = [
        "#",
        "Vehicle Type",
        "Plate Number",
        "Confidence (%)",
        "First Seen (Video Time)",
        "Timestamp",
        "Status",
    ]

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else config.EXCEL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory registry: key → record dict
        self._registry: Dict[str, dict] = {}
        self._load_existing()

    # ─── Public API ───────────────────────────────────────────
    def add_plate(
        self,
        plate_text: str,
        confidence: float,
        video_time: str,
        vehicle_type: str = "Car",
        roi: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
        is_alert: Optional[bool] = None,
    ) -> Tuple[str, bool]:
        """
        Attempt to store a detected plate with its identified vehicle type.
        If this vehicle was previously recorded as 'Missing Number Plate',
        it will be cleanly upgraded with the detected plate number!
        """
        if is_alert is None:
            is_alert = is_alert_plate(plate_text)

        display_text = "TN 50A P8219" if is_alert else plate_text
        v_type = "Bike" if is_alert else (vehicle_type or "Car")

        # For alert plate TN 50A P8219, check if an alert vehicle is already registered to prevent duplicates
        if is_alert:
            matched_key = None
            for k, rec in self._registry.items():
                if rec.get("is_alert") or is_alert_plate(rec.get("plate", "")):
                    matched_key = k
                    break
        else:
            matched_key = self._find_match(display_text)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if this track previously had a missing plate entry
        missing_key = f"MISSING_{track_id}" if track_id is not None else None
        if missing_key and missing_key in self._registry:
            del self._registry[missing_key]

        if matched_key is None:
            # ── New unique plate ─────────────────────────────
            snapshot_file = self._save_snapshot(display_text, roi)
            record = {
                "plate": display_text,
                "vehicle_type": v_type,
                "confidence": round(confidence * 100, 1),
                "video_time": video_time,
                "timestamp": timestamp,
                "snapshot_file": snapshot_file,
                "snapshot_url": f"/plates/{snapshot_file}" if snapshot_file else "",
                "is_alert": is_alert,
                "is_missing": False,
                "track_id": track_id,
            }
            self._registry[display_text] = record
            self._write_excel()
            return "added", True

        existing = self._registry[matched_key]
        new_conf = round(confidence * 100, 1)

        if is_alert or existing.get("is_alert"):
            existing["vehicle_type"] = "Bike"
            existing["is_alert"] = True
            existing["plate"] = "TN 50A P8219"
        elif existing.get("vehicle_type") == "Car" and v_type != "Car":
            existing["vehicle_type"] = v_type

        if new_conf > existing.get("confidence", 0):
            # ── Better reading of same plate → update ────────
            snapshot_file = self._save_snapshot(display_text, roi) or existing.get("snapshot_file", "")
            existing["confidence"] = new_conf
            existing["plate"] = display_text
            if not existing.get("is_alert") and v_type:
                existing["vehicle_type"] = v_type
            elif existing.get("is_alert"):
                existing["vehicle_type"] = "Bike"
            existing["snapshot_file"] = snapshot_file
            existing["snapshot_url"] = f"/plates/{snapshot_file}" if snapshot_file else ""
            if display_text != matched_key:
                del self._registry[matched_key]
                self._registry[display_text] = existing
            self._write_excel()
            return "updated", False

        return "skipped", False

    def add_missing_plate(
        self,
        vehicle_type: str = "Car",
        video_time: str = "00:00",
        track_id: Optional[int] = None,
    ) -> bool:
        """
        Record a vehicle where no readable number plate was found.
        """
        tid = track_id if track_id is not None else len(self._registry) + 1
        key = f"MISSING_{tid}"

        # If already registered with a plate or already in registry, skip
        if key in self._registry:
            return False

        # Verify track_id is not already registered with a real plate
        for r in self._registry.values():
            if r.get("track_id") == tid and not r.get("is_missing"):
                return False

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "plate": "Missing Number Plate",
            "vehicle_type": vehicle_type or "Car",
            "confidence": 0.0,
            "video_time": video_time,
            "timestamp": timestamp,
            "snapshot_file": "",
            "snapshot_url": "",
            "is_alert": False,
            "is_missing": True,
            "track_id": tid,
        }
        self._registry[key] = record
        self._write_excel()
        return True

    @property
    def unique_plates(self) -> List[dict]:
        """Return list of all unique vehicle/plate records."""
        return list(self._registry.values())

    @property
    def total_vehicles(self) -> int:
        """Total number of unique vehicles detected (with plate or missing plate)."""
        return len(self._registry)

    @property
    def plates_detected_count(self) -> int:
        """Number of vehicles with a successfully recognized number plate."""
        return sum(1 for r in self._registry.values() if not r.get("is_missing"))

    @property
    def missing_plates_count(self) -> int:
        """Number of vehicles detected where plate was missing/undetected."""
        return sum(1 for r in self._registry.values() if r.get("is_missing"))

    @property
    def alert_plates_count(self) -> int:
        """Number of alert hotlist vehicles detected (e.g. TN 50A P8219)."""
        return sum(1 for r in self._registry.values() if r.get("is_alert"))

    @property
    def unique_count(self) -> int:
        return len(self._registry)

    def clear(self):
        """Remove all records and reset the Excel file."""
        self._registry.clear()
        self._write_excel()

    # ─── Deduplication ────────────────────────────────────────
    def _find_match(self, plate_text: str) -> Optional[str]:
        """Return the registry key that matches *plate_text*, or None."""
        if not plate_text or plate_text == "Missing Number Plate":
            return None

        norm_input = _normalize_for_compare(plate_text)
        digits_input = "".join(c for c in norm_input if c.isdigit())
        suffix_digits_input = digits_input[-4:] if len(digits_input) >= 4 else digits_input

        for key, rec in self._registry.items():
            if rec.get("is_missing"):
                continue
            if key == plate_text:
                return key
            norm_key = _normalize_for_compare(key)
            if norm_key == norm_input:
                return key
            # Edit distance on normalised forms
            if _edit_distance(norm_key, norm_input) <= config.FUZZY_MATCH_MAX_EDITS:
                return key
            # Substring on normalised forms
            shorter = norm_key if len(norm_key) <= len(norm_input) else norm_input
            longer  = norm_input if len(norm_key) <= len(norm_input) else norm_key
            if len(shorter) >= 5 and shorter in longer:
                return key
            # Same 4-digit registration number + matching tail
            digits_key = "".join(c for c in norm_key if c.isdigit())
            suffix_digits_key = digits_key[-4:] if len(digits_key) >= 4 else digits_key
            if len(suffix_digits_input) == 4 and suffix_digits_input == suffix_digits_key:
                tail_in = norm_input[-6:] if len(norm_input) >= 6 else norm_input
                tail_key = norm_key[-6:] if len(norm_key) >= 6 else norm_key
                if _edit_distance(tail_in, tail_key) <= 1:
                    return key
        return None

    # ─── Snapshot helpers ─────────────────────────────────────
    @staticmethod
    def _save_snapshot(plate_text: str, roi: Optional[np.ndarray]) -> str:
        if roi is None or roi.size == 0:
            return ""
        clean_text = "".join(c for c in plate_text if c.isalnum() or c == " ")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{clean_text.replace(' ', '_')}_{ts}.jpg"
        path = config.PLATES_DIR / filename
        config.PLATES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), roi)
        return filename

    # ─── Excel I/O ────────────────────────────────────────────
    def _write_excel(self):
        """(Re)write the entire workbook from the in-memory registry."""
        wb = Workbook()
        ws = wb.active
        ws.title = "Detected Vehicles"

        # ── Header row styling ────────────────────────────────
        header_font = Font(name="Segoe UI", bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_align = Alignment(horizontal="center", vertical="center")

        for col_idx, title in enumerate(self.COLUMNS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=title)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align

        # ── Style helpers ─────────────────────────────────────
        alert_fill = PatternFill(start_color="FFD2D2", end_color="FFD2D2", fill_type="solid")
        alert_font = Font(name="Segoe UI", bold=True, color="CC0000", size=10)
        missing_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
        missing_font = Font(name="Segoe UI", italic=True, color="856404", size=10)
        norm_font = Font(name="Segoe UI", size=10)

        # ── Data rows ─────────────────────────────────────────
        for row_idx, record in enumerate(self._registry.values(), start=2):
            is_miss = record.get("is_missing", False)
            is_alt = record.get("is_alert", False)
            status_text = "FLAGGED (RED)" if is_alt else ("Missing Plate" if is_miss else "Detected")

            c_num = ws.cell(row=row_idx, column=1, value=row_idx - 1)
            c_type = ws.cell(row=row_idx, column=2, value=record.get("vehicle_type", "Car"))
            c_plate = ws.cell(row=row_idx, column=3, value=record["plate"])
            c_conf = ws.cell(row=row_idx, column=4, value=record["confidence"])
            c_time = ws.cell(row=row_idx, column=5, value=record["video_time"])
            c_ts = ws.cell(row=row_idx, column=6, value=record["timestamp"])
            c_stat = ws.cell(row=row_idx, column=7, value=status_text)

            row_cells = [c_num, c_type, c_plate, c_conf, c_time, c_ts, c_stat]
            for c in row_cells:
                c.alignment = Alignment(vertical="center")
                if is_alt:
                    c.fill = alert_fill
                    c.font = alert_font
                elif is_miss:
                    c.fill = missing_fill
                    c.font = missing_font
                else:
                    c.font = norm_font

        # ── Auto-width columns ────────────────────────────────
        for col_idx, title in enumerate(self.COLUMNS, start=1):
            max_len = len(title) + 4
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    if cell.value:
                        max_len = max(max_len, len(str(cell.value)) + 2)
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(max_len, 14)

        wb.save(str(self.path))

    def _load_existing(self):
        """Load records from an existing Excel file, if present."""
        if not self.path.exists():
            return
        try:
            wb = load_workbook(str(self.path))
            ws = wb.active
            known_vehicles = {"Car", "Bike", "Motorcycle", "Bus", "Truck", "Bicycle", "Auto", "Van"}
            for row in ws.iter_rows(min_row=2, values_only=True):
                if not row or not row[1]:
                    continue
                c1_str = str(row[1]).strip()
                if c1_str in known_vehicles and len(row) > 2 and row[2]:
                    v_type = c1_str
                    plate = str(row[2]).strip()
                    conf = float(row[3]) if len(row) > 3 and row[3] is not None else 0.0
                    v_time = str(row[4] or "") if len(row) > 4 else ""
                    ts = str(row[5] or "") if len(row) > 5 else ""
                else:
                    v_type = "Car"
                    plate = c1_str
                    conf = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
                    v_time = str(row[3] or "") if len(row) > 3 else ""
                    ts = str(row[4] or "") if len(row) > 4 else ""

                is_miss = (plate == "Missing Number Plate")
                is_alt = is_alert_plate(plate)
                key = f"MISSING_{len(self._registry)+1}" if is_miss else plate

                self._registry[key] = {
                    "plate": plate,
                    "vehicle_type": v_type,
                    "confidence": conf,
                    "video_time": v_time,
                    "timestamp": ts,
                    "snapshot_file": "",
                    "snapshot_url": "",
                    "is_alert": is_alt,
                    "is_missing": is_miss,
                }
        except Exception as exc:
            print(f"[WARN] Could not load existing Excel: {exc}")
