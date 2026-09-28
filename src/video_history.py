"""
Persistent video history storage.
Keeps track of all uploaded videos, upload timestamps, processing status,
and detected plates across browser refreshes and server restarts.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config

HISTORY_FILE = config.DATA_DIR / "video_history.json"


def _read_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_history(records: List[Dict[str, Any]]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def get_all_videos() -> List[Dict[str, Any]]:
    """Return all uploaded videos sorted by newest first."""
    records = _read_history()
    # Sort descending by upload_timestamp
    records.sort(key=lambda r: r.get("upload_timestamp", ""), reverse=True)
    return records


def add_video(
    jid: str,
    filename: str,
    video_url: str,
    output_video_url: str = "",
    status: str = "processing",
    unique_count: int = 0,
    unique_plates: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Add a new uploaded video record to persistent history."""
    records = _read_history()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_display = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # If already exists, update it
    for r in records:
        if r.get("job_id") == jid:
            r["filename"] = filename
            r["status"] = status
            r["video_url"] = video_url
            if output_video_url:
                r["output_video_url"] = output_video_url
            _write_history(records)
            return r

    new_record = {
        "job_id": jid,
        "filename": filename,
        "upload_timestamp": now_str,
        "upload_display": now_display,
        "video_url": video_url,
        "output_video_url": output_video_url,
        "status": status,
        "unique_count": unique_count,
        "unique_plates": unique_plates or [],
    }
    records.insert(0, new_record)
    _write_history(records)
    return new_record


def update_video_result(
    jid: str,
    status: str,
    unique_count: int,
    unique_plates: List[dict],
    output_video_url: str,
    fps: float = 0.0,
    total_vehicles: int = 0,
    plates_detected: int = 0,
    missing_plates: int = 0,
    alert_plates: int = 0,
):
    """Update video record when processing completes."""
    records = _read_history()
    for r in records:
        if r.get("job_id") == jid:
            r["status"] = status
            r["unique_count"] = unique_count
            r["unique_plates"] = unique_plates
            r["output_video_url"] = output_video_url
            r["fps"] = fps
            r["total_vehicles"] = total_vehicles or unique_count
            r["plates_detected"] = plates_detected
            r["missing_plates"] = missing_plates
            r["alert_plates"] = alert_plates
            break
    _write_history(records)


def get_video(jid: str) -> Optional[Dict[str, Any]]:
    records = _read_history()
    for r in records:
        if r.get("job_id") == jid:
            return r
    return None


def delete_video(jid: str) -> bool:
    records = _read_history()
    initial_len = len(records)
    records = [r for r in records if r.get("job_id") != jid]
    if len(records) < initial_len:
        _write_history(records)
        return True
    return False
