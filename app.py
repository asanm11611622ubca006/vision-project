"""
FastAPI Web Server — VisionPlate AI
====================================
Upload a video → frame-by-frame plate detection + OCR → deduplicated Excel storage → live dashboard.

Run:
    python app.py
    # Opens at http://127.0.0.1:8000
"""
import warnings
warnings.filterwarnings("ignore")

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.sample_generator import generate_sample_video
from src.video_processor import VideoProcessor

# ─────────────────────────────────────────────────────────────
app = FastAPI(title="VisionPlate AI", version="2.0.0")

BASE = Path(__file__).resolve().parent
WEB   = BASE / "web"
for d in [config.UPLOADS_DIR, config.OUTPUTS_DIR, config.PLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/plates",  StaticFiles(directory=str(config.PLATES_DIR)),  name="plates")
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOADS_DIR)), name="uploads")

from src import video_history

# In-memory job tracker
JOBS: Dict[str, Dict[str, Any]] = {}

# Shared processor instance (detector + OCR loaded once)
_processor: VideoProcessor | None = None


def _get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor


# ── Routes ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def homepage():
    html = WEB / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video, persist it, and kick off background processing."""
    jid = uuid.uuid4().hex[:8]
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    src = config.UPLOADS_DIR / f"{jid}{ext}"
    out = config.OUTPUTS_DIR / f"annotated_{jid}.mp4"

    with open(src, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    fname = file.filename or f"video_{jid}.mp4"
    v_url = f"/uploads/{jid}{ext}"
    video_history.add_video(
        jid=jid,
        filename=fname,
        video_url=v_url,
        status="processing",
    )

    JOBS[jid] = _new_job(jid, fname)
    threading.Thread(target=_run, args=(jid, str(src), str(out)), daemon=True).start()
    return {"job_id": jid, "status": "processing"}


@app.post("/api/create-sample")
async def create_sample():
    """Synthesise a test video and process it immediately."""
    jid = uuid.uuid4().hex[:8]
    src = config.UPLOADS_DIR / f"sample_{jid}.mp4"
    out = config.OUTPUTS_DIR / f"annotated_{jid}.mp4"
    generate_sample_video(str(src), num_seconds=6, fps=25)

    fname = "sample_traffic.mp4"
    v_url = f"/uploads/sample_{jid}.mp4"
    video_history.add_video(
        jid=jid,
        filename=fname,
        video_url=v_url,
        status="processing",
    )

    JOBS[jid] = _new_job(jid, fname)
    threading.Thread(target=_run, args=(jid, str(src), str(out)), daemon=True).start()
    return {"job_id": jid, "status": "processing"}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id in JOBS:
        return JOBS[job_id]
    # Fallback to persistent video history across reloads/refreshes
    historical = video_history.get_video(job_id)
    if historical:
        return historical
    return JSONResponse({"error": "Job not found"}, status_code=404)


@app.get("/api/videos")
async def list_history_videos():
    """Return persistent list of all uploaded videos with timestamps and counts."""
    return video_history.get_all_videos()


@app.get("/api/videos/{job_id}")
async def get_video_details(job_id: str):
    v = video_history.get_video(job_id)
    if not v:
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return v


@app.delete("/api/videos/{job_id}")
async def delete_history_video(job_id: str):
    ok = video_history.delete_video(job_id)
    return {"status": "deleted" if ok else "not_found"}


@app.get("/api/export/excel")
async def export_excel():
    if config.EXCEL_PATH.exists():
        return FileResponse(str(config.EXCEL_PATH), filename="vehicle_detections.xlsx",
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return JSONResponse({"error": "No detections yet"}, status_code=404)


@app.get("/api/plates")
async def list_plates():
    proc = _get_processor()
    return {
        "unique_count": proc.storage.total_vehicles,
        "total_vehicles": proc.storage.total_vehicles,
        "plates_detected": proc.storage.plates_detected_count,
        "missing_plates": proc.storage.missing_plates_count,
        "alert_plates": proc.storage.alert_plates_count,
        "unique_plates": proc.storage.unique_plates,
    }


@app.post("/api/clear")
async def clear_records():
    proc = _get_processor()
    proc.storage.clear()
    return {"status": "cleared"}


# ── Background worker ────────────────────────────────────────
def _new_job(jid: str, fname: str) -> dict:
    return {
        "job_id": jid, "status": "processing", "progress_percent": 0,
        "frame": 0, "total_frames": 0, "fps": 0.0,
        "unique_count": 0, "total_vehicles": 0, "plates_detected": 0,
        "missing_plates": 0, "alert_plates": 0,
        "unique_plates": [], "filename": fname,
    }


def _run(jid: str, src: str, out: str):
    def _cb(u: dict):
        if jid in JOBS:
            JOBS[jid].update(u)

    proc = _get_processor()
    try:
        res = proc.process_video(src, out, progress_callback=_cb)
        out_url = f"/outputs/{Path(out).name}"
        total_veh = res.get("total_vehicles", res.get("unique_count", 0))
        plates_det = res.get("plates_detected", 0)
        miss_plates = res.get("missing_plates", 0)
        alt_plates = res.get("alert_plates", 0)

        JOBS[jid].update({
            "status": "completed", "progress_percent": 100,
            "frame": res["total_frames"], "total_frames": res["total_frames"],
            "fps": res["average_fps"],
            "unique_count": total_veh,
            "total_vehicles": total_veh,
            "plates_detected": plates_det,
            "missing_plates": miss_plates,
            "alert_plates": alt_plates,
            "unique_plates": res["unique_plates"],
            "output_video_url": out_url,
        })
        video_history.update_video_result(
            jid=jid,
            status="completed",
            unique_count=total_veh,
            unique_plates=res["unique_plates"],
            output_video_url=out_url,
            fps=res["average_fps"],
            total_vehicles=total_veh,
            plates_detected=plates_det,
            missing_plates=miss_plates,
            alert_plates=alt_plates,
        )
    except Exception as exc:
        JOBS[jid].update({"status": "failed", "error": str(exc)})
        video_history.update_video_result(
            jid=jid,
            status="failed",
            unique_count=0,
            unique_plates=[],
            output_video_url="",
        )


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  VisionPlate AI — Number Plate Recognition Dashboard")
    print("  http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
