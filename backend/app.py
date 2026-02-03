# backend/app.py
# -*- coding: utf-8 -*-
import os
import json, time, math
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv

# ====== URL Prefix Configuration ======
URL_PREFIX = os.getenv("URL_PREFIX", "/watermark")

# ====== Path setup ======
BASE_DIR = Path(__file__).resolve().parent            # .../backend
ROOT_DIR = BASE_DIR.parent                            # repo root
FRONTEND_DIR = ROOT_DIR / "frontend"                  # .../frontend
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "reports.jsonl"

# กันกรณี build/boot แรกที่ frontend ยังไม่มีไฟล์ -> อย่าทำให้แอปล่ม
if not (FRONTEND_DIR / "index.html").exists():
    FRONTEND_DIR = ROOT_DIR

# ====== Env (.env ได้ทั้งที่ root และ backend) ======
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

# ====== JSON-safe helper ======
def _sanitize_for_json(obj):
    """แทน NaN/Inf ด้วย None เพื่อให้ json.dumps ทำงานได้"""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj

# ====== Calibration: y' = 0.9553*x + 0.0325 ======
def calibrate(level_m: float, ndigits: int = 2) -> float:
    if level_m is None or (isinstance(level_m, float) and math.isnan(level_m)):
        return level_m
    y = 0.9553 * float(level_m) + 0.0325
    return round(y, ndigits)

# ====== Inference loader ======
predict_height_m = None
predict_height_debug = None
infer_status_fn = None

try:
    # uvicorn backend.app:app
    from .infer_service import predict_height_m as _phm
    predict_height_m = _phm
    try:
        from .infer_service import predict_height_debug as _phd
        predict_height_debug = _phd
    except Exception:
        predict_height_debug = None
    try:
        from .infer_service import infer_status as _istat
        infer_status_fn = _istat
    except Exception:
        infer_status_fn = None
except Exception:
    try:
        # python backend/app.py
        from infer_service import predict_height_m as _phm
        predict_height_m = _phm
        try:
            from infer_service import predict_height_debug as _phd
            predict_height_debug = _phd
        except Exception:
            predict_height_debug = None
        try:
            from infer_service import infer_status as _istat
            infer_status_fn = _istat
        except Exception:
            infer_status_fn = None
    except Exception:
        def _not_ready(*_, **__):
            raise RuntimeError("ยังโหลดโมเดลไม่สำเร็จ (โปรดวางไฟล์ weights และปรับ ENV ให้ครบ)")
        predict_height_m = _not_ready
        predict_height_debug = None
        infer_status_fn = None

# ====== Sub-App for /watermark prefix ======
watermark_app = FastAPI(title="Flood Mark API")

watermark_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟไฟล์อัปโหลด (ไม่ชน /api/*)
watermark_app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ====== Helpers ======
def _save_record(rec: dict):
    rec = _sanitize_for_json(rec)
    with DB_FILE.open("a", encoding="utf-8") as f:
        # บังคับห้าม NaN/Inf ในไฟล์ถาวร
        f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")

def _load_records(limit: int = 200) -> List[dict]:
    if not DB_FILE.exists():
        return []
    items = []
    with DB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                pass
    return items[-limit:]

# ====== API ======
@watermark_app.get("/api/health")
def health():
    # infer_ready = true ถ้ามีฟังก์ชัน predict_height_m และไม่ใช่ placeholder
    infer_ready = callable(predict_height_m) and predict_height_m.__name__ != "_not_ready"
    # ถ้ามี infer_status_fn ให้แนบสถานะละเอียด ๆ ไปด้วย
    status = infer_status_fn() if callable(infer_status_fn) else {"model_ready": infer_ready}
    return {"ok": True, "ts": int(time.time()), "infer_ready": infer_ready, "status": status}

@watermark_app.get("/api/infer_status")
def api_infer_status():
    if not callable(infer_status_fn):
        # อย่างน้อยบอกว่า import ไม่ได้
        return {"ok": False, "error": "infer_service not ready or infer_status() missing"}
    try:
        data = infer_status_fn()
        return data | {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}

@watermark_app.get("/api/reports")
def get_reports(limit: int = 200):
    items = _load_records(limit=limit)
    items = [_sanitize_for_json(x) for x in items]  # กัน record เก่า ๆ ที่มี NaN
    return {"ok": True, "items": items}

@watermark_app.post("/api/report")
async def create_report(
    image: UploadFile = File(...),
    lat: str = Form(...), lng: str = Form(...),
    date_iso: str = Form(...),
    object_type: str = Form(...),
    description: str = Form(""),
    address: str = Form(""),
    skip_infer: int = 0
):
    # 1) save file
    safe_name = f"{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    # 2) infer / mock
    if int(skip_infer or 0) == 1:
        water_level_m = 0.25
    else:
        try:
            water_level_m = predict_height_m(str(out_path))
        except Exception:
            water_level_m = float("nan")

    # 3) calibrate
    water_level_m = calibrate(water_level_m)

    rec = {
        "lat": float(lat), "lng": float(lng),
        "date_iso": date_iso,
        "object_type": object_type,
        "description": description,
        "address": address if address else f"{lat},{lng}",
        "photo_url": f"/uploads/{safe_name}",
        "water_level_m": water_level_m
    }
    rec = _sanitize_for_json(rec)  # << สำคัญ
    _save_record(rec)
    return JSONResponse({"ok": True, **rec})

# -------- Debug infer endpoint --------
@watermark_app.post("/api/debug_infer")
async def debug_infer(image: UploadFile = File(...)):
    """
    เซฟรูปแล้วรัน predict_height_debug (ถ้ามี) คืนรายละเอียดดีบั๊กทั้งหมด
    """
    safe_name = f"_debug_{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    if not callable(predict_height_debug):
        return JSONResponse(
            {"ok": False, "photo_url": f"/uploads/{safe_name}", "error": "infer_service not ready"},
            status_code=200
        )

    try:
        dbg = predict_height_debug(str(out_path))
        return JSONResponse(
            {"ok": bool(dbg.get("ok")), "photo_url": f"/uploads/{safe_name}", "debug": dbg},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "photo_url": f"/uploads/{safe_name}", "error": f"{e.__class__.__name__}: {e}"},
            status_code=200
        )

# ====== Static frontend ======
watermark_app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ====== Main App - Mounts sub-app at /watermark ======
app = FastAPI(title="Water Mark Measure")

# Redirect root to /watermark
@app.get("/")
def redirect_to_watermark():
    return RedirectResponse(url=URL_PREFIX)

# Mount the watermark sub-application
app.mount(URL_PREFIX, watermark_app)

# For local run: python backend/app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=False)
