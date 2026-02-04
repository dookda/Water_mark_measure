# backend/infer_service.py
# -*- coding: utf-8 -*-
"""
ตัวห่อโมเดล:
- โหลด YOLO + Fusion model ครั้งเดียว แล้ว reuse
- รับ path รูป -> คืนค่าความสูงรอยน้ำ (เมตร, ปัด 2 ตำแหน่ง)
- ถ้า ENV เป็น URL จะดาวน์โหลดไฟล์มาเก็บใน backend/weights อัตโนมัติ

หมายเหตุสำคัญ:
- ไฟล์นี้เป็น "service" ล้วน ๆ ห้ามมี FastAPI endpoint (@app.get/...)
- endpoint /api/infer_status อยู่ใน backend/app.py แล้ว และจะ import infer_status() จากไฟล์นี้ไปใช้
"""
from __future__ import annotations

from pathlib import Path
import os, json, re
import urllib.parse
import requests
import pandas as pd
import torch
from ultralytics import YOLO

from .infer_res34_adjusted import (
    WATER_CLASS, REF_CLASSES,
    combine_water_masks, clean_mask, waterline_y_curve, ref_sizes_from_result,
    FusionRegressor, load_checkpoint, add_engineered_features, apply_scaler_df,
    make_feature_matrix, load_image_tensor, r2
)

# ====== Path/Env ======
BASE_DIR    = Path(__file__).resolve().parent   # .../backend
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _env(name: str, default: str | None = None) -> str | None:
    """อ่าน ENV แล้วตัดช่องว่าง/CRLF ท้ายสตริงออก"""
    v = os.environ.get(name, default)
    if v is None:
        return None
    return re.sub(r"[\r\n]+$", "", str(v).strip())

# ENV ที่รับได้ (ทั้งพาธไฟล์และ URL)
YOLO_URL     = _env("YOLO_URL")  # optional: ให้ URL แยกของ yolo ได้
WEIGHTS_YOLO = _env("WEIGHTS_YOLO", str(WEIGHTS_DIR / "best.pt"))

# CKPT_FUSION มาก่อน; ถ้าไม่มีก็ยอมรับ MODEL_URL (เดิม)
CKPT_FUSION  = _env("CKPT_FUSION") or _env("MODEL_URL") or str(WEIGHTS_DIR / "fusion_resnet34_best.pt")

SCALER_URL   = _env("SCALER_URL")  # optional: URL ของ scaler
SCALER_JSON  = _env("SCALER_JSON", str(WEIGHTS_DIR / "scaler.json"))

IMG_SIZE_YOLO = int(_env("IMG_SIZE_YOLO", "960"))
CONF_YOLO     = float(_env("CONF_YOLO", "0.35"))
IOU_YOLO      = float(_env("IOU_YOLO", "0.50"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Lazy singletons ======
_yolo            = None
_fusion_model    = None
_feat_names      = None
_img_size_fusion = 384
_scaler          = None

# เก็บ error ล่าสุด (เอาไว้ให้ infer_status() รายงาน)
_last_init_error: str | None = None

def _is_url(s: str) -> bool:
    try:
        u = urllib.parse.urlparse(str(s))
        return u.scheme in ("http", "https")
    except Exception:
        return False

def _download_to(url: str, dst: Path):
    """ดาวน์โหลดไฟล์แบบ binary (follow redirects)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)

def _ensure_local_file(maybe_url_or_path: str, fallback_name: str) -> Path:
    """
    รับค่าที่อาจเป็น URL หรือพาธไฟล์
    - ถ้าเป็น URL: ดาวน์โหลดเป็นชื่อ fallback_name ใน WEIGHTS_DIR แล้วคืนพาธไฟล์ที่โหลดได้
    - ถ้าเป็นไฟล์: คืนพาธเดิม (ถ้ามีอยู่)
    """
    s = str(maybe_url_or_path)
    if _is_url(s):
        dst = WEIGHTS_DIR / fallback_name
        if not dst.exists():
            _download_to(s, dst)
        return dst
    else:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p

def _lazy_init():
    """โหลดโมเดล/สเกลเลอร์ครั้งเดียว (และดาวน์โหลดไฟล์หากต้อง)"""
    global _yolo, _fusion_model, _feat_names, _img_size_fusion, _scaler
    global WEIGHTS_YOLO, CKPT_FUSION, SCALER_JSON, _last_init_error

    try:
        # ---- YOLO weights (รับทั้ง URL/Path) ----
        if _yolo is None:
            # ถ้าแยก YOLO_URL มา และปลายทาง WEIGHTS_YOLO ยังไม่มี -> ดาวน์โหลดไปปลายทางนั้น
            if YOLO_URL and (not Path(WEIGHTS_YOLO).exists()):
                _download_to(YOLO_URL, Path(WEIGHTS_YOLO))

            yolo_path = _ensure_local_file(WEIGHTS_YOLO, "best.pt")
            WEIGHTS_YOLO = str(yolo_path)

            _yolo = YOLO(WEIGHTS_YOLO)
            _yolo.to(str(DEVICE))

        # ---- Fusion checkpoint (รับทั้ง URL/Path) ----
        if _fusion_model is None:
            ckpt_path = _ensure_local_file(CKPT_FUSION, "fusion_resnet34_best.pt")
            CKPT_FUSION = str(ckpt_path)

            _fusion_model, _feat_names, _img_size_fusion = load_checkpoint(
                CKPT_FUSION, default_img_size=384
            )
            _fusion_model.to(DEVICE).eval()

        # ---- Scaler (json; รับทั้ง URL/Path) ----
        if _scaler is None:
            # ถ้ามี SCALER_URL และไฟล์ปลายทางยังไม่มี -> ดาวน์โหลดก่อน
            if SCALER_URL and (not Path(SCALER_JSON).exists()):
                _download_to(SCALER_URL, Path(SCALER_JSON))

            p = Path(SCALER_JSON)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    _scaler = json.load(f)
            else:
                _scaler = None  # ไม่มีสเกลเลอร์ก็รันได้ (แต่ฟีเจอร์บางส่วนอาจไม่ถูก scale)

        _last_init_error = None

    except Exception as e:
        _last_init_error = f"{e.__class__.__name__}: {e}"
        raise

def infer_status() -> dict:
    """
    ฟังก์ชันสถานะ (ธรรมดา ไม่ใช่ endpoint):
    app.py จะ import ไป expose เป็น /api/infer_status เอง
    """
    info = {
        "model_ready": False,
        "device": DEVICE.type,
        "weights_yolo": str(WEIGHTS_YOLO),
        "ckpt_fusion": str(CKPT_FUSION),
        "scaler_json": str(SCALER_JSON),
        "has_scaler": False,
        "error": _last_init_error,
    }
    try:
        _lazy_init()
        info.update({
            "model_ready": True,
            "weights_yolo": str(WEIGHTS_YOLO),
            "ckpt_fusion": str(CKPT_FUSION),
            "scaler_json": str(SCALER_JSON),
            "has_scaler": bool(_scaler is not None),
            "error": None,
        })
        return info
    except Exception as e:
        info["error"] = f"{e.__class__.__name__}: {e}"
        return info

def predict_height_m(image_path: str) -> float:
    """
    รับ path รูป -> คืนความสูงรอยน้ำหน่วย 'เมตร' (ปัด 2 ตำแหน่ง)
    - ถ้าตรวจไม่เจอ waterline หรือเกิด error จะคืน NaN
    """
    try:
        _lazy_init()

        r = _yolo.predict(
            source=image_path,
            imgsz=IMG_SIZE_YOLO,
            conf=CONF_YOLO,
            iou=IOU_YOLO,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]

        H, W = r.orig_img.shape[:2]

        wm = combine_water_masks(r, WATER_CLASS)
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)

        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        if y_curve is None:
            return float("nan")

        pixel_height = (H - 1) - int(y_med)
        ref_dict = ref_sizes_from_result(r)

        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)
        im = load_image_tensor(image_path, _img_size_fusion).to(DEVICE)

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            y = _fusion_model(im, X_t).detach().cpu().item()

        return float(r2([y], 2)[0])

    except Exception as e:
        # อย่ากลืนเงียบ ๆ อย่างน้อย log ไว้ให้ดูใน server logs
        print("predict_height_m error:", f"{e.__class__.__name__}: {e}")
        return float("nan")

def predict_height_debug(image_path: str) -> dict:
    """เวอร์ชันดีบั๊ก: คืนรายละเอียดทุกขั้นตอน + trace เมื่อพัง"""
    dbg = {"ok": False}
    try:
        _lazy_init()
        dbg.update({
            "device": DEVICE.type,
            "weights_yolo": str(WEIGHTS_YOLO),
            "ckpt_fusion": str(CKPT_FUSION),
            "scaler_json": str(SCALER_JSON),
            "has_scaler": bool(_scaler is not None),
        })

        r = _yolo.predict(
            source=image_path,
            imgsz=IMG_SIZE_YOLO,
            conf=CONF_YOLO,
            iou=IOU_YOLO,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]

        H, W = r.orig_img.shape[:2]
        dbg["H"], dbg["W"] = int(H), int(W)

        # boxes summary
        boxes = []
        names = getattr(r, "names", {})
        if getattr(r, "boxes", None) is not None:
            for b in r.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item()) if getattr(b, "conf", None) is not None else None
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append({
                    "cls": cls,
                    "name": names.get(cls, str(cls)),
                    "conf": round(conf or 0, 3),
                    "xyxy": [x1, y1, x2, y2]
                })
        dbg["detections"] = boxes

        wm = combine_water_masks(r, WATER_CLASS)
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)
        dbg["water_pixels"] = int(wm.astype("uint8").sum())
        dbg["water_pixels_clean"] = int(wm_clean.astype("uint8").sum())

        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        if y_curve is None:
            dbg["reason"] = "no_waterline_detected"
            return dbg

        pixel_height = (H - 1) - int(y_med)
        dbg["y_median"] = int(y_med)
        dbg["pixel_height"] = int(pixel_height)

        ref_dict = ref_sizes_from_result(r)
        dbg["ref_features"] = ref_dict

        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)
        dbg["used_cols"] = list(used_cols)

        im = load_image_tensor(image_path, _img_size_fusion).to(DEVICE)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            y = _fusion_model(im, X_t).detach().cpu().item()

        dbg["y_raw"] = float(y)
        dbg["y_m_rounded"] = float(r2([y], 2)[0])
        dbg["ok"] = True
        return dbg

    except Exception as e:
        import traceback
        dbg["ok"] = False
        dbg["reason"] = f"{e.__class__.__name__}: {e}"
        dbg["trace"] = traceback.format_exc()
        return dbg
