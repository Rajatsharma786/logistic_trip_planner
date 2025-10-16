from __future__ import annotations

import os
import json
import joblib #type: ignore
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pytz
from fastapi import FastAPI, HTTPException #type: ignore
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.tools import (
    END_POINTS,
    WAYPOINTS,
    get_weather_data,
    get_traffic_data,
    get_travel_time_osrm,  
)

from src.rag_service import (
        make_default_rag_service,
        RAGService,
        RAGConfig,
        RAGQueryRequest,
        RAGQueryResponse,
    ) 

MEL_TZ = pytz.timezone("Australia/Melbourne")

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "src/model_folder_v1")
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "pipeline.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_xgb.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "safe_features.json")

if not os.path.exists(ARTIFACT_DIR):
    raise RuntimeError(f"Artifact directory not found: {ARTIFACT_DIR}")

# Load artifacts (prefer full pipeline if present)
pipeline = preprocessor = model = None
if os.path.exists(PIPELINE_PATH):
    pipeline = joblib.load(PIPELINE_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["regressor"]
else:
    if not (os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH)):
        raise RuntimeError("Missing preprocessor/model artifacts. Expected preprocessor.pkl and model_xgb.pkl")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

if not os.path.exists(FEATURES_PATH):
    raise RuntimeError(f"Missing features file: {FEATURES_PATH}")

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    _feat_payload = json.load(f)
SAFE_FEATURES: List[str] = _feat_payload.get("features", [])
if not SAFE_FEATURES:
    raise RuntimeError("SAFE_FEATURES list is empty; cannot build inference frame")

class MinimalPredictRequest(BaseModel):
    route: str = Field(..., description="Route identifier, e.g., Hume_Highway or Inland_Route")
    ts_utc: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp for the prediction context; defaults to now"
    )

class PredictResponse(BaseModel):
    route: str
    ts_utc: datetime
    ts_local: str
    features_used: Dict[str, Any]
    prediction_seconds: int
    prediction_minutes: float
    prediction_hours: float 
    sources: Dict[str, Any]

app = FastAPI(title="ETA Prediction Service", version="1.0.0")

@app.get("/")
def root():
    """Root endpoint - shows available endpoints"""
    return {
        "message": "ETA Prediction Service API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict_travel_time",
            "rag_query": "/rag/query",
            "agent": "/agent/compose",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "running"
    }

def _localize(ts_utc: datetime) -> datetime:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    return ts_utc.astimezone(MEL_TZ)

def _derive_time_parts(ts_utc: datetime) -> Dict[str, int]:
    local = _localize(ts_utc)
    return {"day_of_week": local.weekday(), "hour_of_day": local.hour}

def _to_float(x, default: float = 0.0) -> float:
    try:
        if x in (None, "", "-", "NaN"):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _build_route_context(route: str) -> Dict[str, Any]:
    if route not in WAYPOINTS:
        raise HTTPException(status_code=400, detail=f"Unknown route: {route}")

    origin_cfg = END_POINTS["melbourne"]
    dest_cfg   = END_POINTS["sydney"]
    wp_cfgs    = list(WAYPOINTS[route].values())

    waypoints = [(wp["lat"], wp["lon"]) for wp in wp_cfgs]
    all_lats  = [origin_cfg["lat"], dest_cfg["lat"]] + [lat for lat, _ in waypoints]
    all_lons  = [origin_cfg["lon"], dest_cfg["lon"]] + [lon for _, lon in waypoints]
    bbox      = f"{min(all_lons)},{min(all_lats)},{max(all_lons)},{max(all_lats)}"

    context = {
        "route": route,
        "origin": (origin_cfg["lat"], origin_cfg["lon"]),
        "destination": (dest_cfg["lat"], dest_cfg["lon"]),
        "waypoints": waypoints,
        "bbox": bbox,
        "bom_station_id": origin_cfg.get("weather_station_id")
    }
    return context

def build_features(ts_utc: datetime,
                   route_name: str,
                   bom: Dict[str, Any],
                   tomtom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map live tool outputs -> the exact SAFE_FEATURES schema used at training time.
    Adds engineered features so serving matches training (weekend, time-of-day segment,
    feels_like_temp, and simple interaction terms). Any column not present in SAFE_FEATURES
    is safely ignored at inference time.
    """
    # ---- time parts ----
    t = _derive_time_parts(ts_utc)
    day_of_week = int(t["day_of_week"])         
    hour_of_day = int(t["hour_of_day"])

    # ---- weather ----
    wx_temp_c   = _to_float(bom.get("temperature"), np.nan)
    wx_rain_mm  = _to_float(bom.get("rainfall_since_9am"), 0.0)
    wx_wind_kmh = _to_float(bom.get("wind_speed_kmh"), 0.0)
    wx_location = bom.get("location") or "unknown"

    # ---- traffic (from TomTom) ----
    incidents = tomtom.get("incidents", []) if isinstance(tomtom, dict) else []
    leg_incident_count = len(incidents)

    leg_incident_delay_sum_sec = 0.0
    for inc in incidents:
        d = inc.get("incident_delay")
        if d is None:
            d = inc.get("incident_delay_sec")
        leg_incident_delay_sum_sec += _to_float(d, 0.0)

    # ---- flags / segments you asked for ----
    is_weekend = 1 if day_of_week in (5, 6) else 0  # Sat/Sun

    # 3-way time-of-day segment (string; OneHot by your preprocessor)
    if 6 <= hour_of_day <= 9:
        time_of_day_segment = "morning"
    elif 16 <= hour_of_day <= 19:
        time_of_day_segment = "pm_peak"
    else:
        time_of_day_segment = "offpeak"

    if pd.isna(wx_temp_c):
        feels_like_temp = np.nan
    else:
        feels_like_temp = wx_temp_c - 0.10 * wx_wind_kmh - 0.20 * wx_rain_mm


    rain_flag = 1 if wx_rain_mm > 0 else 0
    extreme_weather_flag = 1 if (wx_wind_kmh >= 60 or (not pd.isna(wx_temp_c) and (wx_temp_c <= 2 or wx_temp_c >= 42))) else 0

    rain_x_delay  = wx_rain_mm  * leg_incident_delay_sum_sec
    wind_x_delay  = wx_wind_kmh * leg_incident_delay_sum_sec
    temp_x_delay  = (wx_temp_c if not pd.isna(wx_temp_c) else 0.0) * leg_incident_delay_sum_sec

    is_peak = 1 if time_of_day_segment in ("morning", "pm_peak") else 0
    peak_x_delay     = is_peak    * leg_incident_delay_sum_sec
    weekend_x_delay  = is_weekend * leg_incident_delay_sum_sec
    rain_x_peak      = rain_flag  * is_peak
    wind_x_peak      = wx_wind_kmh * is_peak

    raw_feat = {

        "route": route_name,
        "wx_location": wx_location,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "wx_temp_c": wx_temp_c,
        "wx_rain_mm": wx_rain_mm,
        "wx_wind_kmh": wx_wind_kmh,
        "leg_incident_count": leg_incident_count,
        "leg_incident_delay_sum_sec": leg_incident_delay_sum_sec,
        "rain_flag": rain_flag,
        "extreme_weather_flag": extreme_weather_flag,

        "is_weekend": is_weekend,
        "time_of_day_segment": time_of_day_segment,  
        "feels_like_temp": feels_like_temp,
        "rain_x_delay":  rain_x_delay,
        "wind_x_delay":  wind_x_delay,
        "temp_x_delay":  temp_x_delay,
        "peak_x_delay":  peak_x_delay,
        "weekend_x_delay": weekend_x_delay,
        "rain_x_peak":   rain_x_peak,
        "wind_x_peak":   wind_x_peak,
    }

    # ---- final dict aligned to SAFE_FEATURES ----
    clean: Dict[str, Any] = {}
    for col in SAFE_FEATURES:
        v = raw_feat.get(col, np.nan)
        # keep strings (categoricals) as-is; coerce numerics
        clean[col] = v if isinstance(v, str) else _to_float(v, np.nan)
    return clean

def _dtype_enforce(X: pd.DataFrame) -> pd.DataFrame:
    """Force dtypes to match the fitted ColumnTransformer (prevents isnan errors)."""
    cat_cols, num_cols = [], []
    for name, trans, cols in getattr(preprocessor, "transformers_", []):
        if name == "remainder":
            continue
        cols = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else [cols]
        trans_str = str(trans)
        is_cat = ("OneHotEncoder" in trans_str) or hasattr(trans, "categories_")
        if not is_cat and hasattr(trans, "named_steps"):
            is_cat = any(("OneHotEncoder" in str(s[1])) for s in trans.named_steps.items())
        (cat_cols if is_cat else num_cols).extend(cols)

    if not num_cols:
        num_cols = list(set(SAFE_FEATURES) - set(cat_cols))

    X.replace({"NaN": np.nan, "nan": np.nan, "": np.nan, "-": np.nan}, inplace=True)

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("object")

    for c in num_cols:
        if c in X.columns:
            if X[c].dtype == "O":
                X[c] = pd.to_numeric(
                    X[c].astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
                    errors="coerce"
                )
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].astype("float64")
    return X

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "features_count": len(SAFE_FEATURES),
        "artifact_dir": os.path.abspath(ARTIFACT_DIR),
    }

@app.post("/predict_travel_time", response_model=PredictResponse)
def predict_travel_time(req: MinimalPredictRequest):
    """
    Predict travel time given just a route (e.g., 'Hume_Highway').
    Origin/destination/waypoints/bbox/station are auto-derived from constants.
    """
    # 1) Build route context
    ctx = _build_route_context(req.route)

    # 2) Live sources
    bom = get_weather_data(ctx["bom_station_id"]) or {}
    tomtom = get_traffic_data(ctx["bbox"]) or {}

    bom_ctx = (
        {"error": bom["error"]}
        if isinstance(bom, dict) and "error" in bom else
        {
            "location": bom.get("location"),
            "time_utc": bom.get("time_utc"),
            "temperature": bom.get("temperature"),
            "rainfall_since_9am": bom.get("rainfall_since_9am"),
            "wind_speed_kmh": bom.get("wind_speed_kmh"),
            "wind_direction": bom.get("wind_direction"),
        }
    )
    tomtom_ctx = (
        {"error": tomtom["error"], "bbox": ctx["bbox"], "count": 0}
        if isinstance(tomtom, dict) and "error" in tomtom else
        {"bbox": ctx["bbox"], "count": len(tomtom.get("incidents", [])) if isinstance(tomtom, dict) else 0}
    )

    # 3) Features
    feat = build_features(req.ts_utc, ctx["route"], bom, tomtom)
    X = pd.DataFrame([feat], columns=SAFE_FEATURES)
    X = _dtype_enforce(X)

    # 4) Predict
    try:
        X_enc = preprocessor.transform(X)
        y_pred = model.predict(X_enc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    y_sec = int(round(float(y_pred[0])))
    y_min = round(y_sec / 60.0, 2)
    y_hours = round(y_sec / 3600.0, 2) 

    return PredictResponse(
        route=ctx["route"],
        ts_utc=req.ts_utc,
        ts_local=_localize(req.ts_utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        features_used=feat,
        prediction_seconds=y_sec,
        prediction_minutes=y_min,
        prediction_hours=y_hours,
        sources={"bom": bom_ctx, "tomtom_summary": tomtom_ctx},
    )

# ---- RAG integration ----

_rag: RAGService | None = None

def _get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = make_default_rag_service()
    return _rag

@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    try:
        res = _get_rag().query(req.question, k=req.k)
        return RAGQueryResponse(**res)  # type: ignore[arg-type]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ---- Composed agent endpoint ----
class ComposeRequest(BaseModel):
    question: str
    k: int = 5
    route: Optional[str] = Field(default=None, description="If provided, will compute ETA with the ML model")
    ts_utc: Optional[datetime] = Field(default=None)

class ComposeResponse(BaseModel):
    answer: str
    used_k: int
    included: Dict[str, Any]

@app.post("/agent/compose", response_model=ComposeResponse)
def agent_compose(req: ComposeRequest):
    try:
        # 1) Retrieve RAG context
        rag = _get_rag()
        retrieved = rag.retrieve_context(req.question, k=req.k)
        context_text = retrieved["context"]

        # 2) Optionally compute ETA and live summaries
        eta_block = None
        weather_block = None
        traffic_block = None
        if req.route is not None:
            ts = req.ts_utc or datetime.now(timezone.utc)
            ctx = _build_route_context(req.route)

            # live sources
            bom = get_weather_data(ctx["bom_station_id"]) or {}
            tomtom = get_traffic_data(ctx["bbox"]) or {}

            # features + prediction
            feat = build_features(ts, ctx["route"], bom, tomtom)
            X = pd.DataFrame([feat], columns=SAFE_FEATURES)
            X = _dtype_enforce(X)
            X_enc = preprocessor.transform(X)
            y_pred = model.predict(X_enc)
            y_sec = int(round(float(y_pred[0])))

            eta_block = {
                "route": ctx["route"],
                "prediction_seconds": y_sec,
                "prediction_minutes": round(y_sec / 60.0, 2),
                "prediction_hours": round(y_sec / 3600.0, 2),
                "ts_local": _localize(ts).strftime("%Y-%m-%d %H:%M:%S %Z"),
            }

            # compact summaries for the model
            weather_block = {
                "location": bom.get("location") if isinstance(bom, dict) else None,
                "temperature": (bom.get("temperature") if isinstance(bom, dict) else None),
                "rainfall_since_9am": (bom.get("rainfall_since_9am") if isinstance(bom, dict) else None),
                "wind_speed_kmh": (bom.get("wind_speed_kmh") if isinstance(bom, dict) else None),
            }
            traffic_block = {
                "bbox": ctx["bbox"],
                "count": (len(tomtom.get("incidents", [])) if isinstance(tomtom, dict) else 0),
            }

        chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_retries=8, timeout=60)
        mix_prompt = ChatPromptTemplate.from_template(
            "You are a logistics assistant.\n"
            "You receive a user question, retrieved context, and optionally an ETA prediction plus brief weather/traffic.\n"
            "Use whatever is most relevant. If ETA is provided, you may integrate it into the answer; otherwise rely on context.\n"
            "Do not hallucinate beyond the provided information.\n\n"
            "Question:\n{q}\n\nRetrieved context (text):\n{ctx}\n\nETA (optional):\n{eta}\n\nWeather (optional):\n{wx}\n\nTraffic summary (optional):\n{tt}\n\nAnswer:"
        )

        rendered = mix_prompt | chat
        res = rendered.invoke({
            "q": req.question,
            "ctx": context_text,
            "eta": json.dumps(eta_block, ensure_ascii=False) if eta_block else "None",
            "wx": json.dumps(weather_block, ensure_ascii=False) if weather_block else "None",
            "tt": json.dumps(traffic_block, ensure_ascii=False) if traffic_block else "None",
        })

        answer_text = getattr(res, "content", str(res))
        return ComposeResponse(answer=answer_text, used_k=retrieved["used_k"], included={
            "eta": eta_block,
            "weather": weather_block,
            "traffic": traffic_block,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compose failed: {e}")