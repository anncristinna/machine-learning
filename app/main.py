import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Breast Cancer Diagnostic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

try:
    with open(DATA_DIR / "metrics.json",         encoding="utf-8") as f: METRICS        = json.load(f)
    with open(DATA_DIR / "metrics_tuned.json",   encoding="utf-8") as f: METRICS_TUNED  = json.load(f)
    with open(DATA_DIR / "patients.json",        encoding="utf-8") as f: PATIENTS       = json.load(f)
    with open(DATA_DIR / "feature_columns.json", encoding="utf-8") as f: FEATURE_COLUMNS= json.load(f)
    with open(DATA_DIR / "correlation.json",     encoding="utf-8") as f: CORRELATION    = json.load(f)
    with open(DATA_DIR / "distributions.json",   encoding="utf-8") as f: DISTRIBUTIONS  = json.load(f)
    with open(DATA_DIR / "boxplots.json",        encoding="utf-8") as f: BOXPLOTS       = json.load(f)
    with open(DATA_DIR / "pca_boundary.json",    encoding="utf-8") as f: PCA_BOUNDARY   = json.load(f)
    SCALER = joblib.load(DATA_DIR / "scaler.pkl")
    MODELS = {key: joblib.load(DATA_DIR / f"model_{key}.pkl") for key in METRICS["models"].keys()}
except FileNotFoundError as e:
    raise SystemExit(f"\n[ERRO] Artefatos não encontrados em data/.\nRode: python train_and_export.py\nDetalhe: {e}\n")


class CustomFeatures(BaseModel):
    features: dict[str, float] = Field(..., description="Mapa feature -> valor")
    model: str = Field("lr", description="lr | lda | qda | nb")


@app.get("/api/models")
def list_models():
    return [{"key": k, "label": m["label"]} for k, m in METRICS["models"].items()]

@app.get("/api/metrics")
def get_metrics():
    return METRICS

@app.get("/api/metrics/tuned")
def get_metrics_tuned():
    return METRICS_TUNED

@app.get("/api/patients")
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=1000),
    actual: Optional[str] = Query(None),
):
    items = PATIENTS
    if actual:
        items = [p for p in items if p["actual"] == actual]
    start = (page - 1) * page_size
    return {"total": len(items), "page": page, "page_size": page_size, "results": items[start:start + page_size]}

@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int):
    patient = next((p for p in PATIENTS if p["id"] == patient_id), None)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente não encontrado")
    return patient

@app.post("/api/predict")
def predict_custom(payload: CustomFeatures):
    if payload.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{payload.model}' inválido")
    missing = [c for c in FEATURE_COLUMNS if c not in payload.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltam {len(missing)} features")
    df     = pd.DataFrame([{c: payload.features[c] for c in FEATURE_COLUMNS}])
    scaled = SCALER.transform(df)
    model  = MODELS[payload.model]
    pred   = int(model.predict(scaled)[0])
    proba  = float(model.predict_proba(scaled)[0][1])
    return {"model": payload.model, "label": "maligno" if pred == 1 else "benigno", "probability": round(proba, 4)}

@app.get("/api/analysis/correlation")
def get_correlation(): return CORRELATION

@app.get("/api/analysis/distributions")
def get_distributions(): return DISTRIBUTIONS

@app.get("/api/analysis/boxplots")
def get_boxplots(): return BOXPLOTS

@app.get("/api/analysis/pca_boundary")
def get_pca_boundary(): return PCA_BOUNDARY

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")