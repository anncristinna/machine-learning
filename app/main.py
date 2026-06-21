"""
API de avaliação de modelos - Breast Cancer Wisconsin Diagnostic

Endpoints:
  GET  /api/models             -> lista os modelos disponíveis
  GET  /api/metrics            -> métricas e matriz de confusão de todos os modelos
  GET  /api/patients           -> lista paginada dos pacientes do conjunto de teste
  GET  /api/patients/{id}      -> detalhe de um paciente (todas as features + previsões)
  POST /api/predict            -> roda os modelos em cima de um conjunto de features customizado
  GET  /api/analysis/correlation   -> matriz de correlação das features _mean (treino)
  GET  /api/analysis/distributions -> curvas KDE por feature/classe (treino)
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(
    title="Breast Cancer Diagnostic API",
    description="API para servir previsões e métricas dos modelos treinados no notebook "
                 "(Regressão Logística, LDA, QDA, Naive Bayes Gaussiano).",
    version="1.0.0",
)

# Libera acesso do front-end (ajuste para o domínio real em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Carrega artefatos uma única vez, na subida da API
# ---------------------------------------------------------------------------
try:
    with open(DATA_DIR / "metrics.json", encoding="utf-8") as f:
        METRICS = json.load(f)

    with open(DATA_DIR / "patients.json", encoding="utf-8") as f:
        PATIENTS = json.load(f)

    with open(DATA_DIR / "feature_columns.json", encoding="utf-8") as f:
        FEATURE_COLUMNS = json.load(f)

    with open(DATA_DIR / "correlation.json", encoding="utf-8") as f:
        CORRELATION = json.load(f)

    with open(DATA_DIR / "distributions.json", encoding="utf-8") as f:
        DISTRIBUTIONS = json.load(f)

    SCALER = joblib.load(DATA_DIR / "scaler.pkl")
    MODELS = {
        key: joblib.load(DATA_DIR / f"model_{key}.pkl")
        for key in METRICS["models"].keys()
    }
except FileNotFoundError as e:
    raise SystemExit(
        "\n[ERRO] Artefatos de dados/modelo não encontrados em data/.\n"
        "Rode primeiro: python3 train_and_export.py (com o data.csv na raiz do projeto)\n"
        f"Detalhe: {e}\n"
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CustomFeatures(BaseModel):
    # Aceita um dicionário livre {nome_da_feature: valor}; validado contra
    # FEATURE_COLUMNS no momento da previsão.
    features: dict[str, float] = Field(..., description="Mapa feature -> valor")
    model: str = Field("lr", description="Chave do modelo: lr | lda | qda | nb")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/models")
def list_models():
    return [
        {"key": key, "label": m["label"]}
        for key, m in METRICS["models"].items()
    ]


@app.get("/api/metrics")
def get_metrics():
    return METRICS


@app.get("/api/patients")
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=1000),
    actual: Optional[str] = Query(None, description="Filtra por 'benigno' ou 'maligno'"),
):
    items = PATIENTS
    if actual:
        items = [p for p in items if p["actual"] == actual]

    start = (page - 1) * page_size
    end = start + page_size
    return {
        "total": len(items),
        "page": page,
        "page_size": page_size,
        "results": items[start:end],
    }


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
        raise HTTPException(
            status_code=422,
            detail=f"Faltam {len(missing)} features. Exemplo faltante: {missing[:5]}",
        )

    df = pd.DataFrame([{c: payload.features[c] for c in FEATURE_COLUMNS}])
    scaled = SCALER.transform(df)
    model = MODELS[payload.model]
    pred = int(model.predict(scaled)[0])
    proba = float(model.predict_proba(scaled)[0][1])

    return {
        "model": payload.model,
        "label": "maligno" if pred == 1 else "benigno",
        "probability": round(proba, 4),
    }


@app.get("/api/analysis/correlation")
def get_correlation():
    return CORRELATION


@app.get("/api/analysis/distributions")
def get_distributions():
    return DISTRIBUTIONS


from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
