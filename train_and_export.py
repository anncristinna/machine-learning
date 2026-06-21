"""
Replica EXATAMENTE o pipeline do notebook "Breast Cancer Wisconsin Diagnostic"
e exporta os artefatos que a API (app/main.py) precisa para servir os dados:

  - modelos treinados (.pkl) -> Regressão Logística, LDA, QDA, Naive Bayes Gaussiano
  - scaler treinado (.pkl)   -> RobustScaler, igual ao notebook
  - métricas de avaliação de cada modelo (data/metrics.json)
  - pacientes do conjunto de TESTE, com a previsão de cada modelo (data/patients.json)
  - lista das colunas de features, na ordem usada pelo modelo (data/feature_columns.json)

COMO RODAR:
  1. Coloque o "data.csv" (o mesmo arquivo usado no Colab) na raiz deste projeto,
     ao lado deste script.
  2. python3 train_and_export.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler

DATA_CSV = "data.csv"
OUT_DIR = "data"

if not os.path.exists(DATA_CSV):
    sys.exit(
        f'Arquivo "{DATA_CSV}" não encontrado.\n'
        f"Coloque o data.csv usado no notebook na raiz do projeto (mesma pasta deste script) e rode de novo."
    )

# ---------------------------------------------------------------------------
# 1. Carregamento e limpeza -> IDÊNTICO às células 5, 9 e 11 do notebook
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV)

# Normaliza nomes de coluna (o CSV original do Kaggle traz "concave points_mean"
# com espaço; padronizamos para "concave_points_mean" para facilitar o uso na API).
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Remove colunas inúteis (id e a coluna fantasma "Unnamed: 32")
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
df = df.drop(columns=["id"], errors="ignore")

print(f"Formato da base após limpeza: {df.shape}")

# Mapeia o alvo para binário (M=1, B=0), igual ao notebook
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# ---------------------------------------------------------------------------
# 2. Split em X / y -> células 13 e 15 do notebook
# ---------------------------------------------------------------------------
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

# ---------------------------------------------------------------------------
# 3. Padronização -> RobustScaler, célula 45/47 do notebook (TIPO_ESCALA='robusto')
# ---------------------------------------------------------------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Treinamento dos 4 modelos -> célula 56 do notebook
# ---------------------------------------------------------------------------
model_lr = LogisticRegression(max_iter=10000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train_scaled, y_train)

# QDA: tentamos primeiro EXATAMENTE como no notebook (sem reg_param). Se a base
# real tiver a mesma colinearidade forte (raio/perímetro/área, ~0.98 de
# correlação, identificada na própria EDA do grupo) e isso deixar a matriz de
# covariância singular, caímos com reg_param=0.15 para regularizar e o script
# avisa no console -> NÃO é necessário mudar nada no notebook por causa disso,
# é só uma rede de segurança aqui no script de exportação.
try:
    model_qda = QuadraticDiscriminantAnalysis()
    model_qda.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"[aviso] QDA sem reg_param falhou ({e}). Tentando com reg_param=0.15...")
    model_qda = QuadraticDiscriminantAnalysis(reg_param=0.15)
    model_qda.fit(X_train_scaled, y_train)

model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

print("Regressão Logística, LDA, QDA e Naive Bayes treinados!")

fitted = {
    "lr": ("Regressão Logística", model_lr),
    "lda": ("Análise Discriminante Linear (LDA)", model_lda),
    "qda": ("Análise Discriminante Quadrática (QDA)", model_qda),
    "nb": ("Naive Bayes Gaussiano", model_nb),
}

# ---------------------------------------------------------------------------
# 5. Avaliação -> seção 4 do notebook (matriz de confusão, recall, f1, etc.)
# ---------------------------------------------------------------------------
metrics = {}
predictions_per_model = {}
proba_per_model = {}

for key, (label, model) in fitted.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    predictions_per_model[key] = y_pred
    proba_per_model[key] = y_proba

    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    metrics[key] = {
        "label": label,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
        },
    }

# Recall como métrica norteadora -> é a própria conclusão do notebook (célula 38):
# "elegendo-se o Recall como métrica norteadora devido à gravidade médica de um falso negativo"
best_model_key = max(metrics, key=lambda k: metrics[k]["recall"])
metrics_payload = {
    "models": metrics,
    "best_model": best_model_key,
    "test_size": len(y_test),
    "train_size": len(y_train),
    "class_balance_test": {
        "benigno": int((y_test == 0).sum()),
        "maligno": int((y_test == 1).sum()),
    },
}

# ---------------------------------------------------------------------------
# 6. Lista de pacientes do conjunto de teste (com previsão de TODOS os modelos)
# ---------------------------------------------------------------------------
candidate_display = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "concavity_mean", "symmetry_mean",
]
display_features = [f for f in candidate_display if f in X.columns]

patients = []
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

for i in range(len(X_test_reset)):
    row = X_test_reset.iloc[i]
    patient = {
        "id": i + 1,
        "actual": "maligno" if y_test_reset.iloc[i] == 1 else "benigno",
        "features": {f: round(float(row[f]), 3) for f in display_features},
        "all_features": {f: round(float(row[f]), 4) for f in X.columns},
        "predictions": {},
    }
    for key in fitted:
        pred = int(predictions_per_model[key][i])
        proba = float(proba_per_model[key][i])
        patient["predictions"][key] = {
            "label": "maligno" if pred == 1 else "benigno",
            "probability": round(proba, 4),
            "correct": bool(pred == y_test_reset.iloc[i]),
        }
    patients.append(patient)

# ---------------------------------------------------------------------------
# 7. Exporta tudo para data/
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

for key, (label, model) in fitted.items():
    joblib.dump(model, f"{OUT_DIR}/model_{key}.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

with open(f"{OUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

with open(f"{OUT_DIR}/patients.json", "w", encoding="utf-8") as f:
    json.dump(patients, f, ensure_ascii=False, indent=2)

with open(f"{OUT_DIR}/feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

mean_features = [c for c in X.columns if c.endswith("_mean")]

corr_matrix = X_train[mean_features].corr().round(4)
correlation_payload = {
    "features": mean_features,
    "matrix": corr_matrix.values.tolist(),
}
with open(f"{OUT_DIR}/correlation.json", "w", encoding="utf-8") as f:
    json.dump(correlation_payload, f, ensure_ascii=False, indent=2)

from scipy.stats import gaussian_kde

distributions_payload = {"features": mean_features, "benigno": {}, "maligno": {}}
for feature in mean_features:
    values = X_train[feature].values
    span = values.max() - values.min()
    grid = np.linspace(values.min() - 0.1 * span, values.max() + 0.1 * span, 80)

    for label, code in (("benigno", 0), ("maligno", 1)):
        subset = values[(y_train == code).values]
        kde = gaussian_kde(subset)
        density = kde(grid)
        distributions_payload[label][feature] = {
            "x": [round(float(v), 4) for v in grid],
            "y": [round(float(v), 6) for v in density],
        }

with open(f"{OUT_DIR}/distributions.json", "w", encoding="utf-8") as f:
    json.dump(distributions_payload, f, ensure_ascii=False, indent=2)

print("\nExportação concluída!")
print(f"- {len(patients)} pacientes de teste exportados")
print(f"- Melhor modelo (por Recall): {metrics[best_model_key]['label']}")
for key, m in metrics.items():
    print(f"  [{key}] acc={m['accuracy']} recall={m['recall']} f1={m['f1']} auc={m['roc_auc']}")
