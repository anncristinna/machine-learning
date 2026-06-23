import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler

DATA_CSV = "data.csv"
OUT_DIR  = "data"

if not os.path.exists(DATA_CSV):
    sys.exit(f'Arquivo "{DATA_CSV}" não encontrado.')

df = pd.read_csv(DATA_CSV)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
df = df.drop(columns=["id"], errors="ignore")
print(f"Formato da base após limpeza: {df.shape}")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=10000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train_scaled, y_train)

try:
    model_qda = QuadraticDiscriminantAnalysis()
    model_qda.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"[aviso] QDA sem reg_param falhou ({e}). Tentando com reg_param=0.15...")
    model_qda = QuadraticDiscriminantAnalysis(reg_param=0.15)
    model_qda.fit(X_train_scaled, y_train)

model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

print("Modelos baseline treinados!")

fitted = {
    "lr":  ("Regressão Logística", model_lr),
    "lda": ("Análise Discriminante Linear (LDA)", model_lda),
    "qda": ("Análise Discriminante Quadrática (QDA)", model_qda),
    "nb":  ("Naive Bayes Gaussiano", model_nb),
}

metrics = {}
predictions_per_model = {}
proba_per_model = {}

for key, (label, model) in fitted.items():
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    predictions_per_model[key] = y_pred
    proba_per_model[key] = y_proba
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics[key] = {
        "label":    label,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision":round(precision_score(y_test, y_pred), 4),
        "recall":   round(recall_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
        },
    }

best_model_key = max(metrics, key=lambda k: metrics[k]["recall"])
metrics_payload = {
    "models": metrics,
    "best_model": best_model_key,
    "test_size":  len(y_test),
    "train_size": len(y_train),
    "class_balance_test":  {"benigno": int((y_test  == 0).sum()), "maligno": int((y_test  == 1).sum())},
    "class_balance_train": {"benigno": int((y_train == 0).sum()), "maligno": int((y_train == 1).sum())},
}

print("\nExecutando GridSearchCV (scoring=recall)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_configs = {
    "lr": {
        "base":   LogisticRegression(max_iter=10000, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
    },
    "lda": {
        "base":   LinearDiscriminantAnalysis(),
        "params": {"shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7], "solver": ["lsqr", "eigen"]},
    },
    "qda": {
        "base":   QuadraticDiscriminantAnalysis(),
        "params": {"reg_param": [0.05, 0.1, 0.2, 0.3, 0.5]},
    },
    "nb": {
        "base":   GaussianNB(),
        "params": {"var_smoothing": np.logspace(0, -9, num=10).tolist()},
    },
}

metrics_tuned = {}
for key, cfg in gs_configs.items():
    gs = GridSearchCV(
        estimator=cfg["base"],
        param_grid=cfg["params"],
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        error_score=np.nan,
    )
    gs.fit(X_train_scaled, y_train)
    best_m  = gs.best_estimator_
    y_pred  = best_m.predict(X_test_scaled)
    y_proba = best_m.predict_proba(X_test_scaled)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()

    clean_params = {}
    for k, v in gs.best_params_.items():
        if isinstance(v, (np.integer,)):  clean_params[k] = int(v)
        elif isinstance(v, (np.floating,)): clean_params[k] = round(float(v), 9)
        else: clean_params[k] = v

    metrics_tuned[key] = {
        "label":       fitted[key][0],
        "best_params": clean_params,
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred), 4),
        "recall":      round(recall_score(y_test, y_pred), 4),
        "f1":          round(f1_score(y_test, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
        },
    }
    print(f"  [{key}] best={clean_params} recall={metrics_tuned[key]['recall']}")

best_tuned_key = max(metrics_tuned, key=lambda k: metrics_tuned[k]["recall"])
metrics_tuned_payload = {
    "models":     metrics_tuned,
    "best_model": best_tuned_key,
    "scoring":    "recall",
}

candidate_display = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","concavity_mean","symmetry_mean",
]
display_features = [f for f in candidate_display if f in X.columns]

patients = []
X_test_r = X_test.reset_index(drop=True)
y_test_r = y_test.reset_index(drop=True)

for i in range(len(X_test_r)):
    row = X_test_r.iloc[i]
    patient = {
        "id":           i + 1,
        "actual":       "maligno" if y_test_r.iloc[i] == 1 else "benigno",
        "features":     {f: round(float(row[f]), 3) for f in display_features},
        "all_features": {f: round(float(row[f]), 4) for f in X.columns},
        "predictions":  {},
    }
    for key in fitted:
        pred  = int(predictions_per_model[key][i])
        proba = float(proba_per_model[key][i])
        patient["predictions"][key] = {
            "label":       "maligno" if pred == 1 else "benigno",
            "probability": round(proba, 4),
            "correct":     bool(pred == y_test_r.iloc[i]),
        }
    patients.append(patient)

mean_features = [c for c in X.columns if c.endswith("_mean")]
corr_matrix   = X_train[mean_features].corr().round(4)
correlation_payload = {"features": mean_features, "matrix": corr_matrix.values.tolist()}

distributions_payload = {"features": mean_features, "benigno": {}, "maligno": {}}
for feature in mean_features:
    values = X_train[feature].values
    span   = values.max() - values.min()
    grid   = np.linspace(values.min() - 0.1 * span, values.max() + 0.1 * span, 80)
    for label, code in (("benigno", 0), ("maligno", 1)):
        subset  = values[(y_train == code).values]
        kde_fn  = gaussian_kde(subset)
        density = kde_fn(grid)
        distributions_payload[label][feature] = {
            "x": [round(float(v), 4) for v in grid],
            "y": [round(float(v), 6) for v in density],
        }

box_stats = {}
for col in X_train.columns:
    vals = X_train[col].values
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    iqr         = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    box_stats[col] = {
        "whisker_lo": round(float(vals[vals >= lower_fence].min()), 4),
        "q1":         round(float(q1), 4),
        "median":     round(float(median), 4),
        "q3":         round(float(q3), 4),
        "whisker_hi": round(float(vals[vals <= upper_fence].max()), 4),
        "outliers":   [round(float(v), 4) for v in vals[(vals < lower_fence) | (vals > upper_fence)].tolist()],
    }

pca_obj       = PCA(n_components=2, random_state=42)
X_train_pca   = pca_obj.fit_transform(X_train_scaled)
y_train_vals  = y_train.values

lr_2d = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
lr_2d.fit(X_train_pca, y_train_vals)

GRID_SIZE = 60
x_pad  = (X_train_pca[:, 0].max() - X_train_pca[:, 0].min()) * 0.08
y_pad  = (X_train_pca[:, 1].max() - X_train_pca[:, 1].min()) * 0.08
gx_min = float(X_train_pca[:, 0].min() - x_pad)
gx_max = float(X_train_pca[:, 0].max() + x_pad)
gy_min = float(X_train_pca[:, 1].min() - y_pad)
gy_max = float(X_train_pca[:, 1].max() + y_pad)

xx, yy = np.meshgrid(
    np.linspace(gx_min, gx_max, GRID_SIZE),
    np.linspace(gy_min, gy_max, GRID_SIZE),
)
Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

pca_payload = {
    "variance_explained": [round(float(v) * 100, 1) for v in pca_obj.explained_variance_ratio_],
    "grid": {
        "size": GRID_SIZE,
        "x_min": round(gx_min, 3), "x_max": round(gx_max, 3),
        "y_min": round(gy_min, 3), "y_max": round(gy_max, 3),
        "predictions": "".join(str(int(v)) for v in Z.ravel()),
    },
    "points": [
        {"x": round(float(X_train_pca[i, 0]), 3), "y": round(float(X_train_pca[i, 1]), 3), "label": int(y_train_vals[i])}
        for i in range(len(X_train_pca))
    ],
}

os.makedirs(OUT_DIR, exist_ok=True)

for key, (label, model) in fitted.items():
    joblib.dump(model, f"{OUT_DIR}/model_{key}.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

exports = {
    "metrics.json":         metrics_payload,
    "metrics_tuned.json":   metrics_tuned_payload,
    "patients.json":        patients,
    "feature_columns.json": list(X.columns),
    "correlation.json":     correlation_payload,
    "distributions.json":   distributions_payload,
    "boxplots.json":        {"features": list(X_train.columns), "stats": box_stats},
    "pca_boundary.json":    pca_payload,
}
for filename, data in exports.items():
    with open(f"{OUT_DIR}/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("\nExportação concluída!")
print(f"- Melhor modelo baseline (Recall): {metrics[best_model_key]['label']}")
print(f"- Melhor modelo tuned   (Recall): {metrics_tuned[best_tuned_key]['label']}")
for key, m in metrics.items():
    mt = metrics_tuned[key]
    delta = round((mt['recall'] - m['recall']) * 100, 1)
    sign  = '+' if delta >= 0 else ''
    print(f"  [{key}] recall {m['recall']} -> {mt['recall']} ({sign}{delta}%)")