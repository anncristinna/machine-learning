# Breast Cancer Diagnostic API

Backend que treina os mesmos 4 modelos do notebook (Regressão Logística, LDA,
QDA, Naive Bayes Gaussiano) sobre o `data.csv` real e expõe os resultados via
API REST (FastAPI), pronta para um front-end consumir depois.

## ⚠️ O que falta para rodar

Este projeto **não inclui o `data.csv`**, pois ele não estava no upload do
notebook — só o `.ipynb`. Nenhuma alteração é necessária no seu notebook em
si; o script abaixo já replica o pipeline dele de forma independente.

## Estrutura

```
breast-cancer-api/
├── data.csv                 ← VOCÊ PRECISA COLOCAR AQUI (não incluso)
├── train_and_export.py      # treina os modelos e exporta os artefatos
├── requirements.txt
├── data/                    # gerado automaticamente pelo script acima
│   ├── model_lr.pkl
│   ├── model_lda.pkl
│   ├── model_qda.pkl
│   ├── model_nb.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── patients.json
│   └── feature_columns.json
└── app/
    └── main.py               # a API em si (FastAPI)
```

## Passo a passo

### 1. Instalar dependências
```bash
cd breast-cancer-api
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Colocar o `data.csv`
Copie o mesmo `data.csv` que você usa no Colab/notebook para a **raiz** desta
pasta (ao lado de `train_and_export.py`).

### 3. Treinar e exportar os artefatos
```bash
python3 train_and_export.py
```
Isso vai:
- replicar a limpeza (drop de `id`/`Unnamed: 32`, mapear `diagnosis` M/B → 1/0)
- fazer o mesmo split 80/20 (`random_state=42`, `stratify=y`)
- aplicar o mesmo `RobustScaler`
- treinar os 4 modelos
- salvar os `.pkl` e os `.json` dentro de `data/`

Você verá no console um resumo com accuracy/recall/f1/AUC de cada modelo —
confira se os números batem com os do seu notebook.

### 4. Subir a API
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Testar
Abra **http://localhost:8000/docs** — o Swagger gerado automaticamente pelo
FastAPI, onde dá pra testar cada endpoint clicando em "Try it out".

Endpoints disponíveis:

| Método | Rota                       | O que faz |
|--------|----------------------------|-----------|
| GET    | `/api/models`               | Lista os 4 modelos disponíveis |
| GET    | `/api/metrics`               | Métricas + matriz de confusão de cada modelo |
| GET    | `/api/patients`              | Lista paginada dos pacientes de teste (`?page=1&page_size=20&actual=maligno`) |
| GET    | `/api/patients/{id}`         | Detalhe completo de um paciente (todas as 30 features + previsão de cada modelo) |
| POST   | `/api/predict`               | Roda um modelo em cima de um conjunto de features customizado |

Exemplo rápido via terminal:
```bash
curl http://localhost:8000/api/metrics
curl "http://localhost:8000/api/patients?page=1&page_size=5"
```

## ⚠️ Ponto de atenção: QDA

A própria EDA do seu notebook identificou forte colinearidade entre
raio/perímetro/área (~0.98). Dependendo de como o `train_test_split` cair na
sua base real, isso pode deixar a matriz de covariância do QDA singular (erro
`LinAlgError`). O `train_and_export.py` já trata isso automaticamente: tenta
primeiro `QuadraticDiscriminantAnalysis()` (igual ao notebook) e, só se
falhar, cai para `reg_param=0.15` com um aviso no console — **nenhuma ação
sua é necessária**, é só pra você saber por que isso está lá caso apareça o
aviso.

## Próximo passo

Quando você validar que os números batem com o notebook, me avise que a
gente parte para o front-end (o dashboard de avaliação que combinamos).
