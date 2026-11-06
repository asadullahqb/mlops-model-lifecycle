# mlops-model-lifecycle

A self-contained, runnable reference implementation of a **closed-loop ML
lifecycle**:

```
SERVING ──► scheduled DATA REFRESH ──► EVALUATION (key metric)
     ▲                                        │
     │                                        ▼
     │                               DRIFT DETECTION
     │                              (R2 threshold + PSI)
     │                                        │
     └── PROMOTE (if better) ◄── RETRAIN ◄────┘
```

It runs locally with **zero external dependencies** (a deterministic bundled
synthetic tabular regression dataset), is containerised with Docker, and is
structured so it can later be ported to **AWS SageMaker** (see the porting
section below).

* Stack: Python 3.10+, FastAPI, scikit-learn (GradientBoosting), APScheduler,
  Docker + docker-compose, GitHub Actions CI.
* Key metric: **R²** (configurable), plus RMSE for reporting.

---

## Repository layout

```
config.yaml                 # model / data / metric / threshold config (pluggable)
data/dataset.csv            # bundled baseline dataset (deterministic)
src/
  config.py                 # load config.yaml, resolve paths
  data.py                   # data loading + refresh() (simulates new batches)
  train.py                  # train + persist to versioned registry (models/)
  evaluate.py               # R2 + RMSE
  drift.py                  # model-quality drift (R2) + data drift (PSI)
  registry.py               # versioned registry + atomic promotion
  serve.py                  # FastAPI: GET /health, POST /predict
  orchestrate.py            # the closed loop
scripts/
  run_demo.sh               # one-command local demo (bash)
  run_demo.ps1              # one-command local demo (PowerShell)
  predict_once.py           # send a single /predict request
tests/                      # pytest suite (unit + integration)
Dockerfile / docker-compose.yml
Makefile
.github/workflows/ci.yml    # lint + test + build image
```

## Quickstart

No Docker required — the components can run directly. Docker instructions are
at the bottom.

```bash
pip install -r requirements.txt

# 1. train a baseline model (writes models/<version>/ + promotes it)
python -m src.train

# 2. serve it
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# in another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}'
```

Or run the whole demo with one command:

```bash
bash scripts/run_demo.sh      # or: make demo
# Windows: powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

## The closed loop, step by step

`python -m src.orchestrate` runs one cycle:

1. **refresh** — `src/data.py::refresh()` appends a new batch to the dataset.
2. **evaluate** — `src/evaluate.py` scores the *current* model on the new batch
   (R² + RMSE).
3. **detect drift** — `src/drift.py` raises a flag if either:
   * *model-quality drift*: R² on the batch falls below
     `drift.model_quality.threshold`; or
   * *data drift*: PSI of any feature vs the training-time baseline exceeds
     `drift.data.psi_threshold`.
4. **retrain** — if drifted, `src/train.py::retrain_and_compare()` retrains on
   the full dataset and compares old vs new on the *same* held-out split.
5. **promote** — if the new R² beats the current model's R², the registry
   pointer (`models/current.json`) is swapped **atomically** (temp file +
   `os.replace`). The running serve process hot-reloads the new model with no
   restart.

Every cycle appends a JSON event to `runs/` (the audit trail).

Inject a distribution shift on purpose (for the demo / tests):

```bash
python -m src.orchestrate --inject-drift
```

Run on a recurring schedule (what the `scheduler` compose service does):

```bash
python -m src.orchestrate --loop --interval 5
```

## Docker

```bash
docker compose up --build
# serve on :8000, scheduler runs the loop every scheduler.interval_minutes
```

## Configuration (`config.yaml`)

Everything pluggable lives in one file:

| Section | What it controls |
|---|---|
| `data.source` | dataset path; `data.target` label column; `data.features` (or `null` = all but target) |
| `data.refresh` | batch size / seed for simulated batches |
| `data.drift` | demo-only drift injection (shift features by `shift_magnitude`) |
| `model.type` + `model.params` | which model factory to build |
| `metric` | key metric name + promotion/drift threshold |
| `drift.*` | model-quality and data-drift thresholds |
| `registry.dir` / `runs.dir` | registry and audit-trail locations |
| `scheduler.interval_minutes` | loop cadence |

## Swapping in a real model

The model is pluggable through a single seam: `src/train.py::build_model()`
dispatches on `model.type` using `MODEL_FACTORIES`. Add a factory (or set
`model.type` to an existing one) and it flows through train → evaluate →
retrain → serve unchanged.

### Adapter for the UHI XGBoost model (`open-science-challenge`)

That model is an XGBoost **regressor** predicting an UHI index from tabular
features (Landsat LST, Mesonet weather, building footprints). It is a drop-in
fit because the lifecycle is regression-based (R² is already the key metric).

```python
# src/train.py
def _xgboost(params):
    from xgboost import XGBRegressor
    return XGBRegressor(**params)

MODEL_FACTORIES["xgboost.XGBRegressor"] = _xgboost
```

```yaml
# config.yaml
model:
  type: xgboost.XGBRegressor
  params:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    random_state: 42
```

Then replace `src/data.py::refresh()` with a loader that pulls the latest
training features (e.g. `Training_data_uhi_index.csv` plus the refreshed
Mesonet/Landsat features) and writes a new batch to `data.source`. `pip install
xgboost` is the only extra dependency.

### Adapter for the churn model (`TelcoMLOpsPOC`)

Churn is **binary classification**, so two things change: the model factory and
the key metric.

```python
# src/train.py
def _xgb_classifier(params):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)

MODEL_FACTORIES["xgboost.XGBClassifier"] = _xgb_classifier
```

Swap the metric in `src/evaluate.py` to e.g. ROC-AUC or F1 and set the
threshold accordingly:

```yaml
model:
  type: xgboost.XGBClassifier
  params: {n_estimators: 300, max_depth: 5, random_state: 42}
metric:
  name: auc
  threshold: 0.75
drift:
  model_quality:
    threshold: 0.75
```

`src/drift.py::data_drift()` is model-agnostic (PSI is computed on features),
so it works for classification as-is.

### Real data source

The whole data layer is behind `src/data.py::refresh(cfg) -> pd.DataFrame`.
To go live, implement that one function against your store (S3 / Postgres /
Snowflake / Kafka) and append the batch to `cfg["data"]["source"]`. Nothing
downstream changes.

## AWS SageMaker port

The components map 1:1 onto AWS services, so the closed loop can be lifted with
no redesign:

| Local component | AWS service | Notes |
|---|---|---|
| `refresh()` | **SageMaker Processing job** | pull new batch from S3/Glue/Athena, write to the feature store |
| `train()` | **SageMaker Training job** | same `model.type` factory, register artefact in the Model Registry |
| `evaluate()` | **SageMaker Processing job** (post-training) | compute R²/RMSE, store in Model Registry / S3 |
| `drift.py` (data drift) | **SageMaker Model Monitor** | PSI/KL divergence on a monitoring schedule |
| `drift.py` (quality drift) | **Model Monitor** with ground truth | metric below threshold → CloudWatch alarm |
| `orchestrate.py` | **EventBridge → SageMaker Pipelines** | the loop becomes a DAG triggered by the alarm |
| `registry.promote()` | **Model Registry** (approve `PendingManualApproval` or auto-approve) | `models/current.json` ≈ the `latest` alias |
| `serve.py` | **SageMaker Endpoint** | FastAPI app → `inference.py` handler behind the endpoint |

Suggested pipeline: `EventBridge (schedule/alarm)` → `Processing (refresh)` →
`Training` → `Processing (evaluate)` → `Condition (R² better?)` →
`RegisterModel + promote` → `Endpoint update`. Model Monitor keeps an eye on the
live endpoint between runs.

## CI

`.github/workflows/ci.yml` runs `ruff check`, `pytest`, and a `docker build`
on every push/PR.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check src tests
```
