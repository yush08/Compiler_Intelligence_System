# Compiler Intelligence System

A small academic project that takes **Three-Address Code (TAC)**, inspects its structure, decides which compiler optimization is most useful, applies it, and measures how much the code improved.

Strategy selection supports **three modes**: rule-based scoring, a trained Random Forest classifier with SHAP explainability, and a head-to-head comparison of both selectors on the same program.

## How it works

```
TAC
 → Feature Extraction (8 features)
 → Strategy Selection (Rule-Based | ML | Compare Both)
 → Optimization Pass
 → Measurement (instruction + operation reduction)
 → Re-analysis / Convergence
```

The system extracts 8 features from the TAC, scores/selects a strategy, runs it, compares the instruction count, and repeats until no pass reduces the count any further (or the iteration limit is reached).

## Features

**Six optimization strategies**

1. Constant Folding
2. Constant Propagation
3. Common Subexpression Elimination
4. Dead Code Elimination
5. Algebraic Simplification
6. Strength Reduction (documented no-op)

**Three selector modes**

- **Rule-Based** — deterministic scorer (`backend/scoring/rule_based.py`)
- **ML** — trained `RandomForestClassifier` with `predict_proba`, SHAP explanations
- **Compare Both** — two independent runs from the same original TAC, winner chosen by measured outcome

## Implemented now

Everything listed here is working code you can run today.

**Backend — FastAPI (`backend/api/`)**

- `GET /api/health` — service status, mode, readiness
- `POST /api/analyze` — extract features, score all strategies, return rule-based selection + ML prediction + selection metadata
- `POST /api/optimize` — run iterative optimizer (supports `selector: "rule" | "ml" | "compare"` + explicit `strategy`)
- `POST /api/simulate` — bounded simulation with selector support, optional SHAP
- `GET /api/ml/status` — model artifact availability, version, config, SHAP readiness
- `GET /api/ml/metrics` — held-out evaluation (train/validation/test), per-class metrics, confusion matrix
- `GET /api/ml/feature-importance` — native impurity decrease vs SHAP mean|value|
- `POST /api/ml/predict` — ML prediction from TAC (8 features only)
- `POST /api/ml/explain` — SHAP breakdown for a TAC (contributions, base value, reconstruction check)
- `POST /api/compare` — head-to-head Rule vs ML on one program (agreement + measured winner)
- `GET /api/ml/comparison` — corpus-level comparison on the test split (prediction quality + optimization effectiveness)
- `GET /api/dataset/summary` — corpus stats, per-category split coverage (17 categories)
- `GET /api/dataset/samples` — paginated browse with filters
- `GET /api/dataset/samples/{id}` — detail view with live rule-based/ML selections and measured outcomes

**ML package (`backend/ml/`)**

- `train.py` — deterministic training (`python -m backend.ml.train`), 18-point grid search on validation split only, joblib artifacts to `artifacts/ml/`
- `model.py` — artifact persistence (model.joblib, metadata.json, evaluation.json, feature_importance.json, comparison.json)
- `predict.py` — `predict_from_tac` with 6-dp probability rounding
- `explain.py` — TreeExplainer with graceful unavailable fallback; `_get_explainer` is the failure-injection hook
- `evaluate.py` — accuracy, macro-F1 (primary), weighted-F1, per-class, confusion matrix on untouched test set
- `compare.py` — two separate comparisons: prediction quality (macro-F1) vs optimization effectiveness (measured improvement), never merged
- `schemas.py` — Pydantic models for all ML endpoints

**Frontend — React + TypeScript + Vite (`D:\PROJECTS\front\compiler\artifacts\compiler-intelligence`)**

- **Workbench** — original analyzer/optimizer with three selector tabs (Rule/ML/Compare Both)
- **ML Insights** — model status, native vs SHAP importance, per-prediction SHAP explanation with reconstruction check
- **Evaluation** — held-out metrics (train/val/test), per-class, confusion matrix; corpus-level Rule vs ML comparison with `improved_pct` bars and winner
- **Compare** — two independent runs on the same TAC, agreement across full strategy sequence, winner from measured outcome
- **Dataset** — corpus summary, label distribution, per-category train/val/test split matrix (amber for zero-test categories), paginated browser with detail panel showing live selections + measured outcomes

**Measurement (`backend/optimizer/measure.py`)**

- `operation_count` / `work_remaining` — instructions + remaining arithmetic ops
- `measure_pass` — per pass: instruction counts, instruction reduction, reduction %, operation reduction, composite `useful_improvement`
- `best_strategy_from` — argmax over six passes with documented tie-break

**Dataset generation (`experiments/generate_dataset.py`)**

- Reproducible, seeded generator producing 177 unique TAC programs across 17 opportunity categories; re-running produces byte-identical files
- Labels (`best_strategy`) derived by actually running all six passes and taking argmax of `useful_improvement` — no hand-written labels
- Stores all six passes' four measurement counters per program plus the 8 features and generation metadata
- Validation checks (parse round-trip, non-negative counts, percentages in `[0,100]`, duplicate detection, label coverage)
- Deterministic 70/15/15 `train` / `validation` / `test` split (124 / 27 / 26)
- **4 of 17 generation categories have zero test samples** (disclosed in limitations)

## Project structure

```
compiler/
├── backend/                 Python backend
│   ├── api/                 FastAPI server, request pipeline, serialization, dataset_service
│   ├── ml/                  ML package (train, predict, explain, evaluate, compare, model)
│   ├── tac/                 TAC instruction model and parser
│   ├── features/            Feature extractor (8 features — single source of truth)
│   ├── scoring/             Rule-based strategy scoring (STRATEGY_TIE_ORDER)
│   ├── optimizations/       The six optimization passes
│   └── optimizer/           Iterative engine + measure.py pass measurement
├── frontend/                Legacy React UI (Vite, Tailwind) — pre-Phase 3
├── artifacts/
│   └── compiler-intelligence/  Phase 3 React + TypeScript UI (Vite, Tailwind)
│       ├── src/
│       │   ├── views/           Workbench, ML Insights, Evaluation, Compare, Dataset
│       │   ├── components/      Panel kit (SectionHeader, Panel, Tag, Metric, TacInput, RunButton)
│       │   ├── services/        Typed API client (api.ts)
│       │   └── App.tsx          View switch, nav, harness-critical testids
│       ├── verify-render.mjs    Headless SSR render check
│       ├── verify-integration.mjs  End-to-end wiring check
│       └── package.json
├── tests/                   pytest test suite (191 passed)
├── examples/                Sample TAC files
├── experiments/             Benchmark corpus, experiment runner, dataset generator
├── dataset/                 Generated labelled dataset (raw/processed/metadata)
│   └── processed/optimization_dataset.csv  — 177×45, 5 classes
├── artifacts/ml/            Trained model artifacts (model.joblib, metadata.json, evaluation.json, …)
├── report/                  Evaluation output
├── app.py                   Streamlit demo dashboard (legacy)
├── requirements.txt         Python deps (fastapi, uvicorn, pydantic, scikit-learn, joblib, shap, pandas, numpy, pytest, httpx, streamlit, plotly)
└── README.md
```

Notes:

- The real test suite lives in `tests/` and is run with `pytest`.
- `.venv/`, `node_modules/`, `__pycache__/`, `.pytest_cache/`, `frontend/dist/`, `artifacts/ml/` are generated local directories and are not part of the source.
- `dataset/` is produced by `python experiments/generate_dataset.py`; it is regenerated, never hand-edited.
- ML artifacts are produced by `python -m backend.ml.train`; they are regenerated, never hand-edited.

## Requirements

Verified in this environment — you need:

| Tool | Notes |
|---|---|
| Python 3.10+ | Tested with Python 3.12.5 |
| pip | Comes with Python |
| Node.js 18+ | Tested with Node.js v22 |
| npm | The frontend ships `package-lock.json` |
| Git | Optional |

Python packages (from `requirements.txt`): `streamlit`, `plotly`, `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `joblib`, `shap`, `pandas`, `numpy`, and optionally `pytest`, `pytest-cov`, `httpx`.

## Backend setup

Open PowerShell in the project root:

```powershell
cd D:\PROJECTS\compiler

# create a virtual environment the first time only
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Train the ML model** (one-time, deterministic, ~30 s):

```powershell
python -m backend.ml.train
```

This writes `artifacts/ml/model.joblib`, `metadata.json`, `evaluation.json`, `feature_importance.json`, `comparison.json`.

**Run the API server**:

```powershell
uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
```

Check that it is running: <http://127.0.0.1:8000/api/health>

Interactive API docs: <http://127.0.0.1:8000/docs>

## Frontend setup (Phase 3 UI)

In a second PowerShell window:

```powershell
cd D:\PROJECTS\front\compiler\artifacts\compiler-intelligence
npm install
npm run dev
```

Open <http://127.0.0.1:5199/>

Other scripts: `npm run build` (type-check + production build), `npm run lint`, `npm run preview`.

The frontend calls the backend at `http://127.0.0.1:8000` by default (override with `VITE_API_URL`).

## Streamlit dashboard (optional, legacy)

A separate demo UI that imports the backend directly, without the REST API:

```powershell
cd D:\PROJECTS\compiler
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Using the application

1. Start the backend, then the frontend.
2. **Workbench** — pick a built-in scenario or type your own TAC. Choose selector mode (Rule/ML/Compare Both) and press **Run to convergence**. The timeline, diff, and decision trace update with real per-iteration data.
3. **ML Insights** — press **Refresh** for model status, **Load** for global importance, enter any TAC and **Explain prediction** for SHAP breakdown with arithmetic reconstruction check.
4. **Evaluation** — press **Load evaluation** for held-out metrics and corpus-level Rule vs ML comparison (winner from measured improvement).
5. **Compare** — enter any TAC, press **Run both selectors**; agreement is read across the whole pass sequence, winner from measured outcome.
6. **Dataset** — browse the 177-program corpus with per-category split coverage; click any row for detail with live rule-based/ML selections and measured outcomes.

## API endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/api/health` | — | Service status, mode, readiness |
| `POST` | `/api/analyze` | `{ "tac": "..." }` | Features, 6 strategy scores, rule selection + ML prediction |
| `POST` | `/api/optimize` | `{ "tac": "...", "strategy"?, "selector"?: "rule\|ml\|compare" }` | Iterative optimizer with selector support |
| `POST` | `/api/simulate` | `{ "tac": "...", "selector"?: "rule\|ml", "max_iterations"?, "include_shap"? }` | Bounded simulation |
| `GET` | `/api/ml/status` | — | Artifact availability, version, config, SHAP readiness |
| `GET` | `/api/ml/metrics` | — | Train/val/test metrics, per-class, confusion matrix |
| `GET` | `/api/ml/feature-importance` | — | Native impurity decrease + SHAP mean\|value\| |
| `POST` | `/api/ml/predict` | `{ "tac": "..." }` | ML prediction from TAC (8 features) |
| `POST` | `/api/ml/explain` | `{ "tac": "...", "top_k"?: 3 }` | SHAP contributions, base value, reconstruction check |
| `POST` | `/api/compare` | `{ "tac": "..." }` | Head-to-head Rule vs ML on one program |
| `GET` | `/api/ml/comparison` | — | Corpus-level comparison on test split |
| `GET` | `/api/dataset/summary` | — | Corpus stats, per-category split coverage |
| `GET` | `/api/dataset/samples` | query: `page, page_size, split, category, q` | Paginated browse |
| `GET` | `/api/dataset/samples/{id}` | — | Detail with live selections + measured outcomes |

Error policy:

- Read ML endpoints → `200` with `available: false` + `reason`
- `/api/optimize` or `/api/simulate` with `selector: 'ml'` and no model → `503` `{ "code": "model_unavailable", "remedy": "python -m backend.ml.train" }`
- Bad TAC → `400`
- Bad field / validation → `422`

### Example request

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/analyze `
  -ContentType 'application/json' `
  -Body (@{ tac = "t1 = 10`nt2 = 20`nt3 = t1 + t2`nt4 = t1 + t2`nt5 = t3 * 1`nt6 = 100`nt7 = t6 + 50`nunused = t7`nt8 = t3 * 2" } | ConvertTo-Json)
```

## ML training & evaluation

```powershell
cd D:\PROJECTS\compiler
.\.venv\Scripts\Activate.ps1

# Train (deterministic, ~30 s)
python -m backend.ml.train

# Evaluation is embedded in training; artifacts written to artifacts/ml/
#   evaluation.json     — train/val/test metrics, per-class, confusion matrix
#   comparison.json     — Rule vs ML on test split (prediction + optimization)
#   feature_importance.json — native + SHAP importance
#   metadata.json       — seed, config, splits, classes, training command
```

Training is **reproducible**: seed `20261005`, `n_estimators=300`, grid search on validation macro-F1 only (18 configs), test evaluated exactly once.

### Model details

- Algorithm: `RandomForestClassifier`
- Version: `rf-v1-b4ee0236dea17f02`
- Selected config: `max_depth=8`, `min_samples_leaf=4`, `class_weight="balanced"`
- Test split (26 samples): accuracy 1.0000, macro-F1 1.0000, weighted-F1 1.0000
- Rule vs ML on test: rule macro-F1 0.7500 vs ML 1.0000; optimization rule mean useful 6.73 / 69.2% improved vs ML 7.81 / 84.6% → **winner ML**
- 4 of 26 test programs disagree between selectors

## Dataset (verified)

```powershell
cd D:\PROJECTS\compiler

python experiments/generate_dataset.py                              # full dataset -> ./dataset
python experiments/generate_dataset.py --samples 20 --out .\dataset-smoke   # small smoke run
```

Output layout:

```
dataset/
├── raw/tac_programs.json                  original programs + generation category
├── processed/optimization_dataset.csv     one row per program (pandas-friendly)
├── processed/optimization_dataset.json    same records + schema hints
└── metadata/
    ├── summary.json                       label counts, feature ranges, coverage
    ├── validation_report.json             validation checks and warnings
    └── splits.json                        70/15/15 train/validation/test
```

Each row carries the 8 features, all six passes' measurements
(`<strategy>_instruction_reduction`, `<strategy>_reduction_percentage`,
`<strategy>_operation_reduction`, `<strategy>_useful_improvement`), the label
`best_strategy`, `has_opportunity`, the generation category, and the split.

Verified with pandas: **177 rows × 45 columns, 0 nulls, 177 distinct programs**.

Label distribution (measured, not assigned):

| `best_strategy` | count |
|---|---|
| dead_code_elimination | 59 |
| constant_folding | 46 |
| constant_propagation | 31 |
| algebraic_simplification | 23 |
| common_subexpression | 18 |
| strength_reduction | 0 |

`strength_reduction` scores 0 because the pass is a deliberate no-op. The generator records that zero and emits a validation warning instead of inventing wins for it.

**Test split coverage by generation category (17 categories, 13 present in test):**

- 4 categories have **zero test samples**: `algebraic_simplification`, `common_subexpression`, `dead_code_elimination`, `no_opportunity`
- No test-only categories
- All 5 labels present in test with support 2–8
- Dataset has no `operation_reduction` columns (computed at runtime by `measure_pass`)

## Testing

```powershell
cd D:\PROJECTS\compiler
.\.venv\Scripts\Activate.ps1
pytest
```

Current result: **191 passed, 1 warning** (~66 s).

The suite covers:
- All six optimizations, feature extraction, rule-based scoring
- Iterative optimizer, measurement helpers, API endpoints
- ML package: training, prediction, SHAP explanation, evaluation, comparison
- Dataset generator, dataset service (per-category splits)
- API ML endpoints: status, metrics, feature importance, predict, explain, compare, dataset
- Model-unavailable paths (fixture `no_model`) — 503 on execution, `available:false` on reads

## Current limitations

- Strategy selection is **rule-based + one Random Forest model** — no deep learning or ensemble-of-models.
- The optimizer measures improvement by **instruction count + operation count** (`useful_improvement`). A pass that rewrites without removing instructions (e.g., constant folding `t = 5 + 5` → `t = 10`) reports 0 instruction reduction; the composite metric captures the operation reduction.
- The parser covers a **subset of TAC**: assignments, unary minus, integer/float constants, comparisons, arithmetic (`+ - * / %`), labels, `goto`, conditional `if <relop> goto`. No function calls, arrays, pointers.
- **Strength Reduction** is a documented conservative no-op: its instruction-level helper always declines, so the pass never changes the code and measures 0 on every program. It stays because the six-strategy contract requires it, and the dataset reports its zero.
- **Algebraic Simplification** only applies a fixed list of identities (`x + 0`, `x * 1`, `x - 0`, `x / 1`, `x * 0`), so many algebraic rewrites are out of scope.
- **ML model is preliminary** — trained on 124 samples, evaluated on 26 test samples. Macro-F1 1.0000 on this split is encouraging but not generalizable; see `train.py LIMITATIONS` (11 entries) for full disclosure including 4/17 generation categories with zero test samples.
- SHAP explanations reconstruct **only the predicted class** probability (TreeExplainer returns per-class contributions; we report the one for the argmax class). The nearest-other class typically differs by ~0.3 probability.
- The benchmark in `experiments/` is a generated corpus, not an external standard dataset.
- `examples/` is still empty.

## Future work

- Explore deep learning approaches to the same task.
- Extend the TAC grammar (function calls, arrays, more control flow).
- Add more optimization passes and data-flow based analysis.
- Give Strength Reduction enough loop context for it to actually rewrite code.
- Evaluate on a larger, externally recognised benchmark.
- Add cross-validation and confidence intervals to the ML evaluation.