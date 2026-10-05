# Compiler Intelligence System

A small academic project that takes **Three-Address Code (TAC)**, inspects its structure, decides which compiler optimization is most useful, applies it, and measures how much the code improved.

Strategy selection is **rule-based** (scoring formulas over extracted features). Machine learning is **not** implemented — it is listed only as future work.

## How it works

```
TAC
 → Feature Extraction
 → Rule-Based Strategy Scoring
 → Optimization Strategy Selection
 → Optimization
 → Measurement
 → Re-analysis / Convergence
```

The system extracts 8 features from the TAC, scores all six optimization strategies, picks the highest-scoring one, runs it, compares the instruction count, and repeats until no pass reduces the count any further (or the iteration limit is reached).

## Features

**Six optimization strategies**

1. Constant Folding
2. Constant Propagation
3. Common Subexpression Elimination
4. Dead Code Elimination
5. Algebraic Simplification
6. Strength Reduction

**Implemented functionality**

- TAC input and editing (React editor, Streamlit dashboard)
- TAC parsing (assignments, unary minus, integer/float constants, comparisons, arithmetic, labels, `goto`, conditional jumps)
- Feature extraction — 8 features: statements, constants, repeated expressions, dead statements, algebraic ops, loop ops, temporary variables, constant ratio
- Rule-based scoring with a plain-language reason for every strategy
- Automatic strategy selection (highest score)
- Iterative optimization engine (up to 20 iterations, stops when no improvement)
- Before / after TAC comparison
- Instruction-count and percentage reduction measurement
- Per-iteration log with convergence status
- FastAPI REST backend
- React + TypeScript frontend that talks to the backend
- Second Streamlit dashboard (`app.py`) that uses the backend directly

## Project structure

```
compiler/
├── backend/                 Python backend
│   ├── api/                 FastAPI server, request pipeline, serialization
│   ├── tac/                 TAC instruction model and parser
│   ├── features/            Feature extractor (8 features)
│   ├── scoring/             Rule-based strategy scoring
│   ├── optimizations/       The six optimization passes
│   └── optimizer/           Iterative optimization engine
├── frontend/                React + TypeScript UI (Vite, Tailwind)
│   ├── src/
│   │   ├── pages/           Dashboard page
│   │   ├── components/      UI components
│   │   ├── data/            Built-in demo scenarios
│   │   ├── services/        API client
│   │   └── types/           Shared TypeScript types
│   └── package.json
├── tests/                   pytest test suite
├── examples/                (currently empty) placeholder for sample TAC files
├── experiments/             Benchmark corpus and experiment runner
├── report/                  Evaluation output (data/results.json, data/per_program.csv)
├── app.py                   Streamlit demo dashboard
├── requirements.txt         Python dependencies
├── package.json             Root Node metadata (no dependencies)
└── pnpm-lock.yaml           Root lockfile
```

Notes:

- Root-level `debug_*.py`, `test_features.py`, `test_parser.py` and `*_probe.py` files are local helper/debug scripts. The real test suite lives in `tests/`.
- `.venv/`, `node_modules/`, `__pycache__/`, `.pytest_cache/` and `frontend/dist/` are generated local directories and are not part of the source.

## Requirements

Verified in this environment — you need:

| Tool | Notes |
|---|---|
| Python 3.10+ | Tested with Python 3.12.5 |
| pip | Comes with Python |
| Node.js 18+ | Tested with Node.js v22 |
| npm | The frontend ships `package-lock.json` |
| Git | Optional |

Python packages (from `requirements.txt`): `streamlit`, `plotly`, `fastapi`, `uvicorn`, `pydantic`, and optionally `pytest`, `pytest-cov`, `httpx`.

## Backend setup

Open PowerShell in the project root:

```powershell
cd D:\PROJECTS\compiler

# create a virtual environment the first time only
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn backend.api.server:app --reload --port 8000
```

Check that it is running: <http://127.0.0.1:8000/api/health>
Interactive API docs: <http://127.0.0.1:8000/docs>

## Frontend setup

In a second PowerShell window:

```powershell
cd D:\PROJECTS\compiler\frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

Other scripts: `npm run build` (type-check + production build), `npm run lint`, `npm run preview`.

The frontend calls the backend at `http://localhost:8000` by default (override with the `VITE_API_URL` environment variable). A Vite dev proxy for `/api` is also configured in `frontend/vite.config.ts`. The backend allows CORS from `localhost:5173`.

## Streamlit dashboard (optional)

A separate demo UI that imports the backend directly, without the REST API:

```powershell
cd D:\PROJECTS\compiler
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Using the application

1. Start the backend, then the frontend.
2. Pick one of the built-in scenarios (`constant_folding`, `cse`, `dead_code`, `mixed`) or type your own TAC into the editor.
3. Press **Analyze** — the backend returns the 8 features, the 6 strategy scores and the selected strategy.
4. Choose a strategy (or keep the suggested one) and press **Apply**.
5. The result panel shows the before/after TAC, the instruction counts, the reduction percentage, and the iteration log.

## API endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/api/health` | — | Service status |
| `POST` | `/api/analyze` | `{ "tac": "..." }` | Extract features, score all strategies, return the selected one |
| `POST` | `/api/optimize` | `{ "tac": "...", "strategy": "constant_folding" }` | Run the iterative optimizer and return before/after TAC plus iterations |
| `POST` | `/api/simulate` | `{ "tac": "...", "max_iterations": 20 }` | Run a bounded simulation |

`strategy` is optional; when omitted the backend picks the highest-scoring strategy. `max_iterations` must be between 1 and 20.

Invalid or unparsable TAC returns `400`.

### Example request

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/analyze `
  -ContentType 'application/json' `
  -Body (@{ tac = "t1 = 10`nt2 = 20`nt3 = t1 + t2`nt4 = t1 + t2`nt5 = t3 * 1`nt6 = 100`nt7 = t6 + 50`nunused = t7`nt8 = t3 * 2" } | ConvertTo-Json)
```

The response contains `tac`, `features`, `strategies`, `selected_strategy`, `iterations` and `final_result`.

## Example TAC input

```
t1 = 10
t2 = 20
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
t6 = 100
t7 = t6 + 50
unused = t7
t8 = t3 * 2
```

Running this through `/api/optimize` gives:

- Selected strategy: **Constant Folding** (score 0.91)
- Result: **9 → 7 instructions**, 2 reduced, 22.22 %, 1 iteration, status `applied`
- `t6 = 100` and `t7 = t6 + 50` collapse into `t7 = 150`, and the unused statement is dropped

## Testing

```powershell
cd D:\PROJECTS\compiler
.\.venv\Scripts\Activate.ps1
pytest
```

Current result: **16 passed, 1 warning** (~0.4 s).

The suite covers all six optimizations, feature extraction, rule-based scoring, the iterative optimizer, and the API endpoints.

## Current limitations

- Strategy selection is **rule-based only** — there is no machine learning or deep learning model.
- Improvement is measured by **instruction count only**. A pass that rewrites a statement without removing one (for example constant folding `t6 = 5 + 5` → `t6 = 10`) reports no reduction.
- The parser covers a **subset of TAC**: assignments, unary minus, integer and float constants, comparisons, arithmetic (`+ - * /`), labels, `goto` and conditional `if … goto`. There is no dedicated handling for function calls, arrays or pointers.
- **Strength Reduction** is deliberately conservative: it only rewrites multiplication into cheaper operations when it has enough loop context, and otherwise leaves the code unchanged. **Algebraic Simplification** only applies a fixed list of identities (`x + 0`, `x * 1`, `x * 0`, and so on), so many algebraic rewrites are out of scope.
- The benchmark in `experiments/` is a generated corpus, not an external standard dataset.
- `examples/` is still empty.

## Future work

- Explore **machine learning based** strategy selection (train a classifier on the extracted features). *Not implemented today — the current selector is purely rule-based.*
- Explore deep learning approaches for the same task.
- Extend the TAC grammar (function calls, arrays, more control flow).
- Add more optimization passes and data-flow based analysis.
- Evaluate on a larger, externally recognised benchmark.
