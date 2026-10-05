"""
Run the real backend over the benchmark corpus and record measurements.

Everything written to report/data/ is produced by executing the actual project
code (backend.features, backend.scoring, backend.optimizations,
backend.optimizer.engine). No result is hand-written.

Measures per program
--------------------
* the 8 extracted features
* the 6 rule-based strategy scores and the selected strategy
* each of the 6 passes applied once: does it rewrite the TAC, and does it
  change the instruction count (the engine's own improvement metric)
* the adaptive iterative engine: total reduction, iterations, convergence
* wall-clock cost of the analysis step and the optimisation step

Also measures end-to-end HTTP latency through the Vite dev-server proxy, which
doubles as a check that /api/analyze and /api/optimize are routed correctly.

Usage:
    python experiments/run_experiments.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import build_corpus  # noqa: E402

from backend.api.serialize import tac_to_lines  # noqa: E402
from backend.features.extractor import extract_features  # noqa: E402
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization  # noqa: E402
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy  # noqa: E402
from backend.tac.parser import parse_tac  # noqa: E402

FEATURE_KEYS = [
    "num_statements",
    "num_constants",
    "num_repeated_expr",
    "num_dead_stmts",
    "num_algebraic_ops",
    "num_loop_ops",
    "num_tmp_vars",
    "const_ratio",
]

STRATEGIES = list(OPTIMIZATION_FUNCTIONS.keys())

# Deterministic tie-break, mirroring backend.scoring.rule_based.select_strategy
TIE_ORDER = [
    "constant_folding",
    "constant_propagation",
    "common_subexpression",
    "dead_code_elimination",
    "algebraic_simplification",
    "strength_reduction",
]

API_BASE = "http://localhost:5173"  # Vite dev server -> proxies /api to FastAPI


# --- helpers -----------------------------------------------------------------


def apply_pass(name: str, tac):
    """Apply one pass to an independent deep copy of `tac`.

    Returns (text, before_count, after_count). A deep copy keeps each pass
    isolated so the measurements of one pass cannot contaminate another, while
    still exercising the real pass implementation.
    """
    from copy import deepcopy

    before = len(tac)
    after_t = OPTIMIZATION_FUNCTIONS[name](deepcopy(tac))
    return "\n".join(tac_to_lines(after_t)), before, len(after_t)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def label_for(pass_reductions: dict[str, int]) -> str:
    """Oracle label: the pass with the largest count reduction, TIE_ORDER tie-break."""
    best = max(pass_reductions.values())
    if best <= 0:
        # Nothing reduces the count; fall back to the deterministic order so the
        # confusion matrix still has a well-defined ground truth.
        return TIE_ORDER[0]
    for name in TIE_ORDER:
        if pass_reductions.get(name, 0) == best:
            return name
    return TIE_ORDER[0]


def prf(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[lab] = {
            "support": sum(1 for t in y_true if t == lab),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return out


# --- main --------------------------------------------------------------------


def run() -> dict:
    corpus = build_corpus()
    rows = []

    for rec in corpus:
        source = rec["tac"]
        tac = parse_tac(source)
        original_count = len(tac)

        # --- analysis step -------------------------------------------------
        t0 = time.perf_counter()
        features = extract_features(tac)
        scores = rule_based_strategy_scores(tac)
        selected = select_strategy(tac)
        analyze_ms = (time.perf_counter() - t0) * 1000

        # --- single application of every pass ------------------------------
        per_pass = {}
        for name in STRATEGIES:
            out_text, before, after = apply_pass(name, tac)
            before_text = "\n".join(tac_to_lines(tac))
            per_pass[name] = {
                "rewrote_tac": out_text != before_text,
                "before": before,
                "after": after,
                "reduction": before - after,
            }

        pass_reductions = {k: v["reduction"] for k, v in per_pass.items()}
        best_reduction = max(pass_reductions.values())
        oracle = label_for(pass_reductions)

        # --- adaptive iterative engine --------------------------------------
        t0 = time.perf_counter()
        engine = run_iterative_optimization(tac, max_iterations=20)
        optimize_ms = (time.perf_counter() - t0) * 1000

        selected_reduction = pass_reductions[selected]

        rows.append(
            {
                "id": rec["id"],
                "source": rec["source"],
                "opportunity_class": rec["opportunity_class"],
                "size": rec["size"],
                "statements_in": original_count,
                "features": {k: features[k] for k in FEATURE_KEYS},
                "scores": {k: round(float(scores[k]), 4) for k in STRATEGIES},
                "selected": selected,
                "oracle": oracle,
                "per_pass": per_pass,
                "best_single_pass_reduction": best_reduction,
                "selected_single_pass_reduction": selected_reduction,
                "achieved_best": selected_reduction == best_reduction and best_reduction > 0,
                "any_pass_reduces": best_reduction > 0,
                "adaptive_reduction": engine["total_reduction"],
                "adaptive_iterations": engine["total_iterations"],
                "adaptive_converged": engine["converged"],
                "analyze_ms": round(analyze_ms, 4),
                "optimize_ms": round(optimize_ms, 4),
            }
        )

    # --- aggregates --------------------------------------------------------
    n = len(rows)
    agg: dict = {"n_programs": n}

    agg["reduction"] = {
        "adaptive_mean_pct": round(
            statistics.mean(
                (r["adaptive_reduction"] / r["statements_in"] * 100)
                for r in rows
                if r["statements_in"]
            ),
            3,
        ),
        "adaptive_mean_abs": round(statistics.mean(r["adaptive_reduction"] for r in rows), 3),
        "selected_single_mean_abs": round(
            statistics.mean(r["selected_single_pass_reduction"] for r in rows), 3
        ),
        "best_single_mean_abs": round(
            statistics.mean(r["best_single_pass_reduction"] for r in rows), 3
        ),
        "programs_with_any_reduction": sum(1 for r in rows if r["adaptive_reduction"] > 0),
        "pct_programs_reduced": round(
            100 * sum(1 for r in rows if r["adaptive_reduction"] > 0) / n, 2
        ),
        "pct_programs_any_pass_reduces": round(
            100 * sum(1 for r in rows if r["any_pass_reduces"]) / n, 2
        ),
    }

    agg["selection"] = {
        "achieved_best_rate": round(100 * sum(1 for r in rows if r["achieved_best"]) / n, 2),
        "accuracy_vs_oracle": round(
            100 * sum(1 for r in rows if r["selected"] == r["oracle"]) / n, 2
        ),
        "confusion": {
            p: {
                o: sum(1 for r in rows if r["selected"] == p and r["oracle"] == o)
                for o in TIE_ORDER
            }
            for p in TIE_ORDER
        },
        "per_strategy_prf": prf(
            [r["oracle"] for r in rows], [r["selected"] for r in rows], TIE_ORDER
        ),
        "times_selected": {
            s: sum(1 for r in rows if r["selected"] == s) for s in TIE_ORDER
        },
        "times_oracle": {s: sum(1 for r in rows if r["oracle"] == s) for s in TIE_ORDER},
    }

    agg["passes"] = {
        s: {
            "rewrote_pct": round(
                100 * sum(1 for r in rows if r["per_pass"][s]["rewrote_tac"]) / n, 2
            ),
            "reduced_count_pct": round(
                100 * sum(1 for r in rows if r["per_pass"][s]["reduction"] > 0) / n, 2
            ),
            "mean_count_delta": round(
                statistics.mean(r["per_pass"][s]["reduction"] for r in rows), 3
            ),
            "max_count_delta": max(r["per_pass"][s]["reduction"] for r in rows),
        }
        for s in STRATEGIES
    }

    agg["by_class"] = {}
    classes = sorted({r["opportunity_class"] for r in rows})
    for cls in classes:
        sub = [r for r in rows if r["opportunity_class"] == cls]
        agg["by_class"][cls] = {
            "n": len(sub),
            "mean_statements": round(statistics.mean(r["statements_in"] for r in sub), 1),
            "mean_adaptive_reduction": round(
                statistics.mean(r["adaptive_reduction"] for r in sub), 3
            ),
            "mean_best_single_reduction": round(
                statistics.mean(r["best_single_pass_reduction"] for r in sub), 3
            ),
            "achieved_best_pct": round(
                100 * sum(1 for r in sub if r["achieved_best"]) / len(sub), 2
            ),
            "pct_any_reduction": round(
                100 * sum(1 for r in sub if r["adaptive_reduction"] > 0) / len(sub), 2
            ),
            "mean_iterations": round(
                statistics.mean(r["adaptive_iterations"] for r in sub), 2
            ),
        }

    generated = [r for r in rows if r["source"] == "generated"]
    agg["by_size"] = {}
    for size in sorted({r["size"] for r in generated}):
        sub = [r for r in generated if r["size"] == size]
        agg["by_size"][str(size)] = {
            "n": len(sub),
            "mean_adaptive_reduction": round(
                statistics.mean(r["adaptive_reduction"] for r in sub), 3
            ),
            "mean_analyze_ms": round(statistics.mean(r["analyze_ms"] for r in sub), 4),
            "mean_optimize_ms": round(statistics.mean(r["optimize_ms"] for r in sub), 4),
        }

    analyze_times = [r["analyze_ms"] for r in rows]
    optimize_times = [r["optimize_ms"] for r in rows]
    agg["latency"] = {
        "analyze_ms": {
            "mean": round(statistics.mean(analyze_times), 4),
            "median": round(statistics.median(analyze_times), 4),
            "p95": round(percentile(analyze_times, 0.95), 4),
            "max": round(max(analyze_times), 4),
        },
        "optimize_ms": {
            "mean": round(statistics.mean(optimize_times), 4),
            "median": round(statistics.median(optimize_times), 4),
            "p95": round(percentile(optimize_times, 0.95), 4),
            "max": round(max(optimize_times), 4),
        },
    }

    agg["engine_behaviour"] = {
        "mean_iterations": round(statistics.mean(r["adaptive_iterations"] for r in rows), 3),
        "max_iterations": max(r["adaptive_iterations"] for r in rows),
        "converged_pct": round(
            100 * sum(1 for r in rows if r["adaptive_converged"]) / n, 2
        ),
        "stopped_after_first_iteration_pct": round(
            100 * sum(1 for r in rows if r["adaptive_iterations"] == 1) / n, 2
        ),
    }

    agg["api_latency"] = measure_api_latency()

    return {"summary": agg, "rows": rows}


def measure_api_latency(requests: int = 25) -> dict:
    """End-to-end HTTP latency through the Vite proxy. Also verifies routing."""
    import urllib.error
    import urllib.request

    # Deliberately NOT the canonical default (that input is special-cased by
    # backend/api/pipeline.py and would time the canned branch, not the engine).
    tac = (
        "k1 = 42\nk2 = k1 + 8\nk3 = input_a + input_b\nk4 = input_a + input_b\n"
        "k5 = k3 * 1\nunused_k = 77\nk6 = k4 + k1"
    )
    payload = json.dumps({"tac": tac}).encode()
    result: dict = {"base_url": API_BASE, "requests_per_endpoint": requests}

    for endpoint in ("/api/analyze", "/api/optimize"):
        times, statuses = [], []
        for _ in range(requests):
            req = urllib.request.Request(
                API_BASE + endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                    statuses.append(resp.status)
            except urllib.error.HTTPError as exc:
                statuses.append(exc.code)
            except Exception as exc:  # pragma: no cover - reported, not hidden
                statuses.append(f"ERR:{type(exc).__name__}")
            times.append((time.perf_counter() - t0) * 1000)

        ok = sum(1 for s in statuses if s == 200)
        result[endpoint] = {
            "status_200": f"{ok}/{requests}",
            "mean_ms": round(statistics.mean(times), 3),
            "median_ms": round(statistics.median(times), 3),
            "p95_ms": round(percentile(times, 0.95), 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
        }
    return result


def main() -> int:
    data = run()

    out_dir = ROOT / "report" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )

    # Flat per-program CSV for external analysis / plotting.
    cols = [
        "id", "source", "opportunity_class", "size", "statements_in",
        "selected", "oracle", "achieved_best", "any_pass_reduces",
        "best_single_pass_reduction", "selected_single_pass_reduction",
        "adaptive_reduction", "adaptive_iterations", "adaptive_converged",
        "analyze_ms", "optimize_ms",
    ] + [f"feat_{k}" for k in FEATURE_KEYS] + [f"score_{s}" for s in STRATEGIES]
    lines = [",".join(cols)]
    for r in data["rows"]:
        flat = [
            r["id"], r["source"], r["opportunity_class"], r["size"], r["statements_in"],
            r["selected"], r["oracle"], r["achieved_best"], r["any_pass_reduces"],
            r["best_single_pass_reduction"], r["selected_single_pass_reduction"],
            r["adaptive_reduction"], r["adaptive_iterations"], r["adaptive_converged"],
            r["analyze_ms"], r["optimize_ms"],
        ]
        flat += [r["features"][k] for k in FEATURE_KEYS]
        flat += [r["scores"][s] for s in STRATEGIES]
        lines.append(",".join(str(v) for v in flat))
    (out_dir / "per_program.csv").write_text("\n".join(lines), encoding="utf-8")

    s = data["summary"]
    print(f"programs evaluated : {s['n_programs']}")
    print(f"reduction summary  : {json.dumps(s['reduction'], indent=2)}")
    print(f"selection          : achieved_best={s['selection']['achieved_best_rate']}%  "
          f"accuracy_vs_oracle={s['selection']['accuracy_vs_oracle']}%")
    print(f"pass effectiveness :")
    for k, v in s["passes"].items():
        print(f"    {k:26} rewrote={v['rewrote_pct']:6.2f}%  reduced_count={v['reduced_count_pct']:6.2f}%")
    print(f"latency            : {json.dumps(s['latency'], indent=2)}")
    print(f"api latency        : {json.dumps(s['api_latency'], indent=2)}")
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
