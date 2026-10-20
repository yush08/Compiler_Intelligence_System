"""Theoretical and measured complexity of the two selection strategies.

Two kinds of numbers that are never mixed:

* **theoretical** - worst-case expressions whose parameters are read from the
  real code and the real fitted artifact: ``O(R x F)`` for the rule scorer
  (R = scoring rules in ``scoring.rule_based``, F = features) and
  ``O(T x D)`` for one forest prediction (T = trees, D = deepest fitted
  tree).
* **measured** - wall-clock latency of both selectors over the *same* parsed
  programs (repeated batches, median/min/max with the sample counts), plus
  artifact sizes and a ``tracemalloc`` footprint, each labelled with the
  method that produced it.

If a figure cannot be taken reliably the field reports ``available: false``
with a reason - no number is ever invented. The latency benchmark is a
repeated wall-clock observation inside one request on the machine running
the server; it is not an isolated benchmark harness and the report says so.
"""

from __future__ import annotations

import os
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter

from ..features.extractor import FEATURE_KEYS, extract_features
from ..scoring import rule_based as rule_based_module
from ..scoring.rule_based import select_strategy

__all__ = ["SCORING_RULES", "complexity_report"]

# Scoring rules in rule_based_strategy_scores(): one formula per strategy
# (folding, propagation, CSE, DCE, algebraic, strength). Pinned against the
# actual source by tests/test_complexity.py - if a rule is added, the test
# fails until this number is updated.
SCORING_RULES = 6

# Benchmark shape: batches of calls over the same programs; the median of the
# per-call batch means is the reported latency. Chosen so a full report stays
# well under a few seconds while giving 100+ timing samples per selector.
_REPETITIONS = 100
_PROGRAM_LIMIT = 30


def _theoretical_block(model_payload: dict | None, model_error: str | None) -> dict:
    """Both complexity expressions with parameters from code and artifact."""
    n_features = len(FEATURE_KEYS)
    theoretical: dict = {
        "rule_based": {
            "expression": "O(R x F)",
            "rules": SCORING_RULES,
            "features": n_features,
            "rule_feature_evaluations": SCORING_RULES * n_features,
            "rules_source": (
                "One scoring formula per strategy in "
                "backend/scoring/rule_based.py (counted in the source by tests)."
            ),
            "note": (
                "Each selection also runs extract_features once, which is "
                "linear in the program's statement count; the scoring stage "
                "itself evaluates R x F rule-feature pairs."
            ),
        },
    }

    if model_payload is None or model_payload.get("model") is None:
        theoretical["ml"] = {
            "available": False,
            "reason": model_error or "No trained model artifact is loaded.",
        }
        return theoretical

    model = model_payload["model"]
    estimators = list(getattr(model, "estimators_", []) or [])
    trees = int(getattr(model, "n_estimators", 0) or len(estimators))
    depths = [int(getattr(tree.tree_, "max_depth", 0) or 0) for tree in estimators]
    max_depth = max(depths, default=0)
    features_in = int(getattr(model, "n_features_in_", n_features) or n_features)

    theoretical["ml"] = {
        "available": True,
        "expression": "O(T x D) per prediction",
        "trees": trees,
        "max_depth": max_depth,
        "features": features_in,
        "tree_node_evaluations_upper_bound": trees * max_depth,
        "parameters_source": (
            "trees, deepest fitted tree and feature count read from the "
            "fitted artifact itself (artifacts/ml/model.joblib)."
        ),
        "note": (
            "A prediction walks at most D nodes in each of T trees; the bound "
            "uses the deepest fitted tree, so average cost is usually lower."
        ),
    }
    return theoretical


def _benchmark_programs(limit: int = _PROGRAM_LIMIT) -> list:
    """Parsed programs for the latency benchmark: the dataset when it exists,
    the project's own example TACs otherwise - always real programs, never
    hand-picked timings."""
    from ..tac.parser import (
        algebraic_simplification_example,
        cse_example,
        constant_folding_example,
        dead_code_example,
        mixed_example,
        parse_tac,
    )

    tacs: list = []
    try:
        from ..ml.dataset import TAC_COLUMN, load_dataset

        frame = load_dataset()
        for text in frame[TAC_COLUMN].head(limit):
            try:
                tac = parse_tac(str(text))
            except Exception:  # noqa: BLE001 - skip a row that will not parse
                continue
            if tac.instructions:
                tacs.append(tac)
    except Exception:  # noqa: BLE001 - dataset missing falls back to examples
        tacs = []

    if len(tacs) < 3:
        examples = [
            constant_folding_example(),
            dead_code_example(),
            cse_example(),
            algebraic_simplification_example(),
            mixed_example(),
        ]
        tacs = [tac for tac in examples if tac.instructions]
    return tacs


def _latency(fn, tacs: list) -> dict:
    """Median/min/max microseconds per call over repeated batches."""
    for tac in tacs:  # warm-up: first-call costs (imports, caches) excluded
        fn(tac)
    samples: list[float] = []
    for _ in range(_REPETITIONS):
        start = perf_counter()
        for tac in tacs:
            fn(tac)
        samples.append((perf_counter() - start) / len(tacs) * 1e6)
    return {
        "median_us": round(median(samples), 3),
        "min_us": round(min(samples), 3),
        "max_us": round(max(samples), 3),
        "batches": _REPETITIONS,
        "calls": _REPETITIONS * len(tacs),
    }


def _model_footprint() -> dict:
    """Fresh joblib.load under tracemalloc: the deserialized model's Python
    allocation footprint, measured - not estimated."""
    try:
        import joblib

        from . import model as model_mod

        tracemalloc.start()
        try:
            joblib.load(model_mod.MODEL_FILE)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return {
            "available": True,
            "method": "tracemalloc around one fresh joblib.load of the artifact",
            "current_bytes": int(current),
            "peak_bytes": int(peak),
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the endpoint
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: footprint not measured.",
        }


def complexity_report() -> dict:
    """Full report: theoretical expressions + measured latency/sizes."""
    from .model import load_payload
    from .predict import predict_from_features

    payload: dict | None = None
    model_error: str | None = None
    try:
        payload = load_payload()
    except Exception as exc:  # noqa: BLE001 - unavailable model is a reported state
        model_error = str(exc) or type(exc).__name__
        payload = None

    theoretical = _theoretical_block(payload, model_error)

    # --- measured latency: the same parsed programs for both selectors -----
    tacs = _benchmark_programs()
    latency: dict = {
        "available": False,
        "reason": None,
        "method": (
            f"perf_counter: {_REPETITIONS} batches of {len(tacs)} programs per "
            "selector on the same parsed programs; the median of per-call "
            "batch means is reported with min/max. Single process on the "
            "server machine, no isolation - a repeated observation, not an "
            "isolated benchmark harness. The ML path reloads the fitted "
            "artifact on every prediction (predict_from_features loads the "
            "payload itself; there is no in-process model cache), so the ML "
            "figures include that joblib load - they measure the selector as "
            "it is actually served, not the bare forest walk."
        ),
        "programs": len(tacs),
        "rule_select_strategy_us": None,
        "ml_predict_us": None,
        "ratio_rule_over_ml": None,
    }
    if len(tacs) < 3:
        latency["reason"] = "Fewer than three real programs were available to time."
    else:
        try:
            rule_stats = _latency(select_strategy, tacs)
            latency["rule_select_strategy_us"] = rule_stats
            latency["available"] = True
            if payload is not None and payload.get("model") is not None:

                def _ml_prediction(tac):
                    # Same first step as the rules (feature extraction), then
                    # the forest - measured end-to-end like the rule path.
                    # predict_from_features loads the fitted payload itself.
                    return predict_from_features(extract_features(tac))

                ml_stats = _latency(_ml_prediction, tacs)
                latency["ml_predict_us"] = ml_stats
                if ml_stats["median_us"] > 0:
                    latency["ratio_rule_over_ml"] = round(rule_stats["median_us"] / ml_stats["median_us"], 3)
            else:
                latency["reason"] = model_error or "Model unavailable: only rule-based latency was measured."
        except Exception as exc:  # noqa: BLE001 - honest failure, no partial guesses
            latency["available"] = False
            latency["reason"] = f"{type(exc).__name__}: {exc}"

    # --- measured sizes ----------------------------------------------------
    sizes: dict = {}
    try:
        from .model import MODEL_FILE

        sizes["model_artifact"] = {
            "file": Path(MODEL_FILE).name,
            "bytes": int(os.path.getsize(MODEL_FILE)),
        }
    except Exception as exc:  # noqa: BLE001
        sizes["model_artifact"] = {"available": False, "reason": f"{type(exc).__name__}: size not measured."}
    try:
        rule_file = Path(rule_based_module.__file__ or "")
        sizes["rule_scorer_source"] = {
            "file": rule_file.name,
            "bytes": int(os.path.getsize(rule_file)),
        }
    except Exception:  # noqa: BLE001
        sizes["rule_scorer_source"] = {"available": False, "reason": "Source size not measured."}
    sizes["model_deserialized"] = _model_footprint()

    return {
        "available": True,
        "theoretical": theoretical,
        "measured": {"latency": latency, "sizes": sizes},
        "limitations": [
            "Latency figures are wall-clock observations inside one request on "
            "the server machine (shared process, no isolation); they compare "
            "the two selectors reliably but will differ between machines and loads.",
            "The theoretical expressions are worst-case bounds with parameters "
            "from this exact artifact; the forest's average traversal is usually below T x D.",
            "Model-side figures report unavailable when no trained artifact is "
            "loaded - they are never estimated.",
        ],
    }
