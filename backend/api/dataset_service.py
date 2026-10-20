"""Read-only dataset API: summary, paginated samples and full sample detail.

Everything is served through the backend so the browser never touches the CSV
and no filesystem path is ever returned (§16).  The CSV is loaded once and
cached, invalidated when its mtime or size changes.

The detail view deliberately separates four different things that are easy to
conflate:

* ``ground_truth_label`` - the stored ``best_strategy``, plus the six measured
  ``useful_improvement`` values that produced it, so a reader can see *why* the
  sample carries that label.
* ``rule_based_selection`` - what the live rule-based scorer picks right now.
* ``ml_prediction`` - what the trained forest picks right now (or an explicit
  unavailable state).
* ``executed_strategy`` - ``null`` here: browsing the dataset executes nothing.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pandas as pd

from ..features.extractor import FEATURE_DESCRIPTIONS, FEATURE_KEYS
from ..ml.dataset import (
    DEFAULT_DATASET_PATH,
    ID_COLUMN,
    SPLIT_COLUMN,
    TAC_COLUMN,
    TARGET_COLUMN,
    STRATEGIES,
    load_dataset,
    validate_dataset,
)
from ..optimizer.measure import best_strategy_from

__all__ = ["dataset_summary", "dataset_samples", "dataset_sample", "refresh_cache"]

_CACHE: dict = {"key": None, "frame": None, "validated": None}
_LOCK = threading.Lock()

_TIE_NOTE = (
    "Ties are resolved by backend.scoring.rule_based.STRATEGY_TIE_ORDER, the "
    "same deterministic order the rule-based selector uses, so labelling is "
    "reproducible and never random."
)


def _load() -> pd.DataFrame:
    key = None
    try:
        stat = DEFAULT_DATASET_PATH.stat()
        key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None

    with _LOCK:
        if key is None:
            raise FileNotFoundError(
                "Dataset not found. Generate it with: python experiments/generate_dataset.py"
            )
        if _CACHE["frame"] is not None and _CACHE["key"] == key:
            return _CACHE["frame"]

        frame = load_dataset()
        _CACHE["key"] = key
        _CACHE["frame"] = frame
        _CACHE["validated"] = validate_dataset(frame)
        return frame


def refresh_cache() -> None:
    """Drop the cached frame (tests and post-generation refresh)."""
    with _LOCK:
        _CACHE.update(key=None, frame=None, validated=None)


def _feature_definitions(df: pd.DataFrame) -> list[dict]:
    definitions = []
    for key in FEATURE_KEYS:
        series = pd.to_numeric(df[key], errors="coerce")
        definitions.append(
            {
                "key": key,
                "description": FEATURE_DESCRIPTIONS.get(key, ""),
                "dtype": str(df[key].dtype),
                "missing": int(series.isna().sum()),
                "min": float(series.min()) if len(series) else 0.0,
                "max": float(series.max()) if len(series) else 0.0,
                "mean": round(float(series.mean()), 6) if len(series) else 0.0,
                "unique_values": int(series.nunique()),
            }
        )
    return definitions


def _histogram(series: pd.Series, bins: int = 10) -> list[dict]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []
    if numeric.nunique() == 1:
        value = float(numeric.iloc[0])
        return [{"from": value, "to": value, "count": int(len(numeric))}]

    counts, edges = pd.cut(numeric, bins=bins, retbins=True, include_lowest=True)
    frequencies = counts.value_counts(sort=False)
    return [
        {"from": round(float(edges[i]), 6), "to": round(float(edges[i + 1]), 6), "count": int(frequencies.iloc[i])}
        for i in range(len(frequencies))
    ]


def dataset_summary() -> dict:
    """Schema, distributions and quality summary for the Dataset Explorer."""
    try:
        df = _load()
    except FileNotFoundError as exc:
        return {"available": False, "reason": str(exc)}

    validated = _CACHE.get("validated") or validate_dataset(df)

    target_counts = {k: int(v) for k, v in df[TARGET_COLUMN].value_counts().items()}
    split_counts = {k: int(v) for k, v in df[SPLIT_COLUMN].value_counts().items()}

    categories = []
    for name, group in df.groupby("generation_category"):
        categories.append(
            {
                "name": str(name),
                "count": int(len(group)),
                # Per-split coverage: a category with zero test rows is never
                # evaluated directly, which is a property worth surfacing
                # rather than hiding behind the totals.
                "splits": {
                    k: int(v) for k, v in group[SPLIT_COLUMN].value_counts().items()
                },
                "labels": {k: int(v) for k, v in group[TARGET_COLUMN].value_counts().items()},
            }
        )
    categories.sort(key=lambda item: (-item["count"], item["name"]))

    return {
        "available": True,
        "dataset_file": DEFAULT_DATASET_PATH.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "validation": {
            "passed": validated.get("passed"),
            "errors": validated.get("errors", []),
            "warnings": validated.get("warnings", []),
            "fingerprint": validated.get("fingerprint"),
            "label_measurement_mismatch_count": validated.get("label_measurement_mismatch_count", 0),
        },
        "model_input_columns": list(FEATURE_KEYS),
        "features": _feature_definitions(df),
        "target": {
            "column": TARGET_COLUMN,
            "distribution": target_counts,
            "class_count": len(target_counts),
            "definition": (
                "The strategy with the largest useful_improvement = statement "
                "reduction + operation reduction, measured by running every "
                "pass on the program."
            ),
            "tie_break": _TIE_NOTE,
        },
        "splits": {
            "distribution": split_counts,
            "column": SPLIT_COLUMN,
            "note": (
                "Partition assigned by the generator and never reassigned, so "
                "the held-out test split stays held out. Splits are grouped "
                "by generation family: one family = one program template, and "
                "a whole family lives in exactly one split, so no template "
                "appears in both training and test data."
            ),
            "grouped_by": "family" if "family" in df.columns else None,
            "family_count": int(df["family"].nunique()) if "family" in df.columns else None,
        },
        "missing_values": {
            "total": validated.get("total_missing_values", 0),
            "by_column": validated.get("missing_values", {}),
        },
        "duplicates": {
            "duplicate_tac_programs": validated.get("duplicate_tac_programs", 0),
            "duplicate_sample_ids": validated.get("duplicate_sample_ids", 0),
        },
        "categories": categories,
        "strategies": list(STRATEGIES),
        "feature_histograms": {key: _histogram(df[key]) for key in FEATURE_KEYS},
        "label_measurement_mismatches": validated.get("label_measurement_mismatches", []),
    }


def dataset_samples(
    page: int = 1, size: int = 25, q: str = "", split: str | None = None
) -> dict:
    """Paginated, searchable sample list. Never returns the full CSV."""
    try:
        df = _load()
    except FileNotFoundError as exc:
        return {"available": False, "reason": str(exc), "items": [], "total": 0}

    page = max(1, int(page))
    size = max(1, min(int(size), 200))

    filtered = df
    if split in {"train", "validation", "test"}:
        filtered = filtered[filtered[SPLIT_COLUMN] == split]

    query = (q or "").strip().lower()
    if query:
        mask = (
            filtered[ID_COLUMN].astype(str).str.lower().str.contains(query, regex=False)
            | filtered["generation_category"].astype(str).str.lower().str.contains(query, regex=False)
            | filtered[TAC_COLUMN].astype(str).str.lower().str.contains(query, regex=False)
            | filtered[TARGET_COLUMN].astype(str).str.lower().str.contains(query, regex=False)
        )
        if "family" in filtered.columns:
            mask = mask | filtered["family"].astype(str).str.lower().str.contains(query, regex=False)
        filtered = filtered[mask]

    filtered = filtered.sort_values(ID_COLUMN, kind="mergesort")
    total = int(len(filtered))
    pages = max(1, -(-total // size))
    page = min(page, pages)
    start = (page - 1) * size
    window = filtered.iloc[start : start + size]

    items = []
    for _, row in window.iterrows():
        tac_text = str(row[TAC_COLUMN])
        lines = tac_text.splitlines()
        items.append(
            {
                "sample_id": row[ID_COLUMN],
                "generation_category": row.get("generation_category"),
                "family": row.get("family"),
                "split": row[SPLIT_COLUMN],
                "best_strategy": row[TARGET_COLUMN],
                # Stored measurement of that winner. It equals the largest
                # `<strategy>_useful_improvement` across the six passes for
                # every row (verified by the generator's validation), i.e.
                # the same quantity the detail endpoint reports as
                # `best_useful_improvement`.
                "best_reduction": _number(row.get("best_reduction")),
                "instruction_count": int(row.get("num_statements", 0)),
                "line_count": len(lines),
                "tac_preview": "\n".join(lines[:4]) + ("\n..." if len(lines) > 4 else ""),
            }
        )

    return {
        "available": True,
        "items": items,
        "page": page,
        "size": size,
        "pages": pages,
        "total": total,
        "query": q or "",
        "split_filter": split,
        "filtered_out": int(len(df)) - total,
    }


def _empty_detail(sample_id: str, reason: str) -> dict:
    return {"available": False, "reason": reason, "sample_id": sample_id}


def dataset_sample(sample_id: str) -> dict:
    """Full detail for one sample, including all six measured outcomes."""
    try:
        df = _load()
    except FileNotFoundError as exc:
        return _empty_detail(str(sample_id), str(exc))

    match = df[df[ID_COLUMN].astype(str) == str(sample_id)]
    if match.empty:
        return _empty_detail(
            str(sample_id), f"No sample with id {sample_id!r} exists in this dataset."
        )

    row = match.iloc[0]
    tac_text = str(row[TAC_COLUMN])

    features = {key: float(row[key]) for key in FEATURE_KEYS}

    measurements = {}
    for strategy in STRATEGIES:
        measurements[strategy] = {
            "optimized_instruction_count": _number(row.get(f"{strategy}_optimized_instruction_count")),
            "instruction_reduction": _number(row.get(f"{strategy}_instruction_reduction")),
            "reduction_percentage": _number(row.get(f"{strategy}_reduction_percentage")),
            "useful_improvement": _number(row.get(f"{strategy}_useful_improvement")),
        }

    derived = None
    label_matches = None
    if all(measurements[s]["useful_improvement"] is not None for s in STRATEGIES):
        try:
            derived = best_strategy_from(
                {s: {"useful_improvement": int(measurements[s]["useful_improvement"])} for s in STRATEGIES},
                _tie_order(),
            )
            label_matches = derived == row[TARGET_COLUMN]
        except ValueError:
            derived = None

    # Live selectors, computed fresh - never read from the stored label.
    rule_selection = None
    rule_error = None
    try:
        from ..scoring.rule_based import select_strategy
        from ..tac.parser import parse_tac

        parsed = parse_tac(tac_text)
        rule_selection = select_strategy(parsed) if parsed.instructions else None
    except Exception as exc:  # noqa: BLE001 - report, do not fail the request
        rule_error = f"{type(exc).__name__}: {exc}"

    from ..ml.predict import predict_from_tac

    try:
        parsed = parse_tac(tac_text)
        ml_prediction = predict_from_tac(parsed)
    except Exception as exc:  # noqa: BLE001
        ml_prediction = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    useful = {s: measurements[s]["useful_improvement"] for s in STRATEGIES}
    present = {k: v for k, v in useful.items() if v is not None}
    best_value = max(present.values()) if present else None

    return {
        "available": True,
        "sample_id": row[ID_COLUMN],
        "generation_category": row.get("generation_category"),
        "family": row.get("family"),
        "split": row[SPLIT_COLUMN],
        "instruction_count": int(row.get("num_statements", 0)),
        "tac_lines": tac_text.splitlines(),
        "tac_text": tac_text,
        "features": features,
        "measurements": measurements,
        "useful_improvement_by_strategy": useful,
        "best_useful_improvement": best_value,
        "strategies_tied_at_best": sorted(
            s for s, v in present.items() if best_value is not None and v == best_value
        ),
        "ground_truth": {
            "label": row[TARGET_COLUMN],
            "definition": (
                "Strategy with the largest useful_improvement (statement "
                "reduction + operation reduction) across all six passes."
            ),
            "derived_label": derived,
            "matches_stored_measurements": label_matches,
            "tie_break": _TIE_NOTE,
        },
        "rule_based_selection": {
            "strategy": rule_selection,
            "error": rule_error,
            "note": "Computed live by backend.scoring.rule_based.select_strategy, not read from the label.",
        },
        "ml_prediction": {
            **ml_prediction,
            "note": "Computed live by the trained forest, not read from the label.",
        },
        "executed_strategy": None,
        "executed_strategy_note": (
            "Browsing the dataset executes nothing. Run the Optimizer to "
            "execute a strategy and record a measured outcome."
        ),
        "raw_columns": {
            k: (None if pd.isna(v) else _scalar(v))
            for k, v in row.items()
            if k not in {TAC_COLUMN} and not str(k).endswith("_tac")
        },
    }


def _tie_order() -> list[str]:
    from ..scoring.rule_based import STRATEGY_TIE_ORDER

    return list(STRATEGY_TIE_ORDER)


def _number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return float(value) if isinstance(value, float) else _scalar(value)


def _scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)
