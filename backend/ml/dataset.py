"""Dataset loading, validation and the feature/label contract.

The dataset is generated once by ``experiments/generate_dataset.py`` and is
never regenerated here. This module's job is to read it back, prove it is still
in the shape the training code expects, and hand over exactly the eight
pre-optimization features as model inputs.

Leakage guard
-------------
Only the eight ``FEATURE_KEYS`` may reach the model. The CSV also contains the
ground-truth label, every strategy's measured outcome, the optimized TAC, sample
ids and the split assignment - all of which would leak the answer. The column
sets are declared here and :func:`feature_matrix` refuses to build a matrix from
anything else, so a leakage regression fails loudly instead of quietly inflating
the score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..features.extractor import FEATURE_KEYS
from ..optimizer.measure import best_strategy_from
from ..scoring.rule_based import STRATEGY_TIE_ORDER
from ..optimizer.engine import OPTIMIZATION_FUNCTIONS

# Re-exported so every consumer agrees on the ordering.
__all__ = [
    "FEATURE_KEYS",
    "TARGET_COLUMN",
    "SPLIT_COLUMN",
    "ID_COLUMN",
    "TAC_COLUMN",
    "STRATEGIES",
    "LEAKY_COLUMNS",
    "REPOSITORY_ROOT",
    "DEFAULT_DATASET_PATH",
    "load_dataset",
    "validate_dataset",
    "split_frames",
    "discover_target_classes",
    "feature_matrix",
    "feature_frame",
    "dataset_fingerprint",
    "label_measurement_column",
]

TARGET_COLUMN = "best_strategy"
SPLIT_COLUMN = "split"
ID_COLUMN = "sample_id"
TAC_COLUMN = "tac_program"

STRATEGIES = list(OPTIMIZATION_FUNCTIONS.keys())

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPOSITORY_ROOT / "dataset" / "processed" / "optimization_dataset.csv"

# Columns that must never be model inputs. Anything not in FEATURE_KEYS is
# rejected, but these are named explicitly because they are the ones that would
# leak the answer rather than merely being irrelevant.
LEAKY_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        SPLIT_COLUMN,
        ID_COLUMN,
        TAC_COLUMN,
        "generation_category",
        # The generation family identifies the program TEMPLATE, which is
        # exactly the provenance information a grouped split is built from;
        # it must never become a model input.
        "family",
        "source",
        "best_reduction",
        "best_instruction_reduction",
        "best_reduction_percentage",
        "has_opportunity",
        "parsed_instruction_count",
        "original_instruction_count",
        "requested_size",
    }
)


def label_measurement_column(strategy: str) -> str:
    """Column holding a strategy's measured ``useful_improvement``."""
    return f"{strategy}_useful_improvement"


def feature_frame(features: dict) -> pd.DataFrame:
    """Build a one-row DataFrame in canonical feature order for inference.

    The forest was fitted on a DataFrame, so predicting with a bare ndarray
    makes sklearn emit a "does not have valid feature names" warning and skips
    its own column-order check.  Passing a labelled frame keeps inference
    silent *and* lets sklearn verify the ordering itself.
    """
    missing = [key for key in FEATURE_KEYS if key not in features]
    if missing:
        raise KeyError("Missing feature value(s): " + ", ".join(missing))
    return pd.DataFrame(
        [[float(features[key]) for key in FEATURE_KEYS]], columns=list(FEATURE_KEYS)
    )


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset CSV as a DataFrame.

    Raises FileNotFoundError with the expected location so a missing dataset
    produces an actionable message rather than a stack trace.
    """
    resolved = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {resolved.name}. "
            "Generate it first with: python experiments/generate_dataset.py"
        )
    return pd.read_csv(resolved)


def discover_target_classes(df: pd.DataFrame) -> list[str]:
    """Target vocabulary discovered from the data - never hard-coded.

    ``strength_reduction`` has no winning examples, so a hard-coded six-class
    list would invent a class the model can never learn. Ordering follows
    ``STRATEGY_TIE_ORDER`` so it is stable across runs and matches the
    deterministic tie-break used to produce the labels.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"dataset is missing the {TARGET_COLUMN!r} column")
    present = set(df[TARGET_COLUMN].unique())
    ordered = [name for name in STRATEGY_TIE_ORDER if name in present]
    # Anything unexpected keeps its sorted position rather than being dropped.
    ordered += sorted(present - set(ordered))
    return ordered


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the dataset's own train/validation/test partitions.

    Samples are never moved between splits: the partition was assigned by the
    generator and must stay stable or the held-out test result stops meaning
    anything.
    """
    if SPLIT_COLUMN not in df.columns:
        raise KeyError(f"dataset is missing the {SPLIT_COLUMN!r} column")
    frames = {name: df[df[SPLIT_COLUMN] == name].reset_index(drop=True) for name in ("train", "validation", "test")}
    missing = [name for name, frame in frames.items() if frame.empty]
    if missing:
        raise ValueError(f"dataset has no rows for split(s): {', '.join(missing)}")
    return frames


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``(X, y)`` restricted to exactly the eight pre-optimization features.

    Raises if any feature column is missing or if a caller tries to pass a
    non-feature column - the leakage guard.
    """
    missing = [key for key in FEATURE_KEYS if key not in df.columns]
    if missing:
        raise KeyError(f"dataset is missing feature column(s): {', '.join(missing)}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"dataset is missing the {TARGET_COLUMN!r} column")

    leaked = sorted(set(df.columns) - set(FEATURE_KEYS))
    # `X` is built from an explicit column list, so extra columns cannot leak in
    # silently; this assertion documents the invariant and fails loudly if the
    # FEATURE_KEYS list itself is ever widened by mistake.
    assert not (set(FEATURE_KEYS) & LEAKY_COLUMNS), "FEATURE_KEYS must not contain a leaking column"

    X = df.loc[:, FEATURE_KEYS].copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    if X.isna().any().any():
        raise ValueError("feature matrix contains missing values")
    if not pd.api.types.is_numeric_dtype(X["const_ratio"]):
        raise TypeError("const_ratio must be numeric")

    return X, y


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable content hash so an artifact can prove which data trained it."""
    payload = json.dumps(
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "target_counts": df[TARGET_COLUMN].value_counts().sort_index().to_dict()
            if TARGET_COLUMN in df.columns
            else {},
            "ids": df[ID_COLUMN].tolist() if ID_COLUMN in df.columns else [],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run the §5 data validation checks and return a structured report.

    The report distinguishes ``errors`` (training must not proceed) from
    ``warnings`` (documented, expected characteristics of this dataset).
    """
    errors: list[str] = []
    warnings: list[str] = []

    report: dict = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "errors": errors,
        "warnings": warnings,
    }

    # --- presence -----------------------------------------------------------
    for required in (*FEATURE_KEYS, TARGET_COLUMN, SPLIT_COLUMN, ID_COLUMN, TAC_COLUMN):
        if required not in df.columns:
            errors.append(f"missing required column: {required}")
    if errors:
        report["passed"] = False
        return report

    # --- missing values -----------------------------------------------------
    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    report["missing_values"] = {col: n for col, n in null_counts.items() if n}
    report["total_missing_values"] = int(sum(null_counts.values()))
    if report["total_missing_values"]:
        errors.append(f"dataset contains {report['total_missing_values']} missing value(s)")

    # --- duplicates ---------------------------------------------------------
    report["duplicate_tac_programs"] = int(df[TAC_COLUMN].duplicated().sum())
    report["duplicate_sample_ids"] = int(df[ID_COLUMN].duplicated().sum())
    if report["duplicate_tac_programs"]:
        errors.append(f"{report['duplicate_tac_programs']} duplicate TAC program(s)")
    if report["duplicate_sample_ids"]:
        errors.append(f"{report['duplicate_sample_ids']} duplicate sample id(s)")

    # --- feature types / ranges -------------------------------------------
    feature_dtypes = {key: str(df[key].dtype) for key in FEATURE_KEYS}
    report["feature_dtypes"] = feature_dtypes
    non_numeric = [k for k, v in feature_dtypes.items() if not v.startswith(("int", "float", "uint"))]
    if non_numeric:
        errors.append(f"non-numeric feature column(s): {', '.join(non_numeric)}")

    ranges = {}
    for key in FEATURE_KEYS:
        values = pd.to_numeric(df[key], errors="coerce")
        if values.isna().any():
            errors.append(f"feature {key} contains non-numeric values")
            continue
        ranges[key] = {"min": float(values.min()), "max": float(values.max())}
        if values.min() < 0:
            errors.append(f"feature {key} has a negative value ({values.min()})")
    report["feature_ranges"] = ranges

    ratio = pd.to_numeric(df["const_ratio"], errors="coerce")
    if not ratio.isna().any() and (ratio.min() < 0 or ratio.max() > 1):
        errors.append("const_ratio must lie in [0, 1]")

    # --- target ------------------------------------------------------------
    known = set(STRATEGIES)
    present = sorted(set(df[TARGET_COLUMN].unique()))
    invalid = sorted(set(present) - known)
    report["target_classes"] = present
    report["target_counts"] = {k: int(v) for k, v in df[TARGET_COLUMN].value_counts().items()}
    if invalid:
        errors.append(f"invalid strategy identifier(s) in {TARGET_COLUMN}: {', '.join(invalid)}")

    absent = [name for name in STRATEGIES if name not in present]
    if absent:
        warnings.append(
            "strategy/class absent from the label set: "
            + ", ".join(absent)
            + " (never predicted - no winning examples)"
        )

    # --- splits ------------------------------------------------------------
    split_counts = {k: int(v) for k, v in df[SPLIT_COLUMN].value_counts().items()}
    report["split_counts"] = split_counts
    unknown_splits = sorted(set(split_counts) - {"train", "validation", "test"})
    if unknown_splits:
        errors.append(f"unknown split value(s): {', '.join(unknown_splits)}")
    if len(split_counts) < 3:
        errors.append("dataset does not have all three splits (train/validation/test)")

    # Overlap between splits would corrupt the held-out evaluation.
    id_sets = {
        name: set(df.loc[df[SPLIT_COLUMN] == name, ID_COLUMN])
        for name in ("train", "validation", "test")
    }
    overlap = []
    for a in id_sets:
        for b in id_sets:
            if a < b and (id_sets[a] & id_sets[b]):
                overlap.append(f"{a}/{b}: {len(id_sets[a] & id_sets[b])}")
    report["split_overlap"] = overlap
    if overlap:
        errors.append("sample ids appear in more than one split: " + "; ".join(overlap))

    # --- label vs stored measurements --------------------------------------
    mismatches = []
    if all(label_measurement_column(s) in df.columns for s in STRATEGIES):
        for _, row in df.iterrows():
            measurements = {
                name: {"useful_improvement": int(row[label_measurement_column(name)])}
                for name in STRATEGIES
            }
            try:
                derived = best_strategy_from(measurements, STRATEGY_TIE_ORDER)
            except ValueError as exc:  # pragma: no cover - defensive
                mismatches.append(f"{row[ID_COLUMN]}: {exc}")
                continue
            if derived != row[TARGET_COLUMN]:
                mismatches.append(f"{row[ID_COLUMN]}: label={row[TARGET_COLUMN]} derived={derived}")
    else:
        warnings.append("per-strategy measurement columns absent; label consistency not checked")

    report["label_measurement_mismatches"] = mismatches[:20]
    report["label_measurement_mismatch_count"] = len(mismatches)
    if mismatches:
        errors.append(
            f"{len(mismatches)} label(s) disagree with the stored useful_improvement measurements"
        )

    # --- leakage -----------------------------------------------------------
    used_leaky = sorted(set(FEATURE_KEYS) & LEAKY_COLUMNS)
    report["model_input_columns"] = list(FEATURE_KEYS)
    report["leakage_columns_excluded"] = sorted(LEAKY_COLUMNS - set(FEATURE_KEYS))
    if used_leaky:
        errors.append(f"model input column(s) would leak the target: {', '.join(used_leaky)}")

    # --- distribution / imbalance -----------------------------------------
    counts = df[TARGET_COLUMN].value_counts()
    if len(counts) > 1:
        ratio_imbalance = float(counts.max() / counts.min())
        report["class_imbalance_ratio"] = round(ratio_imbalance, 3)
        if ratio_imbalance >= 3:
            warnings.append(
                f"classes are imbalanced (largest/smallest = {ratio_imbalance:.1f}); "
                "macro-F1 is the metric to trust"
            )

    small = [name for name, n in counts.items() if n < 5]
    if small:
        warnings.append(
            "class(es) with fewer than 5 samples: "
            + ", ".join(sorted(small))
            + " - per-class metrics on the test split will be unreliable"
        )

    # held-out split small => honest caveat
    n_test = int(split_counts.get("test", 0))
    report["test_samples"] = n_test
    if 0 < n_test < 40:
        warnings.append(
            f"only {n_test} held-out test samples; all test metrics are preliminary"
        )

    report["fingerprint"] = dataset_fingerprint(df)
    report["passed"] = not errors
    return report
