"""Contract tests over the shared dataset as the ML pipeline consumes it.

These assertions pin down the properties the whole approach depends on and that
would silently corrupt every later number if they broke:

* the splits are **disjoint and correctly sized**;
* the stored features are exactly what the extractor produces from the stored
  TAC - the dataset is a cache of real computation, never a second source of
  truth;
* the held-out test split never contains a label or a generation category the
  model has not seen during training (which would make an honest score
  impossible to interpret);
* nothing in the dataset is duplicated.
"""

from backend.features.extractor import FEATURE_KEYS, extract_features
from backend.ml import dataset as ds
from backend.tac.parser import parse_tac


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_sizes_match_the_published_dataset():
    frames = ds.split_frames(ds.load_dataset())
    sizes = {name: len(frame) for name, frame in frames.items()}
    assert sizes == {
        "train": 151,
        "validation": 45,
        "test": 45,
    }
    assert sum(sizes.values()) == 241


def test_no_sample_appears_in_two_splits():
    frames = ds.split_frames(ds.load_dataset())
    ids = {name: set(frame[ds.ID_COLUMN].astype(str)) for name, frame in frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids[left] & ids[right]
        assert not overlap, f"{left} and {right} share {sorted(overlap)}"


def test_every_tac_program_is_unique():
    df = ds.load_dataset()
    duplicates = df[df[ds.TAC_COLUMN].duplicated(keep=False)]
    assert duplicates.empty, sorted(duplicates[ds.ID_COLUMN].astype(str))[:10]


# ---------------------------------------------------------------------------
# The stored features are a cache of real extraction
# ---------------------------------------------------------------------------


def test_stored_features_are_exactly_what_the_extractor_produces():
    """Re-extracting each TAC must reproduce every stored feature value.

    This is the anti-fabrication property: a training row's inputs come from
    the same code path the API uses at inference time, so the model can never
    have learned from values the running system would not produce.
    """
    df = ds.load_dataset()
    mismatches: list[tuple[str, str, float, float]] = []

    for _, row in df.iterrows():
        fresh = extract_features(parse_tac(row[ds.TAC_COLUMN]))
        for key in FEATURE_KEYS:
            stored = float(row[key])
            produced = float(fresh[key])
            if abs(stored - produced) > 1e-9:
                mismatches.append((str(row[ds.ID_COLUMN]), key, produced, stored))

    assert not mismatches, mismatches[:10]
    assert len(df) * len(FEATURE_KEYS) == 241 * 8


def test_features_are_derived_from_the_tac_only():
    """No stored feature may be copied from a post-optimization measurement."""
    df = ds.load_dataset()

    # Columns that describe what an optimizer *did* rather than what the
    # program *is* must not be readable as features.
    forbidden = {
        "best_reduction",
        "best_instruction_reduction",
        "best_reduction_percentage",
        "has_opportunity",
        "parsed_instruction_count",
        "original_instruction_count",
    }
    assert not (set(FEATURE_KEYS) & forbidden)

    # Every feature column is genuinely numeric and finite.
    for key in FEATURE_KEYS:
        values = df[key].astype(float)
        assert values.notna().all(), key
        assert values.map(lambda v: v == v).all(), f"{key} contains NaN"


# ---------------------------------------------------------------------------
# The held-out split is comparable to the training data
# ---------------------------------------------------------------------------


def test_grouped_split_keeps_whole_families_apart():
    frames = ds.split_frames(ds.load_dataset())
    train_labels = set(frames["train"][ds.TARGET_COLUMN])
    test_labels = set(frames["test"][ds.TARGET_COLUMN])
    validation_labels = set(frames["validation"][ds.TARGET_COLUMN])

    # Label coverage: stratification is done at family level, so every class
    # still reaches every split (test rows never show a label train missed).
    assert test_labels <= train_labels, sorted(test_labels - train_labels)
    assert validation_labels <= train_labels, sorted(validation_labels - train_labels)
    assert test_labels == validation_labels == train_labels, (
        "all target labels must appear in all three splits"
    )

    # The grouped guarantee itself: one generation family (one program
    # template) lives in exactly ONE split. The test split therefore holds
    # families that are absent from training - the deliberate opposite of the
    # old random split, where every test category also appeared in training.
    train_families = set(frames["train"]["family"])
    val_families = set(frames["validation"]["family"])
    test_families = set(frames["test"]["family"])
    assert train_families, "train split must have families"
    assert not (train_families & val_families), sorted(train_families & val_families)
    assert not (train_families & test_families), sorted(train_families & test_families)
    assert not (val_families & test_families), sorted(val_families & test_families)

    # Ratios stay near the documented 70/15/15 target (families move as
    # blocks, so "near" and not "exactly").
    n = len(frames["train"]) + len(frames["validation"]) + len(frames["test"])
    assert len(frames["train"]) / n >= 0.60
    assert len(frames["test"]) / n >= 0.12


def test_split_ships_its_pre_registered_method():
    import json
    from pathlib import Path

    path = Path(ds.DEFAULT_DATASET_PATH).parent.parent / "metadata" / "splits.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["grouped_by"] == "family"
    assert payload["method"], "the split method must travel with the data"
    assert payload["families_intact"] is True
    assert payload["family_count"] == 27
    assert payload["labels_missing_from_validation"] == []
    assert payload["labels_missing_from_test"] == []
    assert payload["target_ratios"] == {"train": 0.70, "validation": 0.15, "test": 0.15}
    # Actual ratios are recorded honestly next to the target, not faked.
    assert payload["actual_ratios"] != payload["target_ratios"]
    assert 0.60 <= payload["actual_ratios"]["train"] <= 0.75


def test_classes_are_discovered_from_the_data_not_hard_coded():
    df = ds.load_dataset()
    vocabulary = ds.discover_target_classes(df)

    assert len(vocabulary) == 5
    assert "strength_reduction" in set(ds.STRATEGIES)
    assert "strength_reduction" not in vocabulary
    # The strategy vocabulary (six passes) and the label vocabulary (five
    # classes) are deliberately different sizes.
    assert len(ds.STRATEGIES) == 6
    assert set(vocabulary) <= set(ds.STRATEGIES)


def test_strength_reduction_never_wins_a_label():
    """A pass with no winning examples must not silently gain one."""
    df = ds.load_dataset()
    assert (df[ds.TARGET_COLUMN] == "strength_reduction").sum() == 0

    winning_column = ds.label_measurement_column("strength_reduction")
    assert winning_column in df.columns
    # It is not that the pass never ran - it ran and never beat the others.
    assert df[winning_column].notna().all()


def test_no_leaky_column_is_present_as_a_model_input():
    df = ds.load_dataset()
    X, _ = ds.feature_matrix(df)
    assert list(X.columns) == FEATURE_KEYS
    assert not (set(X.columns) & ds.LEAKY_COLUMNS)
    # And the leaky columns really are in the CSV - they exist, they are just
    # excluded, which is a stronger claim than "the column is absent".
    assert ds.LEAKY_COLUMNS <= set(df.columns)
