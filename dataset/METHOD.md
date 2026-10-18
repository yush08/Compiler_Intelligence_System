# Dataset Method

How the optimization dataset is built, labelled, de-duplicated and split.
The generator is `experiments/generate_dataset.py`; nothing in this document
is hand-picked — every program, label and split assignment is produced by
code that can be re-run byte-identically.

## 1. Corpus (programs)

The corpus is assembled from five sources, in a fixed order:

| Source | Families | What it is |
| --- | --- | --- |
| `experiments/corpus.py` opportunity classes | 7 (`corpus:<class>`) | Seven opportunity templates × 4 sizes (5/10/15/20 statements), the original systematic corpus. |
| Repository examples | 3 (`repo:parser`, `repo:frontend`, `repo:test-api`) | The nine TAC programs that ship with the repository (parser examples, frontend scenarios, the API default input). |
| Live opportunity builders | 6 (`<class>_live`) | One opportunity kind per program, every produced value consumed so the winner is decided by measurement rather than by dead padding. |
| Variant builders | 8 (`<class>_live_b/_c`) | Structurally different second-generation templates for the same opportunities. They exist so **every label class has at least three whole families**, which is what lets a grouped split put an untouched family of each class into train, validation *and* test. |
| Extra categories | 3 (`loops`, `no_opportunity`, `temporaries`) | Control-flow and baseline programs the opportunity builders do not cover. |

A **family** = one program template (all replicates of one builder). The
`family` column records it; it is provenance metadata only and is listed in
`LEAKY_COLUMNS`, so it can never reach the model as an input.

Reproducibility: every builder is seeded from `DATASET_SEED`
(`20261005`) plus category and replicate, so re-running the generator
produces byte-identical programs.

## 2. Labels (measured, never hand-written)

For every program the generator:

1. extracts the eight features (`backend.features.extractor`);
2. runs **each of the six passes on its own independent copy** of the original
   TAC (`backend.optimizer.measure`);
3. records `useful_improvement` = statement reduction + operation reduction
   for each pass;
4. sets `best_strategy` = argmax of those six measurements, ties broken by
   `STRATEGY_TIE_ORDER`.

Validation re-derives every label from the stored measurements and fails the
run on any disagreement. `strength_reduction` is a documented conservative
no-op, so it measures 0 everywhere and never wins — it therefore stays out of
the prediction vocabulary (`discover_target_classes` finds five classes, not
six). Its real zeros are still stored and shown.

## 3. De-duplication

Programs are de-duplicated on whitespace-normalised TAC text **before**
labelling and splitting (`deduplicate`), and validation separately errors if
any two records share a program. Duplicates found are reported
(`duplicates_dropped` in `summary.json`).

## 4. The split (grouped, label-stratified, deterministic)

Implemented by `assign_splits` in the generator; the machine-readable copy of
the method is stored in `dataset/metadata/splits.json` under `"method"`.

1. **Group.** Records are grouped by `family`. A whole family is dealt to
   exactly one split, so no program template ever appears in two splits — the
   test split contains *unseen templates*, not merely unseen rows.
2. **Primary label.** Each family's label is its dominant measured
   `best_strategy` (ties broken by `STRATEGY_TIE_ORDER`).
3. **Deal.** Within each label group, families are ordered by size (the seed
   breaks size ties). The family closest to the group's 15% share goes to
   validation, the next closest to test, everything else to train. Inputs are
   *only* measured labels and family sizes — never features, never model
   output — so this is label stratification to reach the 70/15/15 target, not
   test-set tuning.
4. **Fallbacks.** A label with 2 families lands in train + test (validation
   uncovered); a label with 1 family lands in train only. Both cases are
   reported in `splits.json` (`labels_missing_from_*`), never hidden.

Because families move as blocks, actual ratios are close to but not exactly
70/15/15; `splits.json` records `target_ratios` and `actual_ratios` side by
side, plus `family_count`, `families_intact` and the per-split family lists.
Validation errors if any family straddles splits.

`train.py` reads the `split` column as-is and never moves a sample. Model
selection uses only the validation split; the test split is touched exactly
once, for the final reported metrics.

## 5. Retraining policy

The dataset and model artifacts are only regenerated together:

- after any generator/split change, the previous artifacts are backed up
  first (the artifacts are git-tracked; a temp copy is kept during the run);
- `python -m backend.ml.train` retrains from scratch with the fixed seed;
- no test-set result is ever consulted while choosing configuration,
  corpus or split parameters — the grid is selected on validation macro-F1
  only, and the corpus/split parameters above were fixed by method
  (coverage and ratio targets), not by model scores.

## 6. Outputs

```
dataset/raw/tac_programs.json               programs + provenance (family, source)
dataset/processed/optimization_dataset.csv  239 rows × 46 columns (adds `family`)
dataset/processed/optimization_dataset.json same records, JSON form
dataset/metadata/summary.json               distributions, family_count, label rule
dataset/metadata/validation_report.json     quality checks incl. family integrity
dataset/metadata/splits.json                method text, families, target/actual ratios
```

Regenerate with:

```bash
python experiments/generate_dataset.py
python -m backend.ml.train
```
