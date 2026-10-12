"""Machine-learning strategy selector.

This package turns the already-generated dataset
(``dataset/processed/optimization_dataset.csv``) into a trained
``RandomForestClassifier`` that predicts ``best_strategy`` from the eight
pre-optimization features, plus the evaluation and SHAP explainability
needed to defend that prediction.

Nothing here replaces the rule-based selector: ``backend.scoring`` remains the
default, and the ML path is only ever used when a caller explicitly asks for it.

Modules
-------
dataset      Load and validate the CSV, enforce the feature/label contract.
train        ``python -m backend.ml.train`` - fit, tune, evaluate, persist.
model        Artifact persistence and cached loading.
predict      Inference against the persisted model.
evaluate     Classification metrics (accuracy, macro/weighted F1, per class).
explain      SHAP explanations with an explicit unavailable fallback.
compare      Rule-based vs ML: prediction quality *and* real optimization runs.
schemas      Pydantic request/response models for the API.
"""

__all__ = [
    "dataset",
    "train",
    "model",
    "predict",
    "evaluate",
    "explain",
    "compare",
    "schemas",
]
