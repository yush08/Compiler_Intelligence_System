"""Pydantic request/response schemas for the ML and comparison endpoints.

Requests are strict - an unknown selector, a missing TAC or an out-of-range
iteration count is rejected with a 422 before any work happens.

Responses use ``extra="allow"``.  The evaluation and comparison payloads are
deeply nested and grow as more real measurements are added; silently dropping a
field because a model definition lagged behind would corrupt the numbers the
frontend displays, so extra fields are preserved rather than pruned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ExplainRequest",
    "CompareRequest",
    "MlStatusResponse",
    "MlMetricsResponse",
    "FeatureImportanceResponse",
    "PredictResponse",
    "ExplainResponse",
    "CompareResponse",
    "SelectorMode",
]

SelectorMode = ["rule", "ml"]


class _AllowExtra(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- requests ---------------------------------------------------------------

class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tac: str = Field(..., description="Three Address Code source")
    top_k: int = Field(default=8, ge=1, le=8)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tac: str = Field(..., description="Three Address Code source")
    max_iterations: int = Field(default=20, ge=1, le=20)
    include_shap: bool = Field(default=True)


# --- responses --------------------------------------------------------------

class MlStatusResponse(_AllowExtra):
    available: bool
    ready: bool
    model_version: str | None = None
    model_type: str | None = None
    trained_at: str | None = None
    training_samples: int | None = None
    dataset_fingerprint: str | None = None
    feature_names: list[str] = Field(default_factory=list)
    supported_classes: list[str] = Field(default_factory=list)
    unsupported_classes: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    artifact_file: str | None = None
    reason: str | None = None
    shap: dict = Field(default_factory=dict)


class MlMetricsResponse(_AllowExtra):
    available: bool
    reason: str | None = None
    evaluation: dict = Field(default_factory=dict)
    confusion_matrix: dict = Field(default_factory=dict)
    classification_report: dict = Field(default_factory=dict)


class FeatureImportanceResponse(_AllowExtra):
    available: bool
    reason: str | None = None
    native: dict = Field(default_factory=dict)
    shap: dict = Field(default_factory=dict)
    comparison_note: str | None = None


class PredictResponse(_AllowExtra):
    available: bool
    selector: str = "ml"
    reason: str | None = None
    predicted_strategy: str | None = None
    predicted_probability: float | None = None
    probabilities: dict = Field(default_factory=dict)
    probability_ranking: list = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    unsupported_classes: list[str] = Field(default_factory=list)
    feature_values: dict = Field(default_factory=dict)
    model_version: str | None = None
    model_type: str | None = None
    trained_at: str | None = None
    probabilities_calibrated: bool = False
    note: str | None = None


class ExplainResponse(_AllowExtra):
    available: bool
    reason: str | None = None
    shap: dict = Field(default_factory=dict)
    explainer: str | None = None
    predicted_class: str | None = None
    predicted_class_index: int | None = None
    probabilities: dict = Field(default_factory=dict)
    base_value: float | None = None
    contributions: list = Field(default_factory=list)
    ranked_contributions: list = Field(default_factory=list)
    supporting: list = Field(default_factory=list)
    opposing: list = Field(default_factory=list)
    total_contribution: float | None = None
    reconstructed_probability: float | None = None
    model_probability: float | None = None
    note: str | None = None
    summary: str | None = None
    disclaimer: str | None = None


class CompareResponse(_AllowExtra):
    available: bool
    reason: str | None = None
    original_lines: list[str] = Field(default_factory=list)
    original_text: str = ""
    original_instruction_count: int = 0
    original_operation_count: int = 0
    original_work_remaining: int = 0
    features: dict = Field(default_factory=dict)
    rule_based: dict = Field(default_factory=dict)
    ml: dict = Field(default_factory=dict)
    agreement: dict = Field(default_factory=dict)
    winner: str | None = None
    winner_note: str | None = None
    decided_by: str | None = None
