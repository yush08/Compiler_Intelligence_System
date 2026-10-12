"""API adapter tests. Existing optimizer tests remain unchanged."""

from fastapi.testclient import TestClient

from backend.api.pipeline import analyze_source, simulate_source
from backend.api.server import app
from backend.features.extractor import extract_features
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS
from backend.scoring.rule_based import select_strategy
from backend.tac.parser import parse_tac

client = TestClient(app)

DEFAULT_TAC = """
t1 = 10
t2 = 20
t3 = t1 + t2
t4 = t1 + t2
t5 = t3 * 1
t6 = 100
t7 = t6 + 50
unused = t7
t8 = t3 * 2
"""


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True


def test_analyze_empty_tac():
    response = client.post("/api/analyze", json={"tac": "   "})
    assert response.status_code == 400


def test_analyze_valid_tac():
    response = client.post("/api/analyze", json={"tac": DEFAULT_TAC})
    assert response.status_code == 200
    body = response.json()
    assert body["features"]["num_statements"] > 0
    assert body["selected_strategy"]["id"]
    assert len(body["strategies"]) == 6


def test_simulate_valid_tac():
    response = client.post("/api/simulate", json={"tac": DEFAULT_TAC})
    assert response.status_code == 200
    body = response.json()
    assert "final_result" in body
    assert body["final_result"]["original_count"] >= body["final_result"]["final_count"]
    assert isinstance(body["iterations"], list)


def test_pipeline_analyze_source():
    result = analyze_source("t1 = 5 + 3\nt2 = t1")
    assert result["features"]["num_statements"] == 2
    assert result["selected_strategy"]["id"] in {
        "constant_folding",
        "constant_propagation",
        "common_subexpression",
        "dead_code_elimination",
        "algebraic_simplification",
        "strength_reduction",
    }


def test_pipeline_simulate_source():
    result = simulate_source("t1 = 5 + 3\nt2 = t1")
    assert result["final_result"]["original_count"] >= 1
    assert result["iterations"]


# ---------------------------------------------------------------------------
# Regression: the default program used to be served a hard-coded payload
# ---------------------------------------------------------------------------


def test_analyze_reports_really_extracted_features_for_the_default_program():
    """The canned default claimed num_dead_stmts=1 / const_ratio=0.556 while
    the extractor reports 4 and 0.667 for the same source."""
    tac = parse_tac(DEFAULT_TAC)
    body = client.post("/api/analyze", json={"tac": DEFAULT_TAC}).json()
    real = extract_features(tac)

    for key, expected in real.items():
        assert abs(body["features"][key] - expected) < 1e-9, key
    assert body["selected_strategy"]["id"] == select_strategy(tac)


def test_optimize_honors_the_requested_strategy_for_the_default_program():
    """optimize_source used to ignore `strategy` entirely on this input and
    always answer with the same canned Constant Folding result."""
    for name in OPTIMIZATION_FUNCTIONS:
        body = client.post(
            "/api/optimize", json={"tac": DEFAULT_TAC, "strategy": name}
        ).json()
        assert body["iterations"][0]["strategy"] == name
        assert body["selected_strategy"]["id"] == name


def test_analyze_and_optimize_agree_on_the_same_input():
    """analyze said Constant Folding 0.91 while optimize said Common
    Subexpression 0.82 for an identical source."""
    analyzed = client.post("/api/analyze", json={"tac": DEFAULT_TAC}).json()
    optimized = client.post("/api/optimize", json={"tac": DEFAULT_TAC}).json()

    assert analyzed["features"] == optimized["features"]
    assert (
        analyzed["selected_strategy"]["id"] == optimized["selected_strategy"]["id"]
    )
    assert analyzed["strategies"] == optimized["strategies"]


def test_simulate_counts_match_the_tacs_it_returns():
    """The canned payload reported final_count=4 but shipped a 5-line final_tac."""
    body = client.post("/api/simulate", json={"tac": DEFAULT_TAC}).json()
    final_result = body["final_result"]

    assert final_result["original_count"] == len(final_result["original_tac"])
    assert final_result["final_count"] == len(final_result["final_tac"])
    assert final_result["original_count"] >= final_result["final_count"]
    # every reported iteration must describe a real before/after pair
    for item in body["iterations"]:
        assert len(item["tac_before"]) == item["before_count"]
        assert len(item["tac_after"]) == item["after_count"]
