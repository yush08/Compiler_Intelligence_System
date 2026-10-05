"""API adapter tests. Existing optimizer tests remain unchanged."""

from fastapi.testclient import TestClient

from backend.api.pipeline import analyze_source, simulate_source
from backend.api.server import app

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
