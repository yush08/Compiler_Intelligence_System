"""
Tests for the Automated Optimization Strategy Selection System.

Tests all six optimizations, feature extraction, rule-based scoring,
and the iterative optimizer.
"""

import sys
import os

# Add the project to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tac.parser import parse_tac, constant_folding_example, dead_code_example, cse_example, algebraic_simplification_example, mixed_example
from backend.tac.instruction import Instruction
from backend.features.extractor import extract_features
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy, strategy_reason
from backend.optimizer.engine import run_iterative_optimization


def test_tac_parser():
    """Test TAC parsing."""
    # Test constant folding example
    tac = constant_folding_example()
    assert len(tac) == 3, f"Expected 3 instructions, got {len(tac)}"
    print("  ✓ constant_folding_example parsed correctly")

    # Test dead code example
    tac = dead_code_example()
    assert len(tac) == 2, f"Expected 2 instructions, got {len(tac)}"
    print("  ✓ dead_code_example parsed correctly")

    # Test CSE example
    tac = cse_example()
    assert len(tac) == 3, f"Expected 3 instructions, got {len(tac)}"
    print("  ✓ cse_example parsed correctly")

    # Test algebraic simplification example
    tac = algebraic_simplification_example()
    assert len(tac) == 3, f"Expected 3 instructions, got {len(tac)}"
    print("  ✓ algebraic_simplification_example parsed correctly")

    # Test mixed example
    tac = mixed_example()
    assert len(tac) == 6, f"Expected 6 instructions, got {len(tac)}"
    print("  ✓ mixed_example parsed correctly")

    # Test parsing from source string
    source = """
t1 = 5 + 3
t2 = a + b
t3 = a + b
t4 = t1 * 1
t5 = 100
x = t4
"""
    tac = parse_tac(source)
    assert len(tac) == 6, f"Expected 6 instructions from source, got {len(tac)}"
    print("  ✓ parse_tac from source string works")


def test_constant_folding():
    """Test constant folding optimization."""
    from backend.optimizations.constant_folding import constant_folding

    # Example: t1 = 5 + 3 should become t1 = 8
    tac = constant_folding_example()
    result = constant_folding(tac)
    # t1 = 5 + 3 should fold to t1 = 8
    assert result.instructions[0].opcode == Instruction.CONST, \
        f"Expected CONST opcode after folding, got {result.instructions[0].opcode}"
    assert result.instructions[0].arg1 == "8", \
        f"Expected result '8', got '{result.instructions[0].arg1}'"
    print("  ✓ constant_folding works correctly")


def test_constant_propagation():
    """Test constant propagation optimization."""
    from backend.optimizations.constant_propagation import constant_propagation

    # Example: a = 10; t1 = a + 5 should propagate
    source = """
a = 10
t1 = a + 5
"""
    tac = parse_tac(source)
    result = constant_propagation(tac)
    # After constant propagation, t1 should reference a constant
    print(f"    Constant prop result: {result}")
    print("  ✓ constant_propagation runs without error")


def test_common_subexpression():
    """Test CSE optimization."""
    from backend.optimizations.common_subexpression import common_subexpression_elimination

    # t1 = a + b; t2 = a + b should eliminate the second
    tac = cse_example()
    result = common_subexpression_elimination(tac)
    # The second t2 = a + b should become t2 = t1
    print(f"    CSE result: {result}")
    print("  ✓ common_subexpression_elimination runs without error")


def test_dead_code():
    """Test dead code elimination optimization."""
    from backend.optimizations.dead_code import dead_code_elimination

    # t1 = a + b (dead if t1 never used); t2 = 10
    tac = dead_code_example()
    result = dead_code_elimination(tac)
    print(f"    Dead code result: {result}")
    print("  ✓ dead_code_elimination runs without error")


def test_algebraic_simplification():
    """Test algebraic simplification optimization."""
    from backend.optimizations.algebraic_simplification import algebraic_simplification

    # x + 0 → x
    source = """
t1 = 5 + 0
"""
    tac = parse_tac(source)
    result = algebraic_simplification(tac)
    print(f"    Algebraic simplification result: {result}")
    print("  ✓ algebraic_simplification runs without error")


def test_feature_extraction():
    """Test feature extraction from TAC."""
    # Constant folding example
    tac = constant_folding_example()
    features = extract_features(tac)
    assert features["num_statements"] == 3
    assert features["num_constants"] > 0
    assert features["const_ratio"] > 0
    print(f"    Features: {features}")
    print("  ✓ feature_extractor works correctly")


def test_rule_based_scoring():
    """Test rule-based strategy scoring."""
    # CSE example - should have high CSE score
    tac = cse_example()
    scores = rule_based_strategy_scores(tac)
    selected = select_strategy(tac)
    reason = strategy_reason(tac)

    assert "constant_folding" in scores
    assert "common_subexpression" in scores
    assert selected is not None
    assert reason is not None
    print(f"    Scores: {scores}")
    print(f"    Selected: {selected}")
    print(f"    Reason: {reason}")
    print("  ✓ rule_based_strategy_scores works correctly")


def test_iterative_optimizer():
    """Test the iterative optimization engine."""
    # Mixed example with multiple opportunities
    tac = mixed_example()

    result = run_iterative_optimization(tac, max_iterations=5)

    assert "history" in result
    assert "final_tac" in result
    assert "total_reduction" in result
    assert "total_iterations" in result

    print(f"    Initial: {len(tac)} instructions")
    print(f"    Final: {len(result['final_tac'])} instructions")
    print(f"    Total reduction: {result['total_reduction']}")
    print(f"    Iterations: {result['total_iterations']}")
    print(f"    Converged: {result['converged']}")
    print("  ✓ iterative optimizer runs correctly")


def test_all_examples():
    """Test all built-in examples run through the backend."""
    examples = [
        ("Constant Folding", constant_folding_example),
        ("Dead Code", dead_code_example),
        ("CSE", cse_example),
        ("Algebraic", algebraic_simplification_example),
        ("Mixed", mixed_example),
    ]

    for name, example_fn in examples:
        tac = example_fn()
        assert tac is not None and len(tac) > 0, f"{name} example failed"
        print(f"  ✓ {name} example: {len(tac)} instructions")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Tests")
    print("=" * 60)

    print("\n--- TAC Parser Tests ---")
    test_tac_parser()

    print("\n--- Optimization Tests ---")
    test_constant_folding()
    test_constant_propagation()
    test_common_subexpression()
    test_dead_code()
    test_algebraic_simplification()

    print("\n--- Feature Extraction Tests ---")
    test_feature_extraction()

    print("\n--- Rule-Based Scoring Tests ---")
    test_rule_based_scoring()

    print("\n--- Iterative Optimizer Tests ---")
    test_iterative_optimizer()

    print("\n--- All Examples Test ---")
    test_all_examples()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)