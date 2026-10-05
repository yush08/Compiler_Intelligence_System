import sys
sys.path.insert(0, ".")
from backend.tac.parser import parse_tac
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS
from backend.api.serialize import tac_to_lines

cases = {
 "constant_folding": "t1 = 5 + 3\nt2 = 7 * 8\nx = t1 + t2",
 "constant_propagation": "a = 10\nt1 = a + 5\nx = t1 * 2",
 "common_subexpression": "t1 = a + b\nt2 = a + b\nt3 = a + b\nx = t2",
 "dead_code_elimination": "t1 = a + b\nx = 10\nt2 = 3 + 4",
 "algebraic_simplification": "t1 = 5 + 0\nt2 = 7 * 1\nx = t2",
 "strength_reduction": "t1 = x * 8\nt2 = y * 16\nx = t2",
}
for name, src in cases.items():
    fn = OPTIMIZATION_FUNCTIONS[name]
    before = parse_tac(src); after = fn(parse_tac(src))
    print(f"=== {name}  ({len(before)} -> {len(after)})")
    for l in tac_to_lines(before): print("   - " + l)
    for l in tac_to_lines(after):  print("   + " + l)
    print()
