import sys
sys.path.insert(0, ".")
from backend.tac.parser import parse_tac
from backend.optimizer.engine import OPTIMIZATION_FUNCTIONS, run_iterative_optimization
from backend.scoring.rule_based import select_strategy

samples = {
 "fold": "t1 = 5 + 3\nt2 = 7 * 8\nt3 = t1 + t2\nx = t3",
 "prop": "a = 10\nt1 = a + 5\nt2 = t1 * 2\nx = t2",
 "cse":  "t1 = a + b\nt2 = a + b\nt3 = a + b\nx = t2",
 "dce":  "t1 = a + b\nx = 10",
 "alg":  "t1 = 5 + 0\nt2 = 7 * 1\nt3 = 9 - 0\nt4 = t3 / 1\nx = t4",
 "sr":   "t1 = x * 8\nt2 = y * 16\nt3 = z * 100\nx = t3",
 "mixed":"t1 = 5 + 3\nt2 = a + b\nt3 = a + b\nt4 = t1 * 1\nunused = 99\nx = t4",
 "demo": "t1 = 10\nt2 = 20\nt3 = t1 + t2\nt4 = t1 + t2\nt5 = t3 * 1\nt6 = 100\nt7 = t6 + 50\nunused = t7\nt8 = t3 * 2",
}
hdr = "sample     n   sel                        best                       bestred  adaptive iters conv"
print(hdr); print("-"*len(hdr))
for name, src in samples.items():
    tac = parse_tac(src)
    per = {k: len(tac) - len(fn(parse_tac(src))) for k, fn in OPTIMIZATION_FUNCTIONS.items()}
    sel = select_strategy(tac)
    r = run_iterative_optimization(tac, max_iterations=20)
    best = max(per, key=per.get)
    print(f"{name:10} {len(tac):3}  {sel:26} {best:26} {max(per.values()):7}  {r['total_reduction']:8} {r['total_iterations']:5} {r['converged']}")
    print(f"           per-pass={per}")
