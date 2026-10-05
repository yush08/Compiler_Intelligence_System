import sys
sys.path.insert(0, r'D:\PROJECTS\compiler')
from backend.tac.parser import parse_tac

source = """
t1 = 5 + 3
t2 = a + b
t3 = a + b
t4 = t1 * 1
t5 = 100
x = t4
"""

tac = parse_tac(source)
print(f'Parsed {len(tac)} instructions (expected 6)')
for i, instr in enumerate(tac.instructions):
    print(f'  {i}: {repr(instr)}')