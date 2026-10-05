import sys
sys.path.insert(0, r'D:\PROJECTS\compiler')
import re

line = 't1 = 5 + 3'
print(f'Line: {line!r}')

# Test different regex patterns
patterns = [
    r"^([A-Za_][A-Za0-9_]*)\s*=\s*(.+)$",
    r"^([A-Za_][A-Za0-9_]*)\s*=\s*(.+?)$",
    r"^(\w+)\s*=\s*(.+)$",
]

for p in patterns:
    m = re.match(p, line)
    print(f'Pattern {p}: {m}')
    if m:
        print(f'  Groups: {m.groups()}')