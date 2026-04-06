"""Quick syntax & logic check for all modified files."""
import ast
import sys

files = [
    "EmailTriage/server/EmailTriage_environment.py",
    "EmailTriage/models.py",
    "EmailTriage/client.py",
    "EmailTriage/server/app.py",
    "inference.py",
]

ok = True
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK  {f}")
    except SyntaxError as e:
        print(f"FAIL  {f}: {e}")
        ok = False

if not ok:
    sys.exit(1)

print("\nAll files pass syntax check.")
