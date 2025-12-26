import os
import glob

# Audit Script Template
# Usage: python audit_structure.py > STRUCTURE_REPORT.md

ROOT = "experiments"
ALLOWED_EXTENSIONS = [".py", ".sh"]

print("# STRUCTURE AUDIT REPORT\n")
print("| Experiment | File | Issue | Proposed Action |")
print("|---|---|---|---|")

for d in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, d)
    if not os.path.isdir(path) or not d.startswith("v"):
        continue
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    for f in files:
        # Check for Models in Root
        if "model" in f.lower() and f.endswith(".py") and f != "train.py":
            print(f"| `{d}` | `{f}` | Model in Root | Move to `models/` |")
            
        # Check for Logs in Root
        if f.endswith(".log"):
            print(f"| `{d}` | `{f}` | Log in Root | Move to `logs/` |")

        # Check for Checkpoints in Root
        if f.endswith(".pt"):
            print(f"| `{d}` | `{f}` | Checkpoint in Root | Move to `checkpoints/` |")

        # Check for Loose Tests
        if "test" in f or "diag" in f:
             print(f"| `{d}` | `{f}` | Loose Test | Move to `tests/` |")
