import os
import shutil

# Generalized Cleanup Script for Agentic Research Repos
# Usage: python cleanup_structure.py

ROOT = "experiments"
SUBDIRS = ["models", "tests", "scripts", "logs", "audits", "plots", "checkpoints", "journal_assets"]

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

print(f"Scanning {ROOT} for loose files...")

for d in os.listdir(ROOT):
    path = os.path.join(ROOT, d)
    if not os.path.isdir(path) or d == "archive":
        continue
    
    # Optional: Check for version prefix (e.g. "v")
    # if not d.startswith("v"): continue
    
    # Create standard subdirs
    for s in SUBDIRS:
        mkdir_p(os.path.join(path, s))

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    for f in files:
        src = os.path.join(path, f)
        
        # 1. Media
        if f.endswith((".pdf", ".png", ".jpg", ".svg")):
            shutil.move(src, os.path.join(path, "journal_assets", f))
            print(f"[{d}] Moved Media: {f}")

        # 2. Models
        elif "model" in f or "net" in f or "transformer" in f:
            # Avoid moving the main train script if it's named 'train_model.py'
            if not f.startswith("train") and not f.startswith("audit"):
                shutil.move(src, os.path.join(path, "models", f))
                print(f"[{d}] Moved Model: {f}")

        # 3. Tests keywods
        elif "test" in f or "diag" in f or "debug" in f or "inspect" in f:
             shutil.move(src, os.path.join(path, "tests", f))
             print(f"[{d}] Moved Test: {f}")

        # 4. Results/Logs
        elif "result" in f.lower() or "report" in f.lower() or f.endswith(".json"):
             if f != "MANIFEST.json":
                shutil.move(src, os.path.join(path, "audits", f))
                print(f"[{d}] Moved Result: {f}")
        
        elif f.endswith(".log"):
             shutil.move(src, os.path.join(path, "logs", f))
             print(f"[{d}] Moved Log: {f}")

        # 5. Loose Python Scripts
        elif f.endswith(".py"):
            # Keep main entry points
            if f.startswith("train") or f.startswith("audit") or f.startswith("run") or f == "setup.py":
                continue
            shutil.move(src, os.path.join(path, "scripts", f))
            print(f"[{d}] Moved Script: {f}")

print("Cleanup Complete.")
