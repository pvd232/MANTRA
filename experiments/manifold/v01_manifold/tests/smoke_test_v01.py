import sys
import os
import types

# Robust mocking of torch package structure (if needed by dependencies, though ManifoldLearner shouldn't needing it directly if using scanpy, but just in case)
# ManifoldLearner uses scanpy/numpy/scipy. Torch is not used.
# But if we had it, we'd mock it.

# 2. Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
# tests/ -> experiments/v01_manifold/
exp_root = os.path.dirname(current_dir)

sys.path.append(exp_root)
print(f"Testing imports from: {exp_root}")

# 3. Test Imports
try:
    from models import ManifoldLearner
    print("[PASS] Imported ManifoldLearner.")
except ImportError as e:
    print(f"[FAIL] Import Error: {e}")
    sys.exit(1)

# 4. Test Instantiation
try:
    learner = ManifoldLearner()
    print(f"[PASS] ManifoldLearner instantiated.")
except Exception as e:
    print(f"[FAIL] Logic Error: {e}")
    sys.exit(1)
