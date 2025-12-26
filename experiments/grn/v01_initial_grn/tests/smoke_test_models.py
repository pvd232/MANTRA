import sys
import os
import types

# Robust mocking of torch package structure
class MockModule:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return MockTensor()
    def parameters(self): return []

class MockTensor:
    def view(self, *args): return self
    def expand(self, *args): return self
    def unsqueeze(self, *args): return self
    def chunk(self, *args, **kwargs): return (self, self)
    def to(self, *args): return self
    def squeeze(self, *args): return self
    def __rmul__(self, other): return self
    def __mul__(self, other): return self

class NoGrad:
    def __enter__(self): pass
    def __exit__(self, *args): pass
    def __call__(self, func): return func

# Create the top-level package
torch_mock = types.ModuleType("torch")
torch_mock.__path__ = [] # Mark as package
torch_mock.Tensor = MockTensor
torch_mock.randn = lambda *args: MockTensor()
torch_mock.cat = lambda *args, **kwargs: MockTensor()
torch_mock.einsum = lambda *args: MockTensor()
torch_mock.device = lambda x: x
torch_mock.no_grad = NoGrad

# Create submodules
torch_nn = types.ModuleType("torch.nn")
torch_nn.Module = MockModule
torch_nn.Linear = MockModule
torch_nn.LayerNorm = MockModule
torch_nn.ReLU = MockModule
torch_nn.Dropout = MockModule
torch_nn.Embedding = MockModule
torch_nn.Sequential = MockModule
torch_nn.Parameter = lambda x: x
torch_nn.ModuleList = list # Expects iterable in init

torch_optim = types.ModuleType("torch.optim")
torch_utils = types.ModuleType("torch.utils")
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = lambda *args, **kwargs: []

# Link them up
torch_mock.nn = torch_nn
torch_mock.optim = torch_optim
torch_mock.utils = torch_utils
torch_mock.utils.data = torch_utils_data

# Inject into sys.modules
sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.optim"] = torch_optim
sys.modules["torch.utils"] = torch_utils
sys.modules["torch.utils.data"] = torch_utils_data

# 2. Setup path
# Assuming we run this from project root, but let's be robust
current_dir = os.path.dirname(os.path.abspath(__file__))
# tests/ -> experiments/v01.../tests/ -> experiments/v01.../
exp_root = os.path.dirname(current_dir)
# experiments/v01.../
src_path = exp_root

sys.path.append(src_path)
print(f"Testing imports from: {src_path}")

# 3. Test Imports
try:
    from models import GRNGNN, ConditionEncoder, GeneGNNLayer, TraitHead
    print("[PASS] Imported all components.")
except ImportError as e:
    print(f"[FAIL] Import Error: {e}")
    sys.exit(1)

# 4. Test Instantiation (Logic Check)
try:
    cond_enc = ConditionEncoder(n_regulators=10)
    print(f"[PASS] ConditionEncoder instantiated.")
    
    gnn = GRNGNN(n_regulators=10, n_genes=50)
    print(f"[PASS] GRNGNN instantiated.")
    
    # Test forward pass logic minimally
    res = gnn(torch_mock.Tensor(), None, torch_mock.Tensor())
    print(f"[PASS] GRNGNN forward pass simulated.")

except Exception as e:
    print(f"[FAIL] Logic Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
