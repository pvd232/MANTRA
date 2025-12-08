# src/mantra/programs/__init__.py

from .cnmf import run_cnmf   # or whatever your main entry point is
from .config import CNMFConfig, CNMFResults

__all__ = [
    "run_cnmf",
    "CNMFConfig",
    "CNMFResults",
]
