# ====== Code Summary ======
# Top-level public interface for the core model components.
# Resolves Intel MKL error for certain environments before importing heavy dependencies.

# ====== Environment Fix for Intel MKL ======
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Intel MKL-related crash on some platforms

# ====== Internal Imports: Model Types ======
from .model import (
    # OBB Detection
    ObbModel, ObbInference,
    # Classification
    RankModel, RankInference,
)

# ====== Internal Imports: Composite Model ======
from .meta_model import (
    MetaModel, MetaModelInference,
)

# ====== Public API ======
__all__ = [
    "ObbModel",
    "ObbInference",
    "RankModel",
    "RankInference",
    "MetaModel",
    "MetaModelInference",
]
