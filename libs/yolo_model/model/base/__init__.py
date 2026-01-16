# ------------------- Public Interface ------------------- #
# Exposes the base classes for all model and inference implementations.

# ------------------- Local Project Imports ------------------- #
from .model import BaseModel
from .inference import BaseInference

# -------------------- Public API -------------------- #
__all__ = [
    "BaseModel",
    "BaseInference",
]
