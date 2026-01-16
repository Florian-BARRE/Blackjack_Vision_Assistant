# ------------------- Public Interface ------------------- #
# Exposes the MetaModel and MetaModelInference classes for external use.

# ------------------- Local Project Imports ------------------- #
from .model import MetaModel
from .inference import MetaModelInference

# -------------------- Public API -------------------- #
__all__ = [
    "MetaModel",
    "MetaModelInference",
]
