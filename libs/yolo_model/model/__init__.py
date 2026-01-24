# ------------------- Public Interface ------------------- #
# Aggregates all public model types for classification and detection tasks.

# ------------------- Local Project Imports ------------------- #
from .cls import (
    RankInference,
    RankModel,
)

from .obb import (
    ObbInference,
    ObbModel,
)

# -------------------- Public API -------------------- #
__all__ = [
    "RankInference",
    "RankModel",
    "ObbInference",
    "ObbModel",
]
