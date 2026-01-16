# ====== Code Summary ======
# Composite inference result for models that combine OBB detection and rank classification.
# Inherits from BaseInference and aggregates outputs from both inference types.

# ====== Standard Library Imports ======
from dataclasses import dataclass

# ====== Internal Project Imports ======
from ..model import ObbInference, RankInference
from ..model.base.inference import BaseInference


@dataclass(kw_only=True)
class MetaModelInference(BaseInference):
    """
    Composite inference result for a model combining OBB detection and rank classification.

    Attributes:
        obb_inference (ObbInference): Result from the OBB detection model.
        rank_inference (RankInference | None): Optional result from rank classification.
        confidence (float): Overall confidence score (can be derived or averaged).
    """
    obb_inference: ObbInference
    rank_inference: RankInference | None = None
    confidence: float = 0.0

    @property
    def time(self) -> float:
        """
        Inference timestamp, derived from the OBB inference.

        Returns:
            float: Epoch time from OBB inference.
        """
        return self.obb_inference.time
