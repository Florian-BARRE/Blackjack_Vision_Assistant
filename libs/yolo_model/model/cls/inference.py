# ====== Code Summary ======
# Data structure for representing classification results with ranked predictions.
# Includes primary rank, top-5 alternatives, and confidence scores.

# ====== Standard Library Imports ======
from __future__ import annotations
from dataclasses import dataclass

# ====== Internal Project Imports ======
from public_models.rank import Rank
from ..base import BaseInference


@dataclass
class RankInference(BaseInference):
    """
    Inference result for a rank classification model.

    Attributes:
        rank (Rank): Most confident rank prediction.
        top5_rank (list[Rank]): Top-5 rank predictions.
        top5_confidence (list[float]): Confidence scores for top-5 ranks.
    """
    rank: Rank
    top5_rank: list[Rank]
    top5_confidence: list[float]

    @classmethod
    def default(cls) -> RankInference:
        """
        Generate a default RankInference result with unknown rank.

        Returns:
            RankInference: Default inference with no confidence or rank info.
        """
        return cls(
            confidence=0.0,
            rank=Rank.UNKNOWN,
            top5_rank=[],
            top5_confidence=[],
        )
