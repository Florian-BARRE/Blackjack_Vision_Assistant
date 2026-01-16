# ====== Code Summary ======
# Data model for representing and reasoning about a logical card composed of
# a scanned detection and a real/physical detection. Provides utilities for
# confidence consolidation, rank display, and agreement checks.

# ====== Standard Library Imports ======
from __future__ import annotations

from dataclasses import dataclass

# ====== Internal Project Imports ======
from ..detection import DetectedCard
from public_models.rank import Rank


@dataclass(slots=True)
class CardPair:
    """
    A pair of matched cards (scanned + real) consolidated into one logical card.

    Attributes:
        scanned (DetectedCard | None): The scanned card detection (usually more reliable rank).
        real (DetectedCard | None): The real/physical card detection.
    """

    scanned: DetectedCard | None = None
    real: DetectedCard | None = None

    @property
    def is_matched(self) -> bool:
        """
        Whether scanned and real cards agree on rank (or are unknown/partial).

        Returns:
            bool: True if ranks match or if any side is missing/unknown; otherwise False.
        """
        # 1. Accept if either side is missing
        if self.scanned is None or self.real is None:
            return True  # Single card = OK

        # 2. Accept if either rank is unknown
        if self.scanned.rank == Rank.UNKNOWN or self.real.rank == Rank.UNKNOWN:
            return True

        # 3. Accept only if ranks match
        return self.scanned.rank == self.real.rank

    @property
    def consolidated_rank(self) -> Rank:
        """
        Consolidated rank, preferring scanned rank (more reliable).

        Returns:
            Rank: The chosen Rank, or Rank.UNKNOWN if none available.
        """
        # 1. Prefer scanned rank if known
        if self.scanned is not None and self.scanned.rank != Rank.UNKNOWN:
            return self.scanned.rank

        # 2. Fallback to real rank if known
        if self.real is not None and self.real.rank != Rank.UNKNOWN:
            return self.real.rank

        # 3. Unknown if neither is usable
        return Rank.UNKNOWN

    @property
    def consolidated_confidence(self) -> float:
        """
        Consolidated confidence score.

        If both exist and match, averages confidences.
        If mismatch, penalizes by halving the available confidence.

        Returns:
            float: Confidence in [0.0, 1.0] (best-effort based on inputs).
        """
        # 1. Handle case of no inputs
        if self.scanned is None and self.real is None:
            return 0.0

        # 2. Extract available confidences
        scanned_conf = self.scanned.confidence if self.scanned is not None else 0.0
        real_conf = self.real.confidence if self.real is not None else 0.0

        # 3. Handle matched case
        if self.is_matched:
            if self.scanned is not None and self.real is not None:
                return (scanned_conf + real_conf) / 2.0
            return scanned_conf if self.scanned is not None else real_conf

        # 4. Handle mismatch (penalize)
        return scanned_conf / 2.0 if self.scanned is not None else real_conf / 2.0

    @property
    def has_both(self) -> bool:
        """
        Whether both scanned and real detections are present.

        Returns:
            bool: True if both detections exist; otherwise False.
        """
        return self.scanned is not None and self.real is not None

    @property
    def display_rank(self) -> str:
        """
        Human-friendly rank display.

        Returns:
            str: Rank string, or "?" when unknown.
        """
        rank = self.consolidated_rank
        return rank.value if rank != Rank.UNKNOWN else "?"
