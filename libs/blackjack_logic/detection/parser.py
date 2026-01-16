# ====== Code Summary ======
# Utility module for converting raw model inferences into structured detection objects.
# Produces a `FrameDetections` instance containing parsed cards, card holders, and traps,
# handling partial or missing rank information gracefully.

# ====== Standard Library Imports ======
from __future__ import annotations

# ====== Third-Party Library Imports ======
import numpy as np
from loggerplusplus import LoggerClass

# ====== Internal Project Imports ======
from public_models.obb_type import ObbType
from public_models.rank import Rank
from .card import DetectedCard
from .frame import FrameDetections
from .object import DetectedObject

class DetectionParser(LoggerClass):
    """
    Parse raw inference objects into structured `FrameDetections`.

    Notes:
        - Gracefully handles missing rank inference (defaults to Rank.UNKNOWN).
        - Uses rank confidence when present; otherwise falls back to OBB confidence.
        - Unknown or unsupported OBB types are silently ignored for forward compatibility.
    """

    def __init__(self) -> None:
        """
        Initialize the parser with logging support.
        """
        super().__init__()
        self.logger.debug("START initializing DetectionParser")
        self.logger.debug("END initializing DetectionParser")

    @staticmethod
    def parse(inferences: list[object]) -> FrameDetections:
        """
        Parse a list of raw inferences into `FrameDetections`.

        Args:
            inferences (list[object]): Model output, where each inference must include:
                - obb_inference: object with `obb_type`, `box`, `confidence`
                - rank_inference (optional): object with `rank`, `confidence`

        Returns:
            FrameDetections: Parsed detection data for a single frame.

        Raises:
            ValueError: If an inference is missing required OBB fields.
            TypeError: If the box field cannot be converted to float32 NumPy array.
        """
        # 1. Prepare output structure
        detections = FrameDetections()

        # 2. Iterate over each raw inference
        for idx, inf in enumerate(inferences):
            if not hasattr(inf, "obb_inference"):
                raise ValueError(f"Inference missing obb_inference (index={idx})")

            obb = getattr(inf, "obb_inference")
            rank_inf = getattr(inf, "rank_inference", None)

            # 3. Validate required fields in OBB
            if not hasattr(obb, "obb_type") or not hasattr(obb, "box") or not hasattr(obb, "confidence"):
                raise ValueError(f"OBB inference missing required fields (index={idx})")

            # 4. Parse and normalize box
            try:
                box = np.array(getattr(obb, "box"), dtype=np.float32)
            except Exception as err:
                raise TypeError(f"Invalid box format at index={idx}: {err}")

            obb_type = getattr(obb, "obb_type")

            # 5. Handle card object
            if obb_type == ObbType.CARD:
                rank = getattr(rank_inf, "rank", Rank.UNKNOWN) if rank_inf is not None else Rank.UNKNOWN
                conf = (
                    float(getattr(rank_inf, "confidence"))
                    if rank_inf is not None and hasattr(rank_inf, "confidence")
                    else float(getattr(obb, "confidence"))
                )
                detections.cards.append(
                    DetectedCard(rank=rank, confidence=conf, box=box)
                )

            # 6. Handle card holder object
            elif obb_type == ObbType.CARD_HOLDER:
                detections.card_holders.append(
                    DetectedObject(
                        obb_type=ObbType.CARD_HOLDER,
                        confidence=float(getattr(obb, "confidence")),
                        box=box,
                    )
                )

            # 7. Handle trap object
            elif obb_type == ObbType.TRAP:
                detections.traps.append(
                    DetectedObject(
                        obb_type=ObbType.TRAP,
                        confidence=float(getattr(obb, "confidence")),
                        box=box,
                    )
                )

            # 8. Other types are ignored for compatibility with future model types

        # 9. Return final result
        return detections
