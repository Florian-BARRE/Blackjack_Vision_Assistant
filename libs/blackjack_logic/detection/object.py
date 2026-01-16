# ====== Code Summary ======
# Defines a data model for non-card object detections such as CardHolders or Traps.
# Includes bounding box geometry, confidence scoring, and center point calculation.

# ====== Standard Library Imports ======
from dataclasses import dataclass
import numpy as np

# ====== Internal Project Imports ======
from public_models.obb_type import ObbType


@dataclass
class DetectedObject:
    """
    A detected non-card object (e.g., CardHolder, Trap).

    Attributes:
        obb_type (ObbType): Type of the detected object.
        confidence (float): Detection confidence score.
        box (np.ndarray): Oriented bounding box as a (4, 2) NumPy array.
    """

    obb_type: ObbType
    confidence: float
    box: np.ndarray

    @property
    def x(self) -> float:
        """
        X-coordinate of the object center (average of OBB x-values).

        Returns:
            float: Horizontal center of the object.
        """
        return float(np.mean(self.box[:, 0]))

    @property
    def y(self) -> float:
        """
        Y-coordinate of the object center (average of OBB y-values).

        Returns:
            float: Vertical center of the object.
        """
        return float(np.mean(self.box[:, 1]))
