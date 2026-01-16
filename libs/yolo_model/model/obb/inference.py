# ====== Code Summary ======
# Data structure for holding OBB inference results, including bounding box,
# object type, confidence score, and centroid computation with lazy caching.

# ====== Standard Library Imports ======
from dataclasses import dataclass, field

# ====== Third-Party Library Imports ======
import numpy as np

# ====== Internal Project Imports ======
from public_models.obb_type import ObbType
from ..base import BaseInference


@dataclass
class ObbInference(BaseInference):
    """
    Data container for a single Oriented Bounding Box (OBB) inference result.

    Attributes:
        box (np.ndarray): Bounding box as [x1, y1, x2, y2].
        obb_type (ObbType): Enum representing the detected object's class/type.
        confidence (float): Confidence score of the prediction.
    """
    box: np.ndarray
    obb_type: ObbType
    confidence: float

    _centroid: np.ndarray | None = field(default=None, init=False, repr=False)

    def _compute_centroid(self) -> np.ndarray:
        """
        Compute the centroid of the bounding box.

        Returns:
            np.ndarray: Centroid coordinates [x, y].
        """
        # 1. Unpack box coordinates
        x1, y1, x2, y2 = self.box

        # 2. Calculate midpoint between top-left and bottom-right
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

    @property
    def centroid(self) -> np.ndarray:
        """
        Cached property to retrieve the centroid of the box.

        Returns:
            np.ndarray: Centroid [x, y].
        """
        # 1. Lazy evaluation and cache storage
        if self._centroid is None:
            self._centroid = self._compute_centroid()

        # 2. Return a copy to prevent mutation
        return self._centroid.copy()
