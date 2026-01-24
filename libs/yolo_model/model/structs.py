# ====== Code Summary ======
# Enum to define supported model types for classification and OBB detection tasks.

# ====== Standard Library Imports ======
from enum import StrEnum


class ModelType(StrEnum):
    """
    Enumeration of model types used across the system.

    Attributes:
        OBB (str): Oriented Bounding Box detection model.
        CLS (str): Classification model.
    """
    OBB = "obb"
    CLS = "classify"
