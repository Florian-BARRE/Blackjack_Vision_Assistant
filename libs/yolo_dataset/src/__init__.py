# ====== Code Summary ======
# Public interface for the yolo_dataset.src module.
# Exports core components, configurations, helpers, and augmentation utilities.

# --------------------- Core Components ---------------------- #
from .core import DatasetProcessor

# --------------------- Classification ----------------------- #
from .classification import (
    ClassificationProcessor,
    ClassificationConfig,
    ClassificationField,
    ClsAugmentationConfig,
    ClsAugmentor,
)

# --------------------- Configuration ------------------------ #
from .config import DatasetConfig, TrainingConfig

# --------------------- Constants ---------------------------- #
from .constants import YOLO_CONSTANTS, AUGMENTATION_PRESETS

# --------------------- Helper Utilities --------------------- #
from .helpers import YOLO_HELPERS

# --------------------- Data Augmentation -------------------- #
from .augmentation import AugmentationConfig, DataAugmentor

# ---------------------- Public API -------------------------- #
__all__ = [
    # Core (Detection/OBB)
    "DatasetProcessor",
    # Classification
    "ClassificationProcessor",
    "ClassificationConfig",
    "ClassificationField",
    "ClsAugmentationConfig",
    "ClsAugmentor",
    # Config
    "DatasetConfig",
    "TrainingConfig",
    # Constants
    "YOLO_CONSTANTS",
    "AUGMENTATION_PRESETS",
    # Helpers
    "YOLO_HELPERS",
    # Augmentation (OBB)
    "AugmentationConfig",
    "DataAugmentor",
]
