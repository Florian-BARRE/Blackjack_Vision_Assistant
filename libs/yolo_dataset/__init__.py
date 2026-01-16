# ====== Code Summary ======
# YOLO Dataset Processing Module
#
# A comprehensive toolkit for processing Label Studio exports into
# training-ready datasets for Ultralytics YOLO models.
#
# Supports:
# - Detection (YOLO format with bounding boxes)
# - Oriented Bounding Boxes (OBB)
# - Classification (JSON-MIN format with folder structure)
#
# Features:
# - Auto-detection of task type (detect, OBB, classify)
# - Train/val/test splitting with reproducible seeds
# - Label normalization and validation
# - Configuration file generation (data.yaml, hyp.yaml, train_cfg.yaml)
# - Offline data augmentation with label adjustment
# - Multi-field classification support
#
# Typical usage for Detection/OBB:
#     from yolo_dataset import DatasetProcessor
#
#     processor = DatasetProcessor(Path("./my_dataset"))
#     processor.process()
#     print(processor.get_training_command())
#
# Typical usage for Classification:
#     from yolo_dataset import ClassificationProcessor, ClassificationConfig, ClassificationField
#
#     config = ClassificationConfig(
#         fields=[
#             ClassificationField("rank", ["A", "2", ...], "rank_dataset"),
#             ClassificationField("suit", ["Heart", ...], "suit_dataset"),
#         ]
#     )
#     processor = ClassificationProcessor(
#         json_path=Path("export.json"),
#         images_dir=Path("images"),
#         output_dir=Path("output"),
#         config=config
#     )
#     processor.process()

# --------------------- Public Interface --------------------- #
from .src import (
    # Core (Detection/OBB)
    DatasetProcessor,
    # Classification
    ClassificationProcessor,
    ClassificationConfig,
    ClassificationField,
    ClsAugmentationConfig,
    ClsAugmentor,
    # Config
    DatasetConfig,
    TrainingConfig,
    # Constants
    YOLO_CONSTANTS,
    AUGMENTATION_PRESETS,
    # Helpers
    YOLO_HELPERS,
    # Augmentation (OBB)
    AugmentationConfig,
    DataAugmentor,
)

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
