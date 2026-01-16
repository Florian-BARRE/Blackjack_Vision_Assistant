# ====== Code Summary ======
# Defines constants for YOLO dataset processing, including supported formats,
# default paths, augmentation presets, and task types.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


class YOLO_CONSTANTS:
    """
    Constants used for YOLO dataset processing, configuration management,
    and supported formats.
    """

    # ----- Supported image extensions -----
    SUPPORTED_IMAGE_EXTS: set[str] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }

    # ----- Supported YOLO task types -----
    TASK_DETECT: str = "detect"
    TASK_OBB: str = "obb"
    TASK_SEGMENT: str = "segment"
    TASK_POSE: str = "pose"
    TASK_CLASSIFY: str = "classify"

    # Tasks with label files (detection-style)
    DETECTION_TASKS: set[str] = {TASK_DETECT, TASK_OBB, TASK_SEGMENT, TASK_POSE}

    # Tasks with folder structure (classification-style)
    CLASSIFICATION_TASKS: set[str] = {TASK_CLASSIFY}

    SUPPORTED_TASKS: set[str] = DETECTION_TASKS | CLASSIFICATION_TASKS

    # ----- Default filenames -----
    DEFAULT_DATA_YAML: str = "data.yaml"
    DEFAULT_HYP_YAML: str = "hyp.yaml"
    DEFAULT_TRAIN_CFG_YAML: str = "train_cfg.yaml"
    DEFAULT_CLASSES_FILE: str = "classes.txt"
    DEFAULT_CONFIG_FILENAME: str = "dataset_config.json"

    # ----- Default paths -----
    DEFAULT_CONFIG_PATH: Path = Path("../libs/yolo_dataset/configs")

    # ----- Label format -----
    # Standard YOLO: cls x_center y_center width height (5 values)
    # OBB YOLO: cls x1 y1 x2 y2 x3 y3 x4 y4 (9 values)
    LABEL_VALUES_DETECT: int = 5
    LABEL_VALUES_OBB: int = 9

    # ----- Default split ratio -----
    DEFAULT_VAL_SPLIT: float = 0.15
    DEFAULT_TEST_SPLIT: float = 0.0

    # ----- Default training settings -----
    DEFAULT_IMG_SIZE: int = 640
    DEFAULT_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_WORKERS: int = 4
    DEFAULT_PATIENCE: int = 50

    # ----- Default model weights -----
    DEFAULT_MODELS: Dict[str, str] = {
        TASK_DETECT: "yolo11n.pt",
        TASK_OBB: "yolo11n-obb.pt",
        TASK_SEGMENT: "yolo11n-seg.pt",
        TASK_POSE: "yolo11n-pose.pt",
        TASK_CLASSIFY: "yolo11n-cls.pt",
    }

    # ----- Classification specific -----
    DEFAULT_CLS_IMG_SIZE: int = 224
    AUG_SEPARATOR: str = "_aug"  # For grouping augmented variants


class AUGMENTATION_PRESETS:
    """
    Predefined augmentation configurations for different use cases.
    """

    # ----- Minimal augmentation -----
    MINIMAL: Dict[str, Any] = {
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    # ----- Standard augmentation -----
    STANDARD: Dict[str, Any] = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    }

    # ----- Strong augmentation (for OBB) -----
    STRONG_OBB: Dict[str, Any] = {
        "hsv_h": 0.015,
        "hsv_s": 0.70,
        "hsv_v": 0.40,
        "degrees": 10.0,
        "translate": 0.10,
        "scale": 0.50,
        "shear": 8.0,
        "perspective": 0.001,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    }

    # ----- Heavy augmentation -----
    HEAVY: Dict[str, Any] = {
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 15.0,
        "translate": 0.2,
        "scale": 0.9,
        "shear": 10.0,
        "perspective": 0.001,
        "fliplr": 0.5,
        "flipud": 0.1,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.3,
        "close_mosaic": 15,
    }

    # ----- Classification augmentation -----
    CLASSIFICATION: Dict[str, Any] = {
        "hsv_h": 0.015,
        "hsv_s": 0.50,
        "hsv_v": 0.30,
        "degrees": 15.0,
        "translate": 0.15,
        "scale": 0.40,
        "shear": 5.0,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.0,  # No mosaic for classification
        "mixup": 0.60,
        "copy_paste": 0.70,
        "erasing": 0.4,
        "auto_augment": "randaugment",
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """
        Get an augmentation preset by name.

        Args:
            name: Preset name ('minimal', 'standard', 'strong_obb', 'heavy')

        Returns:
            Dictionary of augmentation parameters.

        Raises:
            ValueError: If preset name is not recognized.
        """
        presets = {
            "minimal": cls.MINIMAL,
            "standard": cls.STANDARD,
            "strong_obb": cls.STRONG_OBB,
            "heavy": cls.HEAVY,
            "classification": cls.CLASSIFICATION,
        }
        if name.lower() not in presets:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(presets.keys())}"
            )
        return presets[name.lower()].copy()
