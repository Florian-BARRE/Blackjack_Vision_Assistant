# ====== Code Summary ======
# Constants for YOLO training analysis module.
# Provides task identifiers, metric names, plot colors, and shared config for analyzers.

# ====== Standard Library Imports ======
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AnalyzerConstants:
    """
    Constants for training analysis.

    Includes metric names, task types, and visualization configuration
    used across analyzer implementations.
    """

    # Task types
    TASK_OBB: str = "obb"
    TASK_CLS: str = "classify"
    TASK_DETECT: str = "detect"
    TASK_SEGMENT: str = "segment"

    # Common metrics
    EPOCH_COL: str = "epoch"

    # OBB/Detection metrics
    OBB_METRICS: tuple[str, ...] = (
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )

    # Classification metrics
    CLS_METRICS: tuple[str, ...] = (
        "train/loss",
        "val/loss",
        "metrics/accuracy_top1",
        "metrics/accuracy_top5",
    )

    # Learning rate columns
    LR_COLS: tuple[str, ...] = (
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
    )

    # Key metrics for summary (set via __post_init__)
    OBB_KEY_METRICS: Dict[str, str] = None
    CLS_KEY_METRICS: Dict[str, str] = None

    # Plot color palette (set via __post_init__)
    COLORS: Dict[str, str] = None

    def __post_init__(self) -> None:
        """
        Populate mutable default fields for a frozen dataclass.
        """
        object.__setattr__(self, "OBB_KEY_METRICS", {
            "mAP50": "metrics/mAP50(B)",
            "mAP50-95": "metrics/mAP50-95(B)",
            "Precision": "metrics/precision(B)",
            "Recall": "metrics/recall(B)",
        })

        object.__setattr__(self, "CLS_KEY_METRICS", {
            "Top-1 Accuracy": "metrics/accuracy_top1",
            "Top-5 Accuracy": "metrics/accuracy_top5",
        })

        object.__setattr__(self, "COLORS", {
            "train": "#2563eb",  # Blue
            "val": "#dc2626",  # Red
            "metric": "#16a34a",  # Green
            "lr": "#9333ea",  # Purple
            "best": "#f59e0b",  # Amber
        })


# Global constants instance
ANALYZER_CONSTANTS = AnalyzerConstants()
