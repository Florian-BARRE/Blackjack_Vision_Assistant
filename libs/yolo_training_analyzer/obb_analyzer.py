# ====== Code Summary ======
# Analyzer for YOLO OBB (Oriented Bounding Box) training results.
# Extracts detection-specific metrics including losses, mAPs, precision-recall,
# and provides interpretation for convergence and performance stability.

from __future__ import annotations

# ====== Standard Library Imports ======
from pathlib import Path
from typing import Any, Dict, Tuple

# ====== Third-Party Library Imports ======
import numpy as np

# ====== Local Project Imports ======
from .base import BaseAnalyzer, TrainingMetrics
from .constants import ANALYZER_CONSTANTS


class ObbAnalyzer(BaseAnalyzer):
    """
    Analyzer for YOLO OBB training results.

    Handles metrics specific to object detection with oriented bounding boxes:
    - Box loss, class loss, DFL loss
    - mAP50, mAP50-95
    - Precision, Recall

    Usage:
        analyzer = ObbAnalyzer(Path("training_runs/obb_run"))
        summary = analyzer.get_summary()
        print(summary)
    """

    # Column mappings
    TRAIN_LOSS_COLS: dict[str, str] = {
        "Box Loss": "train/box_loss",
        "Class Loss": "train/cls_loss",
        "DFL Loss": "train/dfl_loss",
    }

    VAL_LOSS_COLS: dict[str, str] = {
        "Box Loss": "val/box_loss",
        "Class Loss": "val/cls_loss",
        "DFL Loss": "val/dfl_loss",
    }

    METRIC_COLS: dict[str, str] = {
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
    }

    # Primary metric for determining best epoch
    PRIMARY_METRIC: str = "metrics/mAP50(B)"
    PRIMARY_METRIC_NAME: str = "mAP50"

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize OBB analyzer.

        Args:
            run_dir (Path): Directory containing training artifacts.
        """
        super().__init__(run_dir)

        # Warn if task type is unexpected
        if self.task not in (ANALYZER_CONSTANTS.TASK_OBB, ANALYZER_CONSTANTS.TASK_DETECT):
            print(f"Warning: Expected OBB/detect task, got {self.task}")

    def get_train_losses(self) -> dict[str, np.ndarray]:
        """
        Get training loss curves.

        Returns:
            dict[str, np.ndarray]: Loss name mapped to its values.
        """
        losses: dict[str, np.ndarray] = {}
        for name, col in self.TRAIN_LOSS_COLS.items():
            values = self.get_column(col)
            if values is not None:
                losses[name] = values
        return losses

    def get_val_losses(self) -> dict[str, np.ndarray]:
        """
        Get validation loss curves.

        Returns:
            dict[str, np.ndarray]: Loss name mapped to its values.
        """
        losses: dict[str, np.ndarray] = {}
        for name, col in self.VAL_LOSS_COLS.items():
            values = self.get_column(col)
            if values is not None:
                losses[name] = values
        return losses

    def get_metrics(self) -> dict[str, np.ndarray]:
        """
        Get detection performance metrics.

        Returns:
            dict[str, np.ndarray]: Metric name mapped to its values.
        """
        metrics: dict[str, np.ndarray] = {}
        for name, col in self.METRIC_COLS.items():
            values = self.get_column(col)
            if values is not None:
                metrics[name] = values
        return metrics

    def get_best_epoch(self) -> Tuple[int, str, float]:
        """
        Determine the best epoch based on mAP50.

        Returns:
            tuple[int, str, float]: (epoch index, metric name, metric value)
        """
        values = self.get_column(self.PRIMARY_METRIC)
        if values is None:
            return 0, self.PRIMARY_METRIC_NAME, 0.0

        best_idx = int(np.argmax(values))
        return best_idx, self.PRIMARY_METRIC_NAME, float(values[best_idx])

    def get_summary(self) -> TrainingMetrics:
        """
        Get high-level training summary.

        Returns:
            TrainingMetrics: Object summarizing key metrics.
        """
        best_epoch, metric_name, metric_value = self.get_best_epoch()

        return TrainingMetrics(
            task=self.task,
            total_epochs=self.total_epochs,
            best_epoch=best_epoch,
            best_metric_name=metric_name,
            best_metric_value=metric_value,
            final_metrics=self.get_final_metrics(),
            model_name=self.model_name,
        )

    def get_map_analysis(self) -> dict[str, Any]:
        """
        Analyze mAP50 and mAP50-95 progression.

        Returns:
            dict[str, Any]: Trend analysis of mAP metrics.
        """
        map50 = self.get_column(self.METRIC_COLS["mAP50"])
        map50_95 = self.get_column(self.METRIC_COLS["mAP50-95"])

        if map50 is None:
            return {"error": "mAP50 data not found"}

        analysis: dict[str, Any] = {
            "mAP50": {
                "best": float(np.max(map50)),
                "best_epoch": int(np.argmax(map50)),
                "final": float(map50[-1]),
                "mean_last_10": float(np.mean(map50[-10:])) if len(map50) >= 10 else float(np.mean(map50)),
                "std_last_10": float(np.std(map50[-10:])) if len(map50) >= 10 else float(np.std(map50)),
            }
        }

        if map50_95 is not None:
            analysis["mAP50-95"] = {
                "best": float(np.max(map50_95)),
                "best_epoch": int(np.argmax(map50_95)),
                "final": float(map50_95[-1]),
                "mean_last_10": float(np.mean(map50_95[-10:])) if len(map50_95) >= 10 else float(np.mean(map50_95)),
            }

        if len(map50) >= 20:
            first_half = np.mean(map50[:len(map50) // 2])
            second_half = np.mean(map50[len(map50) // 2:])
            analysis["improvement"] = float(second_half - first_half)

        return analysis

    def get_loss_analysis(self) -> dict[str, Any]:
        """
        Analyze training and validation loss trends.

        Returns:
            dict[str, Any]: Initial/final loss values and reduction stats.
        """
        train_losses = self.get_train_losses()
        val_losses = self.get_val_losses()

        analysis: dict[str, Any] = {"train": {}, "val": {}}

        # Analyze training losses
        for name, values in train_losses.items():
            analysis["train"][name] = {
                "initial": float(values[0]),
                "final": float(values[-1]),
                "min": float(np.min(values)),
                "reduction": float(values[0] - values[-1]),
            }

        # Analyze validation losses
        for name, values in val_losses.items():
            analysis["val"][name] = {
                "initial": float(values[0]),
                "final": float(values[-1]),
                "min": float(np.min(values)),
                "min_epoch": int(np.argmin(values)),
            }

        return analysis

    def get_precision_recall_analysis(self) -> dict[str, Any]:
        """
        Analyze precision and recall metrics with derived F1 score.

        Returns:
            dict[str, Any]: Best/final values and balance check.
        """
        precision = self.get_column(self.METRIC_COLS["Precision"])
        recall = self.get_column(self.METRIC_COLS["Recall"])

        if precision is None or recall is None:
            return {"error": "Precision/Recall data not found"}

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            "precision": {
                "best": float(np.max(precision)),
                "final": float(precision[-1]),
            },
            "recall": {
                "best": float(np.max(recall)),
                "final": float(recall[-1]),
            },
            "f1": {
                "best": float(np.max(f1)),
                "best_epoch": int(np.argmax(f1)),
                "final": float(f1[-1]),
            },
            "balance": "good" if abs(precision[-1] - recall[-1]) < 0.1 else "imbalanced",
        }
