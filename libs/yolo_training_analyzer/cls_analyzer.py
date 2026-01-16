# ====== Code Summary ======
# Analyzer for YOLO Classification training results.
# Provides classification-specific logic to compute loss, accuracy metrics,
# learning curve behavior, and efficiency insights for training optimization.

from __future__ import annotations

# ====== Standard Library Imports ======
from pathlib import Path
from typing import Any, Dict, Tuple

# ====== Third-Party Library Imports ======
import numpy as np

# ====== Local Project Imports ======
from .base import BaseAnalyzer, TrainingMetrics
from .constants import ANALYZER_CONSTANTS


class ClsAnalyzer(BaseAnalyzer):
    """
    Analyzer for YOLO Classification training results.

    Handles metrics specific to image classification:
    - Training/Validation loss
    - Top-1 Accuracy
    - Top-5 Accuracy

    Usage:
        analyzer = ClsAnalyzer(Path("training_runs/cls_run"))
        summary = analyzer.get_summary()
        print(summary)
    """

    # Column mappings
    TRAIN_LOSS_COL: str = "train/loss"
    VAL_LOSS_COL: str = "val/loss"

    METRIC_COLS: dict[str, str] = {
        "Top-1 Accuracy": "metrics/accuracy_top1",
        "Top-5 Accuracy": "metrics/accuracy_top5",
    }

    # Primary metric for determining best epoch
    PRIMARY_METRIC: str = "metrics/accuracy_top1"
    PRIMARY_METRIC_NAME: str = "Top-1 Accuracy"

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize CLS analyzer for classification tasks.

        Args:
            run_dir (Path): Path to training run directory.
        """
        super().__init__(run_dir)

        # Warn if task mismatch
        if self.task != ANALYZER_CONSTANTS.TASK_CLS:
            print(f"Warning: Expected classify task, got {self.task}")

    def get_train_losses(self) -> dict[str, np.ndarray]:
        """
        Get training loss.

        Returns:
            dict[str, np.ndarray]: Dictionary with training loss curve.
        """
        values = self.get_column(self.TRAIN_LOSS_COL)
        if values is not None:
            return {"Loss": values}
        return {}

    def get_val_losses(self) -> dict[str, np.ndarray]:
        """
        Get validation loss.

        Returns:
            dict[str, np.ndarray]: Dictionary with validation loss curve.
        """
        values = self.get_column(self.VAL_LOSS_COL)
        if values is not None:
            return {"Loss": values}
        return {}

    def get_metrics(self) -> dict[str, np.ndarray]:
        """
        Get accuracy metrics.

        Returns:
            dict[str, np.ndarray]: Mapping from metric name to values.
        """
        metrics: dict[str, np.ndarray] = {}
        for name, col in self.METRIC_COLS.items():
            values = self.get_column(col)
            if values is not None:
                metrics[name] = values
        return metrics

    def get_best_epoch(self) -> Tuple[int, str, float]:
        """
        Get best epoch based on Top-1 Accuracy.

        Returns:
            tuple[int, str, float]: (epoch_index, metric_name, metric_value)
        """
        values = self.get_column(self.PRIMARY_METRIC)
        if values is None:
            return 0, self.PRIMARY_METRIC_NAME, 0.0

        best_idx = int(np.argmax(values))
        return best_idx, self.PRIMARY_METRIC_NAME, float(values[best_idx])

    def get_summary(self) -> TrainingMetrics:
        """
        Get training summary object.

        Returns:
            TrainingMetrics: Summary of classification training run.
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

    def get_accuracy_analysis(self) -> dict[str, Any]:
        """
        Analyze accuracy progression.

        Returns:
            dict[str, Any]: Summary of top-1 and top-5 accuracy dynamics.
        """
        top1 = self.get_column(self.METRIC_COLS["Top-1 Accuracy"])
        top5 = self.get_column(self.METRIC_COLS["Top-5 Accuracy"])

        if top1 is None:
            return {"error": "Accuracy data not found"}

        analysis: dict[str, Any] = {
            "top1": {
                "best": float(np.max(top1)),
                "best_epoch": int(np.argmax(top1)),
                "final": float(top1[-1]),
                "mean_last_10": float(np.mean(top1[-10:])) if len(top1) >= 10 else float(np.mean(top1)),
                "std_last_10": float(np.std(top1[-10:])) if len(top1) >= 10 else float(np.std(top1)),
                "improvement": float(top1[-1] - top1[0]),
            }
        }

        if top5 is not None:
            analysis["top5"] = {
                "best": float(np.max(top5)),
                "best_epoch": int(np.argmax(top5)),
                "final": float(top5[-1]),
            }

            # Gap between top1 and top5 can indicate class confusion
            gap = top5[-1] - top1[-1]
            analysis["top1_top5_gap"] = float(gap)
            analysis["gap_interpretation"] = (
                "Low gap - confident predictions" if gap < 0.05
                else "Moderate gap - some class confusion" if gap < 0.15
                else "High gap - significant class confusion"
            )

        return analysis

    def get_loss_analysis(self) -> dict[str, Any]:
        """
        Analyze loss progression.

        Returns:
            dict[str, Any]: Train/validation loss trends and generalization status.
        """
        train_loss = self.get_column(self.TRAIN_LOSS_COL)
        val_loss = self.get_column(self.VAL_LOSS_COL)

        analysis: dict[str, Any] = {}

        # 1. Training loss stats
        if train_loss is not None:
            analysis["train"] = {
                "initial": float(train_loss[0]),
                "final": float(train_loss[-1]),
                "min": float(np.min(train_loss)),
                "reduction": float(train_loss[0] - train_loss[-1]),
                "reduction_pct": float((train_loss[0] - train_loss[-1]) / train_loss[0] * 100),
            }

        # 2. Validation loss stats
        if val_loss is not None:
            analysis["val"] = {
                "initial": float(val_loss[0]),
                "final": float(val_loss[-1]),
                "min": float(np.min(val_loss)),
                "min_epoch": int(np.argmin(val_loss)),
            }

        # 3. Generalization gap
        if train_loss is not None and val_loss is not None:
            gap = val_loss[-1] - train_loss[-1]
            gap_ratio = gap / train_loss[-1] if train_loss[-1] > 0 else 0

            analysis["generalization"] = {
                "gap": float(gap),
                "gap_ratio": float(gap_ratio),
                "status": (
                    "excellent" if gap_ratio < 0.1
                    else "good" if gap_ratio < 0.3
                    else "concerning" if gap_ratio < 0.5
                    else "overfitting"
                ),
            }

        return analysis

    def get_learning_curve_analysis(self) -> dict[str, Any]:
        """
        Analyze the learning curve shape based on Top-1 accuracy.

        Returns:
            dict[str, Any]: Curve progression and training recommendation.
        """
        top1 = self.get_column(self.PRIMARY_METRIC)

        if top1 is None or len(top1) < 10:
            return {"error": "Not enough data for learning curve analysis"}

        # 1. Divide into four quarters
        q1 = np.mean(top1[:len(top1) // 4])
        q2 = np.mean(top1[len(top1) // 4:len(top1) // 2])
        q3 = np.mean(top1[len(top1) // 2:3 * len(top1) // 4])
        q4 = np.mean(top1[3 * len(top1) // 4:])

        # 2. Calculate gains
        early_gain = q2 - q1
        mid_gain = q3 - q2
        late_gain = q4 - q3

        # 3. Determine curve type and suggest actions
        if late_gain > mid_gain > early_gain:
            curve_type = "accelerating"
            recommendation = "Model still improving, consider more epochs"
        elif early_gain > mid_gain > late_gain and late_gain > 0:
            curve_type = "decelerating"
            recommendation = "Normal convergence pattern"
        elif late_gain < 0:
            curve_type = "degrading"
            recommendation = "Performance declining, possible overfitting"
        elif late_gain < 0.001:
            curve_type = "plateaued"
            recommendation = "Training converged, no need for more epochs"
        else:
            curve_type = "irregular"
            recommendation = "Unstable training, consider adjusting learning rate"

        return {
            "quarters": {
                "q1_mean": float(q1),
                "q2_mean": float(q2),
                "q3_mean": float(q3),
                "q4_mean": float(q4),
            },
            "gains": {
                "early": float(early_gain),
                "mid": float(mid_gain),
                "late": float(late_gain),
            },
            "curve_type": curve_type,
            "recommendation": recommendation,
        }

    def estimate_optimal_epochs(self) -> dict[str, Any]:
        """
        Estimate optimal epoch count based on performance saturation.

        Returns:
            dict[str, Any]: Epoch recommendations and efficiency notes.
        """
        best_epoch, _, best_value = self.get_best_epoch()
        top1 = self.get_column(self.PRIMARY_METRIC)

        if top1 is None:
            return {"error": "Accuracy data not found"}

        # 1. Find epoch reaching 95% of best accuracy
        threshold_95 = best_value * 0.95
        epoch_95 = int(np.argmax(top1 >= threshold_95))

        # 2. Find epoch reaching 99% of best accuracy
        threshold_99 = best_value * 0.99
        epoch_99 = int(np.argmax(top1 >= threshold_99)) if np.any(top1 >= threshold_99) else best_epoch

        return {
            "best_epoch": best_epoch,
            "best_accuracy": float(best_value),
            "epoch_95_percent": epoch_95,
            "epoch_99_percent": epoch_99,
            "efficiency_recommendation": (
                f"For 95% of best performance, {epoch_95} epochs suffice. "
                f"Best performance at epoch {best_epoch}."
            ),
            "epochs_after_best": self.total_epochs - best_epoch,
        }
