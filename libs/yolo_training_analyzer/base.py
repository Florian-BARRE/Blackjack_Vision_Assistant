# ====== Code Summary ======
# Base analyzer class for processing YOLO training results.
# Provides shared functionality for subclasses to extract and analyze metrics,
# detect tasks, check convergence, and evaluate overfitting behavior.

from __future__ import annotations

# ====== Standard Library Imports ======
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ====== Third-Party Library Imports ======
import numpy as np
import pandas as pd
import yaml

# ====== Local Project Imports ======
from .constants import ANALYZER_CONSTANTS


@dataclass
class TrainingMetrics:
    """
    Container for storing the summary of training metrics.

    Attributes:
        task (str): Type of task (e.g., detection, classification).
        total_epochs (int): Total number of epochs trained.
        best_epoch (int): Epoch number with the best metric.
        best_metric_name (str): Name of the best metric.
        best_metric_value (float): Value of the best metric.
        final_metrics (dict): Final values of all metrics at last epoch.
        training_time (float | None): Total training time, if available.
        model_name (str | None): Name of the model architecture.
    """
    task: str
    total_epochs: int
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    final_metrics: dict[str, float]
    training_time: float | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the TrainingMetrics object into a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of training metrics.
        """
        return {
            "task": self.task,
            "total_epochs": self.total_epochs,
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "final_metrics": self.final_metrics,
            "training_time": self.training_time,
            "model_name": self.model_name,
        }

    def __str__(self) -> str:
        """
        Generate a human-readable string of training metrics.

        Returns:
            str: String summary of the training metrics.
        """
        lines = [
            f"Task: {self.task}",
            f"Model: {self.model_name or 'Unknown'}",
            f"Total epochs: {self.total_epochs}",
            f"Best epoch: {self.best_epoch}",
            f"Best {self.best_metric_name}: {self.best_metric_value:.4f}",
            "",
            "Final metrics:",
        ]
        for name, value in self.final_metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class BaseAnalyzer(ABC):
    """
    Abstract base class for analyzing YOLO training results.

    Handles loading results, extracting metadata, and shared analysis logic.
    Subclasses must implement metric/loss extraction methods.
    """

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize analyzer with a training run directory.

        Args:
            run_dir (Path): Directory containing results.csv and other artifacts.
        """
        self.run_dir: Path = Path(run_dir)
        self._validate_run_dir()

        # 1. Load results and training args
        self.results_df: pd.DataFrame = self._load_results()
        self.args: dict[str, Any] = self._load_args()

        # 2. Detect training task type
        self.task: str = self._detect_task()

    def _validate_run_dir(self) -> None:
        """
        Ensure run directory and results.csv file exist.

        Raises:
            FileNotFoundError: If the directory or required file is missing.
        """
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        results_csv = self.run_dir / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(f"results.csv not found in {self.run_dir}")

    def _load_results(self) -> pd.DataFrame:
        """
        Load results.csv into a cleaned DataFrame.

        Returns:
            pd.DataFrame: Cleaned training results.
        """
        csv_path = self.run_dir / "results.csv"
        df = pd.read_csv(csv_path)

        # 1. Strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df

    def _load_args(self) -> dict[str, Any]:
        """
        Load training arguments from args.yaml.

        Returns:
            dict[str, Any]: Parsed YAML configuration.
        """
        args_path = self.run_dir / "args.yaml"
        if args_path.exists():
            with open(args_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _detect_task(self) -> str:
        """
        Detect task type from training args or known result column names.

        Returns:
            str: Detected task type string.
        """
        # 1. Prefer explicit task from args
        if "task" in self.args:
            return self.args["task"]

        # 2. Infer from known metric columns
        cols = set(self.results_df.columns)
        if "metrics/accuracy_top1" in cols:
            return ANALYZER_CONSTANTS.TASK_CLS
        elif "metrics/mAP50(B)" in cols:
            return ANALYZER_CONSTANTS.TASK_OBB

        return "unknown"

    @property
    def epochs(self) -> np.ndarray:
        """
        Epoch numbers array from results.

        Returns:
            np.ndarray: Array of epoch indices.
        """
        if ANALYZER_CONSTANTS.EPOCH_COL in self.results_df.columns:
            return self.results_df[ANALYZER_CONSTANTS.EPOCH_COL].values
        return np.arange(len(self.results_df))

    @property
    def total_epochs(self) -> int:
        """
        Total number of training epochs.

        Returns:
            int: Number of rows in results DataFrame.
        """
        return len(self.results_df)

    @property
    def model_name(self) -> str | None:
        """
        Model name extracted from training args.

        Returns:
            str | None: Name of the model.
        """
        return self.args.get("model")

    @property
    def weights_dir(self) -> Path:
        """
        Directory containing model weights.

        Returns:
            Path: Path to weights/ subdirectory.
        """
        return self.run_dir / "weights"

    @property
    def best_weights(self) -> Path | None:
        """
        Path to best.pt model weights.

        Returns:
            Path | None: Path if exists, else None.
        """
        best_pt = self.weights_dir / "best.pt"
        return best_pt if best_pt.exists() else None

    @property
    def last_weights(self) -> Path | None:
        """
        Path to last.pt model weights.

        Returns:
            Path | None: Path if exists, else None.
        """
        last_pt = self.weights_dir / "last.pt"
        return last_pt if last_pt.exists() else None

    def get_column(self, col_name: str) -> np.ndarray | None:
        """
        Retrieve a column from results as a numpy array.

        Args:
            col_name (str): Name of column to extract.

        Returns:
            np.ndarray | None: Array of values or None if not found.
        """
        if col_name in self.results_df.columns:
            return self.results_df[col_name].values
        return None

    def get_learning_rates(self) -> dict[str, np.ndarray]:
        """
        Extract all learning rate columns from results.

        Returns:
            dict[str, np.ndarray]: Mapping of LR column names to arrays.
        """
        lr_data: dict[str, np.ndarray] = {}
        for col in ANALYZER_CONSTANTS.LR_COLS:
            values = self.get_column(col)
            if values is not None:
                lr_data[col] = values
        return lr_data

    @abstractmethod
    def get_train_losses(self) -> dict[str, np.ndarray]:
        """
        Abstract method to extract training loss values.

        Returns:
            dict[str, np.ndarray]: Mapping of loss names to values.
        """
        ...

    @abstractmethod
    def get_val_losses(self) -> dict[str, np.ndarray]:
        """
        Abstract method to extract validation loss values.

        Returns:
            dict[str, np.ndarray]: Mapping of loss names to values.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, np.ndarray]:
        """
        Abstract method to extract metric values.

        Returns:
            dict[str, np.ndarray]: Mapping of metric names to arrays.
        """
        ...

    @abstractmethod
    def get_best_epoch(self) -> Tuple[int, str, float]:
        """
        Abstract method to determine best epoch and associated metric.

        Returns:
            tuple[int, str, float]: (epoch, metric_name, value)
        """
        ...

    @abstractmethod
    def get_summary(self) -> TrainingMetrics:
        """
        Abstract method to return training summary object.

        Returns:
            TrainingMetrics: Summary of training run.
        """
        ...

    def get_final_metrics(self) -> dict[str, float]:
        """
        Get metrics from the final epoch.

        Returns:
            dict[str, float]: Final values for each metric.
        """
        metrics = self.get_metrics()
        return {name: values[-1] for name, values in metrics.items()}

    def get_best_metrics(self) -> dict[str, float]:
        """
        Get metrics from the best epoch.

        Returns:
            dict[str, float]: Metric values at best epoch.
        """
        best_epoch, _, _ = self.get_best_epoch()
        metrics = self.get_metrics()
        return {name: values[best_epoch] for name, values in metrics.items()}

    def has_converged(self, patience: int = 10, min_delta: float = 0.001) -> bool:
        """
        Determine if the model has converged (i.e., no recent improvements).

        Args:
            patience (int): Epochs to wait for improvement.
            min_delta (float): Minimum required improvement.

        Returns:
            bool: True if converged.
        """
        best_epoch, _, _ = self.get_best_epoch()
        return (self.total_epochs - best_epoch) >= patience

    def get_overfitting_indicator(self) -> dict[str, Any]:
        """
        Detect potential overfitting trends in training vs. validation losses.

        Returns:
            dict[str, Any]: Overfitting status, message, and diagnostics.
        """
        train_losses = self.get_train_losses()
        val_losses = self.get_val_losses()

        # 1. Get primary loss arrays
        train_loss = list(train_losses.values())[0] if train_losses else None
        val_loss = list(val_losses.values())[0] if val_losses else None

        # 2. Validate existence
        if train_loss is None or val_loss is None:
            return {"status": "unknown", "message": "Missing loss data"}

        # 3. Define analysis window
        n = min(10, len(train_loss) // 4)
        if n < 3:
            return {"status": "unknown", "message": "Not enough epochs"}

        # 4. Fit linear trends to recent losses
        train_trend = np.polyfit(range(n), train_loss[-n:], 1)[0]
        val_trend = np.polyfit(range(n), val_loss[-n:], 1)[0]

        # 5. Measure gap growth
        gap = val_loss[-1] - train_loss[-1]
        gap_start = val_loss[0] - train_loss[0]
        gap_growth = gap - gap_start

        # 6. Analyze conditions
        if train_trend < 0 and val_trend > 0:
            return {
                "status": "overfitting",
                "message": "Training loss decreasing while validation loss increasing",
                "train_trend": train_trend,
                "val_trend": val_trend,
                "gap_growth": gap_growth,
            }
        elif gap_growth > 0.1:
            return {
                "status": "warning",
                "message": "Growing gap between train and val loss",
                "gap_growth": gap_growth,
            }
        else:
            return {
                "status": "healthy",
                "message": "No significant overfitting detected",
                "gap": gap,
            }
