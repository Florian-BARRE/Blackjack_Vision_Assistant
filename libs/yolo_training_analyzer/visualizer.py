# ====== Code Summary ======
# Visualization utilities for YOLO training analysis.
# Provides plots for losses, metrics, learning rates, convergence trends,
# and comparison dashboards using matplotlib.

from __future__ import annotations

# ====== Standard Library Imports ======
from pathlib import Path
from typing import Any, Optional

# ====== Third-Party Library Imports ======
import matplotlib.pyplot as plt
import numpy as np

# ====== Local Project Imports ======
from .base import BaseAnalyzer


class TrainingVisualizer:
    """
    Visualization class for training analysis.

    Provides methods to plot:
    - Loss curves (train/val)
    - Metric curves (mAP, accuracy)
    - Learning rate schedules
    - Combined dashboards
    """

    COLORS: dict[str, str] = {
        "train": "#2563eb",  # Blue
        "val": "#dc2626",  # Red
        "metric1": "#16a34a",  # Green
        "metric2": "#f59e0b",  # Amber
        "metric3": "#8b5cf6",  # Purple
        "lr": "#6b7280",  # Gray
        "best": "#ef4444",  # Red for best epoch
    }

    def __init__(self, analyzer: BaseAnalyzer) -> None:
        """
        Initialize visualizer with a BaseAnalyzer instance.

        Args:
            analyzer (BaseAnalyzer): Analyzer with training data.
        """
        self.analyzer = analyzer
        self.epochs = analyzer.epochs

    def plot_losses(
            self,
            figsize: tuple[int, int] = (12, 5),
            title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation loss curves.

        Args:
            figsize (tuple[int, int]): Size of the plot figure.
            title (Optional[str]): Optional plot title.

        Returns:
            plt.Figure: The generated figure.
        """
        train_losses = self.analyzer.get_train_losses()
        val_losses = self.analyzer.get_val_losses()

        n_plots = len(train_losses)
        if n_plots == 0:
            raise ValueError("No loss data found")

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for ax, (name, train_values) in zip(axes, train_losses.items()):
            ax.plot(self.epochs, train_values,
                    color=self.COLORS["train"], label="Train", linewidth=2)

            if name in val_losses:
                ax.plot(self.epochs, val_losses[name],
                        color=self.COLORS["val"], label="Val", linewidth=2)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title or "Training Losses", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_metrics(
            self,
            figsize: tuple[int, int] = (12, 5),
            title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training metrics over epochs.

        Args:
            figsize (tuple[int, int]): Figure dimensions.
            title (Optional[str]): Optional plot title.

        Returns:
            plt.Figure: Generated matplotlib figure.
        """
        metrics = self.analyzer.get_metrics()
        if not metrics:
            raise ValueError("No metrics data found")

        n_plots = len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        colors = list(self.COLORS.values())
        best_epoch, _, _ = self.analyzer.get_best_epoch()

        for ax, ((name, values), color) in zip(axes, zip(metrics.items(), colors[2:])):
            ax.plot(self.epochs, values, color=color, linewidth=2, label=name)
            ax.axvline(x=best_epoch, color=self.COLORS["best"],
                       linestyle="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
            ax.scatter([best_epoch], [values[best_epoch]],
                       color=self.COLORS["best"], s=100, zorder=5)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title or "Training Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_learning_rate(
            self,
            figsize: tuple[int, int] = (10, 4),
            title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning rate over epochs.

        Args:
            figsize (tuple[int, int]): Figure dimensions.
            title (Optional[str]): Optional title.

        Returns:
            plt.Figure: Learning rate plot.
        """
        lr_data = self.analyzer.get_learning_rates()
        if not lr_data:
            raise ValueError("No learning rate data found")

        fig, ax = plt.subplots(figsize=figsize)
        for name, values in lr_data.items():
            ax.plot(self.epochs, values, linewidth=2, label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title or "Learning Rate Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        return fig

    def plot_convergence_analysis(
            self,
            figsize: tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Plot rolling averages and cumulative changes for convergence tracking.

        Args:
            figsize (tuple[int, int]): Figure size.

        Returns:
            plt.Figure: The generated convergence figure.
        """
        metrics = self.analyzer.get_metrics()
        if not metrics:
            raise ValueError("No metrics available for convergence plot")

        metric_name = list(metrics.keys())[0]
        values = metrics[metric_name]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Rolling mean
        ax1 = axes[0]
        window = min(10, len(values) // 5)
        if window > 1:
            rolling = np.convolve(values, np.ones(window) / window, mode="valid")
            ax1.plot(self.epochs, values, label="Raw", alpha=0.4)
            ax1.plot(self.epochs[window - 1:], rolling, label="Rolling Mean", linewidth=2)
        else:
            ax1.plot(self.epochs, values, linewidth=2)
        ax1.set_title("Metric with Smoothing")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(metric_name)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Delta plot
        ax2 = axes[1]
        delta = np.diff(values)
        ax2.bar(self.epochs[1:], delta, width=0.8,
                color=np.where(delta > 0, self.COLORS["metric1"], self.COLORS["val"]))
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_title("Epoch-to-Epoch Change")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Change")
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative
        ax3 = axes[2]
        cumulative = values - values[0]
        ax3.plot(self.epochs, cumulative, linewidth=2, color=self.COLORS["metric1"])
        ax3.fill_between(self.epochs, 0, cumulative, alpha=0.3, color=self.COLORS["metric1"])
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("Cumulative Improvement")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Gain")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def create_comparison_plot(
        analyzers: list[BaseAnalyzer],
        metric_name: str,
        figsize: tuple[int, int] = (12, 5),
        title: Optional[str] = None
) -> plt.Figure:
    """
    Compare a specific metric across multiple analyzers.

    Args:
        analyzers (list[BaseAnalyzer]): list of analyzer instances.
        metric_name (str): Metric to plot.
        figsize (tuple[int, int]): Plot dimensions.
        title (Optional[str]): Plot title.

    Returns:
        plt.Figure: Matplotlib figure for comparison.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for analyzer in analyzers:
        metrics = analyzer.get_metrics()
        if metric_name in metrics:
            ax.plot(analyzer.epochs, metrics[metric_name],
                    label=analyzer.run_dir.name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Comparison: {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
