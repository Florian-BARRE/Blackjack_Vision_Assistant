# ------------------ Local Project Imports ------------------ #
from .constants import ANALYZER_CONSTANTS
from .base import BaseAnalyzer, TrainingMetrics
from .obb_analyzer import ObbAnalyzer
from .cls_analyzer import ClsAnalyzer
from .visualizer import TrainingVisualizer, create_comparison_plot

# -------------------- Public API -------------------- #
__all__ = [
    "ANALYZER_CONSTANTS",
    "BaseAnalyzer",
    "TrainingMetrics",
    "ObbAnalyzer",
    "ClsAnalyzer",
    "TrainingVisualizer",
    "create_comparison_plot",
    "analyze",
]


def analyze(run_dir, task: str = "auto") -> BaseAnalyzer:
    """
    Factory function to create the appropriate analyzer.

    Automatically detects task type from results.csv when task='auto'.

    Args:
        run_dir (str | Path): Path to training run directory.
        task (str): Task type ('obb', 'cls', 'auto').

    Returns:
        BaseAnalyzer: An instance of ClsAnalyzer or ObbAnalyzer.

    Usage:
        analyzer = analyze("training_runs/obb_run")
        summary = analyzer.get_summary()
    """
    from pathlib import Path
    import pandas as pd

    run_dir = Path(run_dir)

    if task == "auto":
        # Auto-detect from results.csv columns
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, nrows=1)
            cols = set(df.columns.str.strip())

            if "metrics/accuracy_top1" in cols:
                task = "cls"
            else:
                task = "obb"
        else:
            task = "obb"

    if task in ("cls", "classify"):
        return ClsAnalyzer(run_dir)
    return ObbAnalyzer(run_dir)
