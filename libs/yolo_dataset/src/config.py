# ====== Code Summary ======
# Defines configuration dataclasses for YOLO dataset processing.
# Includes DatasetConfig for dataset settings and TrainingConfig for training parameters.

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import YOLO_CONSTANTS, AUGMENTATION_PRESETS


@dataclass
class DatasetConfig:
    """
    Configuration for a YOLO dataset.

    Attributes:
        task: YOLO task type ('detect', 'obb', 'segment', 'pose')
        class_names: List of class names
        val_split: Validation split ratio (0.0 to 1.0)
        test_split: Test split ratio (0.0 to 1.0)
        seed: Random seed for reproducibility
    """

    task: str = YOLO_CONSTANTS.TASK_DETECT
    class_names: List[str] = field(default_factory=list)
    val_split: float = YOLO_CONSTANTS.DEFAULT_VAL_SPLIT
    test_split: float = YOLO_CONSTANTS.DEFAULT_TEST_SPLIT
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.task not in YOLO_CONSTANTS.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{self.task}'. "
                f"Supported: {YOLO_CONSTANTS.SUPPORTED_TASKS}"
            )
        if not 0.0 <= self.val_split <= 1.0:
            raise ValueError(f"val_split must be in [0, 1], got {self.val_split}")
        if not 0.0 <= self.test_split <= 1.0:
            raise ValueError(f"test_split must be in [0, 1], got {self.test_split}")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_names)

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration to a JSON-compatible dictionary.

        Returns:
            Dictionary with configuration and metadata.
        """
        return {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": self.task,
            "class_names": list(self.class_names),
            "val_split": self.val_split,
            "test_split": self.test_split,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Target file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @staticmethod
    def load(path: Path) -> DatasetConfig:
        """
        Load a DatasetConfig from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed DatasetConfig object.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return DatasetConfig(
            task=data.get("task", YOLO_CONSTANTS.TASK_DETECT),
            class_names=data.get("class_names", []),
            val_split=data.get("val_split", YOLO_CONSTANTS.DEFAULT_VAL_SPLIT),
            test_split=data.get("test_split", YOLO_CONSTANTS.DEFAULT_TEST_SPLIT),
            seed=data.get("seed", 42),
        )


@dataclass
class TrainingConfig:
    """
    Configuration for YOLO model training.

    Attributes:
        model: Model weights file (e.g., 'yolo11n.pt')
        task: YOLO task type
        img_size: Input image size
        epochs: Number of training epochs
        batch_size: Batch size
        device: Training device (0 for GPU, 'cpu' for CPU)
        workers: Number of data loading workers
        patience: Early stopping patience
        optimizer: Optimizer type
        lr0: Initial learning rate
        lrf: Final learning rate factor
        momentum: SGD momentum
        weight_decay: Weight decay
        project: Project name for runs
        run_name: Run name
        augmentation: Augmentation parameters dictionary
    """

    model: str = "yolo11n.pt"
    task: str = YOLO_CONSTANTS.TASK_DETECT
    img_size: int = YOLO_CONSTANTS.DEFAULT_IMG_SIZE
    epochs: int = YOLO_CONSTANTS.DEFAULT_EPOCHS
    batch_size: int = YOLO_CONSTANTS.DEFAULT_BATCH_SIZE
    device: Any = 0
    workers: int = YOLO_CONSTANTS.DEFAULT_WORKERS
    patience: int = YOLO_CONSTANTS.DEFAULT_PATIENCE
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    project: str = "training_runs"
    run_name: str = "yolo_run"
    augmentation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(
        cls,
        task: str = YOLO_CONSTANTS.TASK_DETECT,
        aug_preset: str = "standard",
        **kwargs
    ) -> TrainingConfig:
        """
        Create a TrainingConfig from an augmentation preset.

        Args:
            task: YOLO task type.
            aug_preset: Augmentation preset name.
            **kwargs: Override any default parameters.

        Returns:
            Configured TrainingConfig instance.
        """
        model = kwargs.pop("model", YOLO_CONSTANTS.DEFAULT_MODELS.get(task, "yolo11n.pt"))
        augmentation = AUGMENTATION_PRESETS.get_preset(aug_preset)

        return cls(
            model=model,
            task=task,
            augmentation=augmentation,
            **kwargs
        )

    def to_yaml_dict(self, data_yaml_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Convert to a dictionary suitable for YOLO training YAML.

        Args:
            data_yaml_path: Path to data.yaml file.

        Returns:
            Dictionary ready for YAML export.
        """
        cfg = {
            "task": self.task,
            "mode": "train",
            "model": self.model,
            "imgsz": self.img_size,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "device": self.device,
            "seed": 42,
            "workers": self.workers,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "project": self.project,
            "name": self.run_name,
            "exist_ok": True,
            "save": True,
            "plots": True,
            "verbose": True,
        }

        if data_yaml_path:
            cfg["data"] = str(data_yaml_path)

        # Merge augmentation parameters
        cfg.update(self.augmentation)

        return cfg

    def save(self, path: Path, data_yaml_path: Optional[Path] = None) -> None:
        """
        Save training configuration to a YAML file.

        Args:
            path: Target file path.
            data_yaml_path: Path to data.yaml file to include.
        """
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_yaml_dict(data_yaml_path), f, sort_keys=False)

    @staticmethod
    def load(path: Path) -> TrainingConfig:
        """
        Load a TrainingConfig from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Reconstructed TrainingConfig object.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract known fields
        known_fields = {
            "model", "task", "imgsz", "epochs", "batch", "device",
            "workers", "patience", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "project", "name"
        }

        # Separate augmentation parameters
        augmentation = {k: v for k, v in data.items() if k not in known_fields}

        return TrainingConfig(
            model=data.get("model", "yolo11n.pt"),
            task=data.get("task", YOLO_CONSTANTS.TASK_DETECT),
            img_size=data.get("imgsz", YOLO_CONSTANTS.DEFAULT_IMG_SIZE),
            epochs=data.get("epochs", YOLO_CONSTANTS.DEFAULT_EPOCHS),
            batch_size=data.get("batch", YOLO_CONSTANTS.DEFAULT_BATCH_SIZE),
            device=data.get("device", 0),
            workers=data.get("workers", YOLO_CONSTANTS.DEFAULT_WORKERS),
            patience=data.get("patience", YOLO_CONSTANTS.DEFAULT_PATIENCE),
            optimizer=data.get("optimizer", "auto"),
            lr0=data.get("lr0", 0.01),
            lrf=data.get("lrf", 0.01),
            momentum=data.get("momentum", 0.937),
            weight_decay=data.get("weight_decay", 0.0005),
            project=data.get("project", "training_runs"),
            run_name=data.get("name", "yolo_run"),
            augmentation=augmentation,
        )
