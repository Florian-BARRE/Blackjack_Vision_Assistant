# ====== Code Summary ======
# Provides the DatasetProcessor class, the main entry point for processing
# Label Studio YOLO exports into training-ready datasets. Handles splitting,
# label normalization, configuration generation, and data augmentation.

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .config import DatasetConfig, TrainingConfig
from .constants import YOLO_CONSTANTS, AUGMENTATION_PRESETS
from .helpers import YOLO_HELPERS
from .augmentation import AugmentationConfig, DataAugmentor


class DatasetProcessor:
    """
    Main processor for YOLO datasets exported from Label Studio.

    Typical usage:
        processor = DatasetProcessor(dataset_dir)
        processor.process()
        processor.generate_configs()
    """

    def __init__(
        self,
        dataset_dir: Path,
        config: Optional[DatasetConfig] = None
    ) -> None:
        """
        Initialize the dataset processor.

        Args:
            dataset_dir: Root directory of the YOLO export (contains images/, labels/).
            config: Optional dataset configuration. If None, will be auto-detected.
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        # Validate directory structure
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Initialize or auto-detect configuration
        if config:
            self.config = config
        else:
            self.config = self._auto_detect_config()

    # --------------------------
    # Auto-detection
    # --------------------------

    def _auto_detect_config(self) -> DatasetConfig:
        """
        Auto-detect dataset configuration from files.

        Returns:
            Detected DatasetConfig.
        """
        # Detect task type from labels
        task = YOLO_HELPERS.detect_label_format(self.labels_dir)

        # Load class names from classes.txt if available
        classes_file = self.dataset_dir / YOLO_CONSTANTS.DEFAULT_CLASSES_FILE
        class_names = YOLO_HELPERS.load_classes(classes_file)

        # If no classes.txt, infer from labels
        if not class_names:
            num_classes = YOLO_HELPERS.infer_num_classes(self.labels_dir)
            class_names = [f"class_{i}" for i in range(num_classes)]

        return DatasetConfig(
            task=task,
            class_names=class_names,
        )

    # --------------------------
    # Dataset splitting
    # --------------------------

    def _is_already_split(self) -> bool:
        """Check if dataset is already split into train/val."""
        train_images = self.images_dir / "train"
        train_labels = self.labels_dir / "train"
        return train_images.exists() and train_labels.exists()

    def split_dataset(
        self,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Split dataset into train/val/test subsets.

        Args:
            val_ratio: Validation split ratio. Uses config default if None.
            test_ratio: Test split ratio. Uses config default if None.
            force: Force re-split even if already split.

        Returns:
            Report dictionary with split statistics.
        """
        val_ratio = val_ratio or self.config.val_split
        test_ratio = test_ratio or self.config.test_split

        # Check if already split
        if self._is_already_split() and not force:
            print("Dataset already split. Use force=True to re-split.")
            return {"status": "already_split"}

        # List all images
        image_exts = YOLO_CONSTANTS.SUPPORTED_IMAGE_EXTS
        images = [
            p for p in self.images_dir.glob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        if not images:
            raise ValueError(f"No images found in: {self.images_dir}")

        # Shuffle with seed
        random.seed(self.config.seed)
        random.shuffle(images)

        # Calculate split indices
        n = len(images)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:] if n_test > 0 else []

        print(f"Splitting {n} images: {n_train} train / {n_val} val / {n_test} test")

        # Create subset directories
        subsets = ["train", "val"]
        if n_test > 0:
            subsets.append("test")

        for subset in subsets:
            (self.images_dir / subset).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / subset).mkdir(parents=True, exist_ok=True)

        # Move files
        def move_pair(img_path: Path, subset: str) -> None:
            """Move image and corresponding label to subset folder."""
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            # Move image
            shutil.move(str(img_path), str(self.images_dir / subset / img_path.name))

            # Move label if exists
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / subset / label_path.name))

        for p in train_images:
            move_pair(p, "train")
        for p in val_images:
            move_pair(p, "val")
        for p in test_images:
            move_pair(p, "test")

        return {
            "status": "split_complete",
            "total": n,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

    # --------------------------
    # Label normalization
    # --------------------------

    def _parse_and_normalize_obb_line(
        self,
        line: str,
        img_width: int,
        img_height: int
    ) -> str:
        """
        Parse and normalize an OBB label line.

        Args:
            line: Raw label line.
            img_width: Image width.
            img_height: Image height.

        Returns:
            Normalized label line.
        """
        cls_id, coords = YOLO_HELPERS.parse_label_line(line)

        if len(coords) != 8:
            raise ValueError(f"OBB requires 8 coordinates, got {len(coords)}")

        # Normalize if in pixel coordinates
        coords = YOLO_HELPERS.normalize_coords(coords, img_width, img_height)

        # Clamp to [0, 1]
        coords = YOLO_HELPERS.clamp_coords(coords)

        # Reorder to clockwise starting near top-left
        coords = YOLO_HELPERS.reorder_obb_points_clockwise(coords)

        return YOLO_HELPERS.format_label_line(cls_id, coords)

    def _parse_and_normalize_detect_line(
        self,
        line: str,
        img_width: int,
        img_height: int
    ) -> str:
        """
        Parse and normalize a standard detection label line.

        Args:
            line: Raw label line.
            img_width: Image width.
            img_height: Image height.

        Returns:
            Normalized label line.
        """
        cls_id, coords = YOLO_HELPERS.parse_label_line(line)

        if len(coords) != 4:
            raise ValueError(f"Detection requires 4 coordinates, got {len(coords)}")

        # Check if normalization needed (any value > 1.0)
        if any(c > 1.0 for c in coords):
            # Assume format is x_center, y_center, width, height in pixels
            coords[0] /= max(1, img_width)   # x_center
            coords[1] /= max(1, img_height)  # y_center
            coords[2] /= max(1, img_width)   # width
            coords[3] /= max(1, img_height)  # height

        # Clamp to [0, 1]
        coords = YOLO_HELPERS.clamp_coords(coords)

        return YOLO_HELPERS.format_label_line(cls_id, coords)

    def normalize_labels(self, subsets: Tuple[str, ...] = ("train", "val")) -> Dict[str, Any]:
        """
        Normalize all label files in the dataset.

        Args:
            subsets: Subset folders to process.

        Returns:
            Report dictionary with processing statistics.
        """
        report = {
            "processed": 0,
            "skipped": 0,
            "errors": [],
        }

        for subset in subsets:
            labels_subset = self.labels_dir / subset
            images_subset = self.images_dir / subset

            if not labels_subset.exists():
                print(f"Skip '{subset}' (no labels directory)")
                continue

            for label_path in sorted(labels_subset.glob("*.txt")):
                try:
                    # Find matching image
                    img_path = YOLO_HELPERS.find_matching_image(label_path, images_subset)
                    if not img_path:
                        report["errors"].append(f"No image for: {label_path.name}")
                        continue

                    # Get image dimensions
                    img_width, img_height = YOLO_HELPERS.get_image_size(img_path)

                    # Read and normalize labels
                    lines = label_path.read_text(encoding="utf-8").splitlines()
                    out_lines = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            if self.config.task == YOLO_CONSTANTS.TASK_OBB:
                                out_lines.append(
                                    self._parse_and_normalize_obb_line(line, img_width, img_height)
                                )
                            else:
                                out_lines.append(
                                    self._parse_and_normalize_detect_line(line, img_width, img_height)
                                )
                        except Exception as e:
                            report["errors"].append(f"{label_path.name}: {str(e)}")

                    # Write normalized labels
                    label_path.write_text("\n".join(out_lines), encoding="utf-8")
                    report["processed"] += 1

                except Exception as e:
                    report["errors"].append(f"{label_path.name}: {str(e)}")

        print(f"✓ Normalization complete: {report['processed']} files processed")
        return report

    # --------------------------
    # Configuration generation
    # --------------------------

    def generate_data_yaml(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate data.yaml configuration file.

        Args:
            output_path: Output path. Defaults to dataset_dir/data.yaml.

        Returns:
            Path to generated file.
        """
        output_path = output_path or self.dataset_dir / YOLO_CONSTANTS.DEFAULT_DATA_YAML

        # Check for test subset
        has_test = (self.images_dir / "test").exists()

        cfg = {
            "path": str(self.dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(self.config.class_names)},
        }

        if has_test:
            cfg["test"] = "images/test"

        if self.config.task == YOLO_CONSTANTS.TASK_OBB:
            cfg["task"] = "obb"

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"✓ Generated: {output_path}")
        return output_path

    def generate_hyp_yaml(
        self,
        preset: str = "standard",
        custom_params: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate hyperparameter/augmentation YAML file.

        Args:
            preset: Augmentation preset name.
            custom_params: Custom parameters to override preset.
            output_path: Output path. Defaults to dataset_dir/hyp.yaml.

        Returns:
            Path to generated file.
        """
        output_path = output_path or self.dataset_dir / YOLO_CONSTANTS.DEFAULT_HYP_YAML

        # Get preset and merge custom params
        hyp = AUGMENTATION_PRESETS.get_preset(preset)
        if custom_params:
            hyp.update(custom_params)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)

        print(f"✓ Generated: {output_path}")
        return output_path

    def generate_train_config(
        self,
        training_config: Optional[TrainingConfig] = None,
        aug_preset: str = "standard",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate training configuration YAML file.

        Args:
            training_config: Training configuration. Auto-creates if None.
            aug_preset: Augmentation preset for auto-created config.
            output_path: Output path. Defaults to dataset_dir/train_cfg.yaml.

        Returns:
            Path to generated file.
        """
        output_path = output_path or self.dataset_dir / YOLO_CONSTANTS.DEFAULT_TRAIN_CFG_YAML
        data_yaml_path = self.dataset_dir / YOLO_CONSTANTS.DEFAULT_DATA_YAML

        # Create training config if not provided
        if training_config is None:
            training_config = TrainingConfig.from_preset(
                task=self.config.task,
                aug_preset=aug_preset,
            )

        training_config.save(output_path, data_yaml_path)

        print(f"✓ Generated: {output_path}")
        return output_path

    def generate_configs(
        self,
        aug_preset: str = "standard",
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Path]:
        """
        Generate all configuration files.

        Args:
            aug_preset: Augmentation preset name.
            training_config: Optional custom training configuration.

        Returns:
            Dictionary of generated file paths.
        """
        return {
            "data_yaml": self.generate_data_yaml(),
            "hyp_yaml": self.generate_hyp_yaml(preset=aug_preset),
            "train_cfg": self.generate_train_config(
                training_config=training_config,
                aug_preset=aug_preset
            ),
        }

    # --------------------------
    # Data augmentation
    # --------------------------

    def apply_offline_augmentation(
        self,
        aug_config: Optional[AugmentationConfig] = None,
        output_dir: Optional[Path] = None,
        subsets: Tuple[str, ...] = ("train",)
    ) -> Dict[str, Any]:
        """
        Apply offline data augmentation to the dataset.

        Args:
            aug_config: Augmentation configuration. Uses standard if None.
            output_dir: Output directory. Defaults to dataset_dir_augmented.
            subsets: Subsets to augment (typically only 'train').

        Returns:
            Report dictionary with augmentation statistics.
        """
        aug_config = aug_config or AugmentationConfig.standard()
        output_dir = output_dir or self.dataset_dir.parent / f"{self.dataset_dir.name}_augmented"

        # Prepare output structure
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"

        augmentor = DataAugmentor(aug_config)
        reports = {}

        for subset in subsets:
            report = augmentor.augment_dataset(
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                output_images_dir=output_images,
                output_labels_dir=output_labels,
                task=self.config.task,
                subset=subset,
            )
            reports[subset] = report

        # Copy non-augmented subsets (val, test) only if output_dir is different from source
        is_same_dir = output_dir.resolve() == self.dataset_dir.resolve()
        
        if not is_same_dir:
            all_subsets = {"train", "val", "test"}
            for subset in all_subsets - set(subsets):
                src_img = self.images_dir / subset
                src_lbl = self.labels_dir / subset

                if src_img.exists():
                    dst_img = output_images / subset
                    dst_lbl = output_labels / subset
                    shutil.copytree(src_img, dst_img, dirs_exist_ok=True)
                    if src_lbl.exists():
                        shutil.copytree(src_lbl, dst_lbl, dirs_exist_ok=True)

            # Copy classes.txt only if different directory
            classes_src = self.dataset_dir / YOLO_CONSTANTS.DEFAULT_CLASSES_FILE
            if classes_src.exists():
                shutil.copy2(classes_src, output_dir / YOLO_CONSTANTS.DEFAULT_CLASSES_FILE)

        print(f"✓ Augmentation complete. Output: {output_dir}")
        return {"output_dir": str(output_dir), "subsets": reports}

    # --------------------------
    # Full pipeline
    # --------------------------

    def process(
        self,
        split: bool = True,
        normalize: bool = True,
        generate_configs: bool = True,
        aug_preset: str = "standard"
    ) -> Dict[str, Any]:
        """
        Run the full processing pipeline.

        Args:
            split: Whether to split the dataset.
            normalize: Whether to normalize labels.
            generate_configs: Whether to generate configuration files.
            aug_preset: Augmentation preset for config generation.

        Returns:
            Combined report dictionary.
        """
        report = {}

        # Step 1: Split dataset
        if split:
            report["split"] = self.split_dataset()

        # Step 2: Normalize labels
        if normalize:
            subsets = ("train", "val")
            if (self.images_dir / "test").exists():
                subsets = ("train", "val", "test")
            report["normalize"] = self.normalize_labels(subsets)

        # Step 3: Generate configs
        if generate_configs:
            report["configs"] = self.generate_configs(aug_preset=aug_preset)

        # Save dataset config
        config_path = self.dataset_dir / YOLO_CONSTANTS.DEFAULT_CONFIG_FILENAME
        self.config.save(config_path)
        report["dataset_config"] = str(config_path)

        print("\n✅ Processing complete!")
        return report

    # --------------------------
    # Class methods for loading
    # --------------------------

    @classmethod
    def load_config(
        cls,
        directory: Path = YOLO_CONSTANTS.DEFAULT_CONFIG_PATH,
        filename: str = YOLO_CONSTANTS.DEFAULT_CONFIG_FILENAME,
        required: bool = True
    ) -> DatasetProcessor:
        """
        Load a DatasetProcessor from a saved configuration.

        Args:
            directory: Directory containing the config file.
            filename: Config filename.
            required: Whether config is required.

        Returns:
            Configured DatasetProcessor instance.
        """
        config_path = directory / filename

        if config_path.exists():
            config = DatasetConfig.load(config_path)
            return cls(directory, config)

        if required:
            raise FileNotFoundError(f"Config not found: {config_path}")

        return cls(directory)

    def get_training_command(self, config_path: Optional[Path] = None) -> str:
        """
        Get the CLI command to run YOLO training.

        Args:
            config_path: Path to train_cfg.yaml.

        Returns:
            Training command string.
        """
        config_path = config_path or self.dataset_dir / YOLO_CONSTANTS.DEFAULT_TRAIN_CFG_YAML
        return f'yolo train cfg="{config_path}"'

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            "task": self.config.task,
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
            "subsets": {},
        }

        for subset in ["train", "val", "test"]:
            img_dir = self.images_dir / subset
            lbl_dir = self.labels_dir / subset

            if img_dir.exists():
                images = YOLO_HELPERS.list_images(img_dir)
                labels = YOLO_HELPERS.list_labels(lbl_dir) if lbl_dir.exists() else []

                stats["subsets"][subset] = {
                    "images": len(images),
                    "labels": len(labels),
                }

        return stats
