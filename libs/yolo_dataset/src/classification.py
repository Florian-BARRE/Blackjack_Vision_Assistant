# ====== Code Summary ======
# Provides ClassificationProcessor for processing Label Studio JSON-MIN exports
# into YOLO classification datasets. Supports data augmentation matching OBB workflow.

from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import yaml
import cv2
import numpy as np

from .constants import YOLO_CONSTANTS, AUGMENTATION_PRESETS
from .helpers import YOLO_HELPERS


@dataclass
class ClassificationField:
    """
    Configuration for a classification field from JSON-MIN export.

    Attributes:
        json_key: Key in JSON for this classification field
        class_names: List of valid class names (auto-detected if empty)
        dataset_name: Name for the output dataset folder
        alt_keys: Alternative JSON keys to try
    """
    json_key: str
    class_names: List[str] = field(default_factory=list)
    dataset_name: str = ""
    alt_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.dataset_name:
            self.dataset_name = f"{self.json_key}_dataset"

    def get_value(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract the classification value from a JSON item."""
        value = item.get(self.json_key)
        if value is None:
            for alt in self.alt_keys:
                value = item.get(alt)
                if value is not None:
                    break

        if value is None:
            return None

        value = str(value).strip()
        
        # If class_names is empty, accept any value
        if not self.class_names:
            return value
        
        return value if value in self.class_names else None


@dataclass
class ClassificationConfig:
    """
    Configuration for classification dataset processing.

    Attributes:
        fields: List of ClassificationField configurations
        image_key: JSON key for image path
        image_alt_keys: Alternative keys for image path
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        aug_separator: Separator for augmented image variants
    """
    fields: List[ClassificationField] = field(default_factory=list)
    image_key: str = "image"
    image_alt_keys: List[str] = field(default_factory=lambda: ["img", "image_url"])
    val_split: float = 0.15
    test_split: float = 0.0
    seed: int = 42
    aug_separator: str = "_aug"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task": "classify",
            "fields": [
                {
                    "json_key": f.json_key,
                    "class_names": f.class_names,
                    "dataset_name": f.dataset_name,
                    "alt_keys": f.alt_keys,
                }
                for f in self.fields
            ],
            "image_key": self.image_key,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


@dataclass 
class ClsAugmentationConfig:
    """
    Configuration for classification image augmentation.
    
    Attributes:
        num_augmented: Number of augmented copies per image
        flip_horizontal: Probability of horizontal flip
        rotation_range: Max rotation angle in degrees
        brightness_range: Brightness adjustment range (min, max)
        contrast_range: Contrast adjustment range (min, max)
        saturation_range: Saturation adjustment range (min, max)
        noise_probability: Probability of adding noise
        blur_probability: Probability of applying blur
    """
    num_augmented: int = 4
    flip_horizontal: float = 0.5
    rotation_range: float = 15.0
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    noise_probability: float = 0.1
    blur_probability: float = 0.1

    @classmethod
    def standard(cls) -> ClsAugmentationConfig:
        """Standard augmentation preset."""
        return cls()

    @classmethod
    def light(cls) -> ClsAugmentationConfig:
        """Light augmentation preset."""
        return cls(
            num_augmented=2,
            rotation_range=10.0,
            noise_probability=0.05,
            blur_probability=0.05,
        )

    @classmethod
    def heavy(cls) -> ClsAugmentationConfig:
        """Heavy augmentation preset."""
        return cls(
            num_augmented=6,
            rotation_range=25.0,
            brightness_range=(0.5, 1.5),
            contrast_range=(0.6, 1.4),
            noise_probability=0.2,
            blur_probability=0.2,
        )


class ClsAugmentor:
    """
    Augmentation pipeline for classification images.
    """

    def __init__(self, config: ClsAugmentationConfig):
        self.config = config

    def augment(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to an image.

        Args:
            img: Input image (BGR).

        Returns:
            Augmented image.
        """
        img = img.copy()

        # Horizontal flip
        if random.random() < self.config.flip_horizontal:
            img = cv2.flip(img, 1)

        # Rotation
        if self.config.rotation_range > 0:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Color adjustments
        img = self._adjust_colors(img)

        # Noise
        if random.random() < self.config.noise_probability:
            img = self._add_noise(img)

        # Blur
        if random.random() < self.config.blur_probability:
            img = self._add_blur(img)

        return img

    def _adjust_colors(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, and saturation adjustments."""
        # Convert to HSV for saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Saturation
        sat_factor = random.uniform(*self.config.saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Brightness and contrast
        brightness = random.uniform(*self.config.brightness_range)
        contrast = random.uniform(*self.config.contrast_range)

        img = img.astype(np.float32)
        img = img * contrast + (brightness - 1.0) * 127
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def _add_blur(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian blur."""
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


class ClassificationProcessor:
    """
    Processor for Label Studio JSON-MIN classification exports.

    Workflow:
    1. Load JSON-MIN export and build image index
    2. Auto-detect classes if not specified
    3. Split into train/val/test (group-wise for augmented variants)
    4. Apply offline data augmentation
    5. Generate training configuration

    Usage:
        processor = ClassificationProcessor(
            json_path=Path("data.json"),
            images_dir=Path("images"),
            output_dir=Path("output"),
        )
        processor.auto_detect_fields()
        processor.process()
    """

    def __init__(
        self,
        json_path: Path,
        images_dir: Path,
        output_dir: Path,
        config: Optional[ClassificationConfig] = None
    ) -> None:
        """
        Initialize the classification processor.

        Args:
            json_path: Path to Label Studio JSON-MIN export.
            images_dir: Directory containing images.
            output_dir: Output directory for the dataset.
            config: Optional configuration (auto-created if None).
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ClassificationConfig()

        # Validate paths
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Load JSON
        self.items: List[Dict[str, Any]] = json.loads(
            self.json_path.read_text(encoding="utf-8")
        )

        # Build image index
        self._image_index = self._build_image_index()

        print(f"Loaded {len(self.items)} items from JSON")
        print(f"Indexed {sum(len(v) for v in self._image_index.values())} images")

    # --------------------------
    # Image indexing
    # --------------------------

    def _build_image_index(self) -> Dict[str, List[Path]]:
        """Build index of images by lowercase stem."""
        index = defaultdict(list)
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in YOLO_CONSTANTS.SUPPORTED_IMAGE_EXTS:
                index[p.stem.lower()].append(p)
        return dict(index)

    def _get_base_stem(self, stem: str) -> str:
        """Get base stem without augmentation suffix."""
        sep = self.config.aug_separator
        return stem.split(sep)[0] if sep in stem else stem

    def _find_all_variants(self, base_stem: str) -> List[Path]:
        """Find all image variants for a base stem."""
        base_lower = base_stem.lower()
        sep = self.config.aug_separator.lower()
        out = []

        for stem, paths in self._image_index.items():
            if stem == base_lower or stem.startswith(base_lower + sep):
                out.extend(paths)

        return list(set(out))

    @staticmethod
    def _extract_image_name(img_field: str) -> str:
        """Extract filename from JSON image field."""
        parsed = urlparse(str(img_field))
        name = Path(parsed.path).name
        return unquote(name)

    def _get_image_field(self, item: Dict[str, Any]) -> Optional[str]:
        """Get image field value from a JSON item."""
        value = item.get(self.config.image_key)
        if value is None:
            for alt in self.config.image_alt_keys:
                value = item.get(alt)
                if value is not None:
                    break
        return value

    # --------------------------
    # Auto-detection
    # --------------------------

    def analyze_json(self) -> Dict[str, List[str]]:
        """
        Analyze JSON to find potential label fields and their unique values.

        Returns:
            Dictionary mapping field_name -> list of unique values.
        """
        skip_keys = {
            "id", "image", "img", "image_url", "file", "path",
            "annotator", "annotation_id", "created_at", "updated_at",
            "lead_time", "project", "task_id"
        }

        key_values = defaultdict(set)

        for item in self.items:
            for key, value in item.items():
                if key.lower() not in skip_keys and value is not None:
                    key_values[key].add(str(value).strip())

        # Filter to reasonable class counts
        potential = {}
        for key, values in key_values.items():
            if 2 <= len(values) <= 100:
                potential[key] = sorted(values)

        return potential

    def auto_detect_fields(self, label_key: Optional[str] = None) -> None:
        """
        Auto-detect classification fields from JSON.

        Args:
            label_key: Specific label key to use (auto-selects if None).
        """
        potential = self.analyze_json()

        if not potential:
            raise ValueError("No suitable label fields found in JSON")

        print("\nDetected label fields:")
        for key, values in potential.items():
            print(f"  '{key}': {len(values)} classes -> {values[:5]}{'...' if len(values) > 5 else ''}")

        # Select label key
        if label_key:
            if label_key not in potential:
                raise ValueError(f"Label key '{label_key}' not found in JSON")
            selected_key = label_key
        else:
            # Auto-select the one with most classes
            selected_key = max(potential.keys(), key=lambda k: len(potential[k]))

        print(f"\nSelected: '{selected_key}' with {len(potential[selected_key])} classes")

        # Create field config
        self.config.fields = [
            ClassificationField(
                json_key=selected_key,
                class_names=potential[selected_key],
                dataset_name=f"{selected_key}_dataset",
            )
        ]

    # --------------------------
    # Dataset building
    # --------------------------

    def _build_groups(
        self,
        cls_field: ClassificationField
    ) -> Tuple[Dict[str, List[Tuple[str, Path]]], List[str]]:
        """
        Build groups of images for a classification field.

        Returns:
            Tuple of (groups dict, detected class names).
        """
        groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        detected_classes: set = set()
        stats = {"missing_image": 0, "bad_label": 0, "processed": 0}

        for item in self.items:
            img_field = self._get_image_field(item)
            if not img_field:
                stats["bad_label"] += 1
                continue

            class_value = cls_field.get_value(item)
            if not class_value:
                stats["bad_label"] += 1
                continue

            detected_classes.add(class_value)

            # Find image
            base_name = self._extract_image_name(img_field)
            stem = Path(base_name).stem
            base_stem = self._get_base_stem(stem)

            variants = self._find_all_variants(base_stem)
            if not variants:
                stats["missing_image"] += 1
                continue

            for src in variants:
                groups[base_stem].append((class_value, src))
            stats["processed"] += 1

        print(f"  Groups: {len(groups)} | Processed: {stats['processed']} | "
              f"Missing: {stats['missing_image']} | Bad labels: {stats['bad_label']}")

        # Update class names if auto-detecting
        if not cls_field.class_names:
            cls_field.class_names = sorted(detected_classes)

        return dict(groups), cls_field.class_names

    def _split_groups(
        self,
        group_keys: List[str]
    ) -> Dict[str, str]:
        """
        Split group keys into train/val/test.

        Returns:
            Dictionary mapping group_key -> split name.
        """
        keys = sorted(group_keys)
        rng = random.Random(self.config.seed)
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(n * (1 - self.config.val_split - self.config.test_split))
        n_val = int(n * (1 - self.config.test_split))

        split_map = {}
        for i, key in enumerate(keys):
            if i < n_train:
                split_map[key] = "train"
            elif i < n_val:
                split_map[key] = "val"
            else:
                split_map[key] = "test"

        return split_map

    def build_dataset(
        self,
        cls_field: ClassificationField,
        output_root: Path
    ) -> Dict[str, Any]:
        """
        Build the classification dataset.

        Args:
            cls_field: Classification field config.
            output_root: Output root directory.

        Returns:
            Report dictionary.
        """
        # Build groups
        groups, class_names = self._build_groups(cls_field)

        if not groups:
            return {"error": "No valid groups found"}

        # Create folder structure
        splits = ["train", "val"]
        if self.config.test_split > 0:
            splits.append("test")

        for split in splits:
            for cls in class_names:
                (output_root / split / cls).mkdir(parents=True, exist_ok=True)

        # Split groups
        split_map = self._split_groups(list(groups.keys()))

        # Copy files
        stats = {split: defaultdict(int) for split in splits}
        total = 0

        for group_key, items in groups.items():
            split = split_map.get(group_key, "train")

            for i, (cls_name, src) in enumerate(items):
                dst = output_root / split / cls_name / src.name

                if dst.exists():
                    dst = dst.with_name(f"{src.stem}_{i}{src.suffix}")

                shutil.copy2(src, dst)
                stats[split][cls_name] += 1
                total += 1

        print(f"  ✓ Copied {total} images")

        return {
            "total": total,
            "classes": class_names,
            "splits": {k: dict(v) for k, v in stats.items()},
        }

    # --------------------------
    # Augmentation
    # --------------------------

    def apply_augmentation(
        self,
        aug_config: Optional[ClsAugmentationConfig] = None,
        subsets: Tuple[str, ...] = ("train",)
    ) -> Dict[str, Any]:
        """
        Apply offline data augmentation to the dataset.

        Args:
            aug_config: Augmentation configuration.
            subsets: Subsets to augment.

        Returns:
            Report dictionary.
        """
        aug_config = aug_config or ClsAugmentationConfig.standard()
        augmentor = ClsAugmentor(aug_config)

        report = {"subsets": {}}

        for cls_field in self.config.fields:
            ds_root = self.output_dir / cls_field.dataset_name

            for subset in subsets:
                subset_dir = ds_root / subset
                if not subset_dir.exists():
                    continue

                original_count = 0
                augmented_count = 0

                for cls_dir in subset_dir.iterdir():
                    if not cls_dir.is_dir():
                        continue

                    # Get original images (skip already augmented)
                    images = [
                        p for p in cls_dir.glob("*")
                        if p.suffix.lower() in YOLO_CONSTANTS.SUPPORTED_IMAGE_EXTS
                        and self.config.aug_separator not in p.stem
                    ]

                    original_count += len(images)

                    for img_path in images:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                        # Generate augmented versions
                        for i in range(aug_config.num_augmented):
                            aug_img = augmentor.augment(img)
                            aug_name = f"{img_path.stem}{self.config.aug_separator}{i+1:02d}{img_path.suffix}"
                            cv2.imwrite(str(cls_dir / aug_name), aug_img)
                            augmented_count += 1

                report["subsets"][f"{cls_field.dataset_name}/{subset}"] = {
                    "original": original_count,
                    "augmented": augmented_count,
                    "total": original_count + augmented_count,
                }

                print(f"  ✓ {cls_field.dataset_name}/{subset}: "
                      f"{original_count} original + {augmented_count} augmented")

        return report

    # --------------------------
    # Config generation
    # --------------------------

    def generate_train_config(
        self,
        cls_field: ClassificationField,
        output_root: Path,
        model: str = "yolo11s-cls.pt",
        img_size: int = 224,
        epochs: int = 100,
        batch_size: int = 16,
        device: Any = 0,
        patience: int = 20,
        aug_preset: str = "classification",
    ) -> Path:
        """
        Generate training configuration YAML.

        Returns:
            Path to generated config file.
        """
        cfg_path = output_root / "train_cfg.yaml"

        cfg = {
            "task": "classify",
            "mode": "train",
            "model": model,
            "data": str(output_root.resolve()),
            "imgsz": img_size,
            "epochs": epochs,
            "batch": batch_size,
            "device": device,
            "seed": self.config.seed,
            "workers": 4,
            "patience": patience,
            "project": "training_runs",
            "name": cls_field.dataset_name,
            "save": True,
            "plots": True,
            "verbose": True,
        }

        # Add augmentation params
        try:
            aug_params = AUGMENTATION_PRESETS.get_preset(aug_preset)
            cfg.update(aug_params)
        except ValueError:
            pass

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"  ✓ Generated: {cfg_path}")
        return cfg_path

    # --------------------------
    # Main pipeline
    # --------------------------

    def process(
        self,
        apply_augmentation: bool = True,
        aug_config: Optional[ClsAugmentationConfig] = None,
        generate_configs: bool = True,
        training_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the full processing pipeline.

        Args:
            apply_augmentation: Whether to apply offline augmentation.
            aug_config: Augmentation configuration.
            generate_configs: Whether to generate training configs.
            training_params: Additional training parameters.

        Returns:
            Combined report dictionary.
        """
        if not self.config.fields:
            print("No fields configured. Running auto-detection...")
            self.auto_detect_fields()

        training_params = training_params or {}
        report = {"datasets": {}}

        for cls_field in self.config.fields:
            print(f"\n{'='*50}")
            print(f"Processing: {cls_field.dataset_name}")
            print(f"{'='*50}")

            output_root = self.output_dir / cls_field.dataset_name

            # Remove existing if present
            if output_root.exists():
                shutil.rmtree(output_root)

            # Build dataset
            ds_report = self.build_dataset(cls_field, output_root)
            report["datasets"][cls_field.dataset_name] = ds_report

            if "error" in ds_report:
                continue

            # Apply augmentation
            if apply_augmentation:
                print("\nApplying augmentation...")
                aug_report = self.apply_augmentation(aug_config, subsets=("train",))
                ds_report["augmentation"] = aug_report

            # Generate training config
            if generate_configs:
                print("\nGenerating config...")
                cfg_path = self.generate_train_config(
                    cls_field=cls_field,
                    output_root=output_root,
                    **training_params
                )
                ds_report["train_cfg"] = str(cfg_path)

        # Save classification config
        config_path = self.output_dir / "classification_config.json"
        self.config.save(config_path)
        report["config_path"] = str(config_path)

        print(f"\n{'='*50}")
        print("✅ Classification processing complete!")
        print(f"{'='*50}")

        return report

    def get_training_commands(self) -> Dict[str, str]:
        """Get CLI training commands for all datasets."""
        commands = {}
        for cls_field in self.config.fields:
            cfg_path = self.output_dir / cls_field.dataset_name / "train_cfg.yaml"
            commands[cls_field.dataset_name] = f'yolo train cfg="{cfg_path}"'
        return commands

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {}

        for cls_field in self.config.fields:
            ds_root = self.output_dir / cls_field.dataset_name
            if not ds_root.exists():
                continue

            ds_stats = {"classes": cls_field.class_names, "splits": {}}

            for split in ("train", "val", "test"):
                split_dir = ds_root / split
                if not split_dir.exists():
                    continue

                split_counts = {}
                for cls_name in cls_field.class_names:
                    cls_dir = split_dir / cls_name
                    if cls_dir.exists():
                        split_counts[cls_name] = len(list(cls_dir.glob("*")))

                ds_stats["splits"][split] = {
                    "total": sum(split_counts.values()),
                    "per_class": split_counts,
                }

            stats[cls_field.dataset_name] = ds_stats

        return stats
