# ====== Code Summary ======
# Provides offline data augmentation utilities for YOLO datasets.
# Supports geometric and color transformations with proper label adjustment.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import YOLO_CONSTANTS, AUGMENTATION_PRESETS
from .helpers import YOLO_HELPERS


@dataclass
class AugmentationConfig:
    """
    Configuration for offline data augmentation.

    Attributes:
        num_augmented: Number of augmented copies per original image
        flip_horizontal: Probability of horizontal flip
        flip_vertical: Probability of vertical flip
        rotation_range: Maximum rotation angle in degrees
        scale_range: Scale factor range (min, max)
        brightness_range: Brightness adjustment range
        contrast_range: Contrast adjustment range
        saturation_range: Saturation adjustment range
        hue_range: Hue adjustment range
        noise_probability: Probability of adding noise
        blur_probability: Probability of applying blur
        preserve_original: Whether to keep original images
    """

    num_augmented: int = 3
    flip_horizontal: float = 0.5
    flip_vertical: float = 0.0
    rotation_range: float = 15.0
    scale_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_range: float = 0.1
    noise_probability: float = 0.1
    blur_probability: float = 0.1
    preserve_original: bool = True

    @classmethod
    def minimal(cls) -> AugmentationConfig:
        """Create minimal augmentation config."""
        return cls(
            num_augmented=2,
            flip_horizontal=0.5,
            flip_vertical=0.0,
            rotation_range=5.0,
            scale_range=(0.9, 1.1),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.95, 1.05),
            saturation_range=(0.9, 1.1),
            hue_range=0.02,
            noise_probability=0.0,
            blur_probability=0.0,
        )

    @classmethod
    def standard(cls) -> AugmentationConfig:
        """Create standard augmentation config."""
        return cls(
            num_augmented=4,
            flip_horizontal=0.5,
            flip_vertical=0.0,
            rotation_range=10.0,
            scale_range=(0.8, 1.2),
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_range=0.05,
            noise_probability=0.1,
            blur_probability=0.1,
        )

    @classmethod
    def heavy(cls) -> AugmentationConfig:
        """Create heavy augmentation config."""
        return cls(
            num_augmented=6,
            flip_horizontal=0.5,
            flip_vertical=0.2,
            rotation_range=20.0,
            scale_range=(0.6, 1.4),
            brightness_range=(0.6, 1.4),
            contrast_range=(0.7, 1.3),
            saturation_range=(0.6, 1.4),
            hue_range=0.1,
            noise_probability=0.2,
            blur_probability=0.15,
        )


class DataAugmentor:
    """
    Offline data augmentation for YOLO datasets.

    Applies augmentations to images and adjusts labels accordingly.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        """
        Initialize the augmentor with configuration.

        Args:
            config: Augmentation configuration.
        """
        self.config = config

    # --------------------------
    # Image transformations
    # --------------------------

    def _apply_color_augmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply color augmentations to an image.

        Args:
            img: Input image (BGR format).

        Returns:
            Augmented image.
        """
        import cv2

        # Convert to HSV for color adjustments
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue adjustment
        h_shift = random.uniform(-self.config.hue_range, self.config.hue_range) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180

        # Saturation adjustment
        s_factor = random.uniform(*self.config.saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)

        # Value (brightness) adjustment
        v_factor = random.uniform(*self.config.brightness_range)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)

        # Convert back to BGR
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Contrast adjustment
        c_factor = random.uniform(*self.config.contrast_range)
        img = np.clip((img.astype(np.float32) - 128) * c_factor + 128, 0, 255).astype(np.uint8)

        return img

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add random noise to an image.

        Args:
            img: Input image.

        Returns:
            Noisy image.
        """
        if random.random() > self.config.noise_probability:
            return img

        noise_type = random.choice(["gaussian", "salt_pepper"])

        if noise_type == "gaussian":
            sigma = random.uniform(5, 20)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            # Salt and pepper noise
            prob = random.uniform(0.01, 0.03)
            salt_mask = np.random.random(img.shape[:2]) < prob / 2
            pepper_mask = np.random.random(img.shape[:2]) < prob / 2
            img[salt_mask] = 255
            img[pepper_mask] = 0

        return img

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random blur to an image.

        Args:
            img: Input image.

        Returns:
            Blurred image.
        """
        import cv2

        if random.random() > self.config.blur_probability:
            return img

        blur_type = random.choice(["gaussian", "motion"])
        kernel_size = random.choice([3, 5])

        if blur_type == "gaussian":
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        else:
            # Simple motion blur
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1 / kernel_size
            img = cv2.filter2D(img, -1, kernel)

        return img

    # --------------------------
    # Geometric transformations
    # --------------------------

    def _flip_horizontal(
        self,
        img: np.ndarray,
        labels: List[Tuple[int, List[float]]],
        task: str
    ) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Apply horizontal flip to image and labels.

        Args:
            img: Input image.
            labels: List of (class_id, coordinates) tuples.
            task: YOLO task type.

        Returns:
            Tuple of (flipped image, adjusted labels).
        """
        import cv2

        img = cv2.flip(img, 1)

        new_labels = []
        for cls_id, coords in labels:
            if task == YOLO_CONSTANTS.TASK_OBB:
                # OBB: flip x coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
                new_coords = []
                for i in range(0, len(coords), 2):
                    new_coords.append(1.0 - coords[i])  # x
                    new_coords.append(coords[i + 1])     # y
                new_coords = YOLO_HELPERS.reorder_obb_points_clockwise(new_coords)
            else:
                # Standard: flip x_center
                new_coords = [1.0 - coords[0]] + coords[1:]

            new_labels.append((cls_id, new_coords))

        return img, new_labels

    def _flip_vertical(
        self,
        img: np.ndarray,
        labels: List[Tuple[int, List[float]]],
        task: str
    ) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Apply vertical flip to image and labels.

        Args:
            img: Input image.
            labels: List of (class_id, coordinates) tuples.
            task: YOLO task type.

        Returns:
            Tuple of (flipped image, adjusted labels).
        """
        import cv2

        img = cv2.flip(img, 0)

        new_labels = []
        for cls_id, coords in labels:
            if task == YOLO_CONSTANTS.TASK_OBB:
                # OBB: flip y coordinates
                new_coords = []
                for i in range(0, len(coords), 2):
                    new_coords.append(coords[i])          # x
                    new_coords.append(1.0 - coords[i + 1])  # y
                new_coords = YOLO_HELPERS.reorder_obb_points_clockwise(new_coords)
            else:
                # Standard: flip y_center
                new_coords = [coords[0], 1.0 - coords[1]] + coords[2:]

            new_labels.append((cls_id, new_coords))

        return img, new_labels

    def _rotate(
        self,
        img: np.ndarray,
        labels: List[Tuple[int, List[float]]],
        angle: float,
        task: str
    ) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Rotate image and labels.

        Args:
            img: Input image.
            labels: List of (class_id, coordinates) tuples.
            angle: Rotation angle in degrees.
            task: YOLO task type.

        Returns:
            Tuple of (rotated image, adjusted labels).
        """
        import cv2

        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Rotate labels
        rad = np.deg2rad(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        new_labels = []
        for cls_id, coords in labels:
            if task == YOLO_CONSTANTS.TASK_OBB:
                # Rotate each point
                new_coords = []
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i + 1]
                    # Convert to pixel coords
                    px, py = x * w, y * h
                    # Translate to origin
                    px -= center[0]
                    py -= center[1]
                    # Rotate
                    nx = px * cos_a - py * sin_a
                    ny = px * sin_a + py * cos_a
                    # Translate back
                    nx += center[0]
                    ny += center[1]
                    # Normalize
                    new_coords.append(nx / w)
                    new_coords.append(ny / h)
                new_coords = YOLO_HELPERS.clamp_coords(new_coords)
                new_coords = YOLO_HELPERS.reorder_obb_points_clockwise(new_coords)
            else:
                # Rotate center point
                x, y = coords[0] * w, coords[1] * h
                x -= center[0]
                y -= center[1]
                nx = x * cos_a - y * sin_a
                ny = x * sin_a + y * cos_a
                nx = (nx + center[0]) / w
                ny = (ny + center[1]) / h
                new_coords = [nx, ny] + coords[2:]
                new_coords = YOLO_HELPERS.clamp_coords(new_coords)

            new_labels.append((cls_id, new_coords))

        return img, new_labels

    # --------------------------
    # Main augmentation pipeline
    # --------------------------

    def augment(
        self,
        img: np.ndarray,
        labels: List[Tuple[int, List[float]]],
        task: str = YOLO_CONSTANTS.TASK_DETECT
    ) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Apply random augmentations to an image and its labels.

        Args:
            img: Input image (BGR format).
            labels: List of (class_id, coordinates) tuples.
            task: YOLO task type.

        Returns:
            Tuple of (augmented image, adjusted labels).
        """
        # Apply color augmentations
        img = self._apply_color_augmentation(img)

        # Apply geometric augmentations
        if random.random() < self.config.flip_horizontal:
            img, labels = self._flip_horizontal(img, labels, task)

        if random.random() < self.config.flip_vertical:
            img, labels = self._flip_vertical(img, labels, task)

        if self.config.rotation_range > 0:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            if abs(angle) > 1.0:  # Only rotate if angle is significant
                img, labels = self._rotate(img, labels, angle, task)

        # Apply noise and blur
        img = self._apply_noise(img)
        img = self._apply_blur(img)

        return img, labels

    def augment_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_images_dir: Path,
        output_labels_dir: Path,
        task: str = YOLO_CONSTANTS.TASK_DETECT,
        subset: str = "train"
    ) -> Dict[str, Any]:
        """
        Apply augmentation to an entire dataset subset.

        Args:
            images_dir: Directory containing original images.
            labels_dir: Directory containing original labels.
            output_images_dir: Output directory for augmented images.
            output_labels_dir: Output directory for augmented labels.
            task: YOLO task type.
            subset: Subset name ('train', 'val', etc.).

        Returns:
            Report dictionary with statistics.
        """
        import cv2

        # Prepare directories
        src_img_dir = images_dir / subset
        src_lbl_dir = labels_dir / subset
        dst_img_dir = output_images_dir / subset
        dst_lbl_dir = output_labels_dir / subset

        if not src_img_dir.exists():
            return {"error": f"Source images directory not found: {src_img_dir}"}

        # Check if we're augmenting in place (source == destination)
        is_in_place = src_img_dir.resolve() == dst_img_dir.resolve()

        if not is_in_place:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "subset": subset,
            "original_count": 0,
            "augmented_count": 0,
            "total_count": 0,
            "errors": [],
        }

        # Process each image
        image_paths = YOLO_HELPERS.list_images(src_img_dir)
        report["original_count"] = len(image_paths)

        for img_path in image_paths:
            label_path = src_lbl_dir / f"{img_path.stem}.txt"

            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    report["errors"].append(f"Failed to load: {img_path}")
                    continue

                # Load labels
                labels = []
                if label_path.exists():
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            cls_id, coords = YOLO_HELPERS.parse_label_line(line)
                            labels.append((cls_id, coords))

                # Copy original if preserve_original is True AND not in-place
                if self.config.preserve_original and not is_in_place:
                    cv2.imwrite(str(dst_img_dir / img_path.name), img)
                    if label_path.exists():
                        (dst_lbl_dir / label_path.name).write_text(
                            label_path.read_text(encoding="utf-8"),
                            encoding="utf-8"
                        )
                
                # Count original if in-place or preserved
                if is_in_place or self.config.preserve_original:
                    report["total_count"] += 1

                # Generate augmented versions
                for i in range(self.config.num_augmented):
                    aug_img, aug_labels = self.augment(img.copy(), labels.copy(), task)

                    # Save augmented image
                    aug_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                    cv2.imwrite(str(dst_img_dir / aug_name), aug_img)

                    # Save augmented labels
                    aug_label_name = f"{img_path.stem}_aug{i+1}.txt"
                    label_lines = [
                        YOLO_HELPERS.format_label_line(cls_id, coords)
                        for cls_id, coords in aug_labels
                    ]
                    (dst_lbl_dir / aug_label_name).write_text(
                        "\n".join(label_lines),
                        encoding="utf-8"
                    )

                    report["augmented_count"] += 1
                    report["total_count"] += 1

            except Exception as e:
                report["errors"].append(f"{img_path.name}: {str(e)}")

        return report
