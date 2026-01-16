# ====== Code Summary ======
# Utility functions for YOLO dataset processing: image handling, label parsing,
# file operations, and coordinate transformations.

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np

from .constants import YOLO_CONSTANTS


class YOLO_HELPERS:
    """
    Helper utilities for YOLO dataset processing.
    """

    # --------------------------
    # File operations
    # --------------------------

    @staticmethod
    def list_images(
        folder: Path,
        exts: Set[str] = YOLO_CONSTANTS.SUPPORTED_IMAGE_EXTS
    ) -> List[Path]:
        """
        List all supported image files in the specified folder.

        Args:
            folder: Directory to search for image files.
            exts: Allowed file extensions (lowercase, with leading dot).

        Returns:
            Sorted list of image file paths.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")

        paths = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

        return sorted(paths, key=lambda p: p.name.lower())

    @staticmethod
    def list_labels(folder: Path) -> List[Path]:
        """
        List all label (.txt) files in the specified folder.

        Args:
            folder: Directory to search for label files.

        Returns:
            Sorted list of label file paths.
        """
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")

        paths = [p for p in folder.iterdir() if p.suffix.lower() == ".txt"]
        return sorted(paths, key=lambda p: p.name.lower())

    @staticmethod
    def find_matching_image(
        label_path: Path,
        images_dir: Path,
        exts: Set[str] = YOLO_CONSTANTS.SUPPORTED_IMAGE_EXTS
    ) -> Optional[Path]:
        """
        Find the image file matching a label file.

        Args:
            label_path: Path to the label file.
            images_dir: Directory containing images.
            exts: Supported image extensions.

        Returns:
            Path to matching image or None if not found.
        """
        stem = label_path.stem
        for ext in exts:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            # Try uppercase extension
            candidate = images_dir / f"{stem}{ext.upper()}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def load_classes(classes_file: Path) -> List[str]:
        """
        Load class names from a classes.txt file.

        Args:
            classes_file: Path to classes.txt.

        Returns:
            List of class names.
        """
        if not classes_file.exists():
            return []

        lines = classes_file.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    @staticmethod
    def save_classes(class_names: List[str], path: Path) -> None:
        """
        Save class names to a file.

        Args:
            class_names: List of class names.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(class_names), encoding="utf-8")

    @staticmethod
    def to_extended_length_path(p: Path) -> Path:
        """
        Return a Path that supports Windows extended-length paths.

        On non-Windows systems, returns the resolved Path unchanged.
        On Windows, prefixes with \\\\?\\ when necessary.

        Args:
            p: Input path.

        Returns:
            Extended-length compatible path.
        """
        p = Path(p).resolve()

        if os.name != "nt":
            return p

        s = str(p)

        # Already extended
        if s.startswith("\\\\?\\"):
            return Path(s)

        # UNC path (network)
        if s.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + s.lstrip("\\"))

        # Normal absolute path
        return Path("\\\\?\\" + s)

    # --------------------------
    # Label parsing and validation
    # --------------------------

    @staticmethod
    def parse_label_line(line: str) -> Tuple[int, List[float]]:
        """
        Parse a YOLO label line.

        Args:
            line: Label line string.

        Returns:
            Tuple of (class_id, coordinates).

        Raises:
            ValueError: If line format is invalid.
        """
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid label line: {line}")

        cls_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]

        return cls_id, coords

    @staticmethod
    def format_label_line(cls_id: int, coords: List[float], precision: int = 6) -> str:
        """
        Format a label line for YOLO.

        Args:
            cls_id: Class ID.
            coords: Coordinate values.
            precision: Decimal precision for coordinates.

        Returns:
            Formatted label line string.
        """
        coord_str = " ".join(f"{c:.{precision}f}" for c in coords)
        return f"{cls_id} {coord_str}"

    @staticmethod
    def detect_label_format(labels_dir: Path) -> str:
        """
        Detect the label format (detect vs OBB) from label files.

        Args:
            labels_dir: Directory containing label files.

        Returns:
            'detect' for standard YOLO, 'obb' for oriented bounding boxes.
        """
        for txt_file in labels_dir.rglob("*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == YOLO_CONSTANTS.LABEL_VALUES_OBB:
                    return YOLO_CONSTANTS.TASK_OBB
                elif len(parts) == YOLO_CONSTANTS.LABEL_VALUES_DETECT:
                    return YOLO_CONSTANTS.TASK_DETECT

        return YOLO_CONSTANTS.TASK_DETECT

    @staticmethod
    def infer_num_classes(labels_dir: Path) -> int:
        """
        Infer the number of classes from label files.

        Args:
            labels_dir: Directory containing label files.

        Returns:
            Number of classes (max class ID + 1).
        """
        max_id = -1

        for txt_file in labels_dir.rglob("*.txt"):
            for line in txt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^(\d+)\s", line)
                if match:
                    cls_id = int(match.group(1))
                    max_id = max(max_id, cls_id)

        return max_id + 1 if max_id >= 0 else 0

    # --------------------------
    # Coordinate transformations
    # --------------------------

    @staticmethod
    def normalize_coords(
        coords: List[float],
        img_width: int,
        img_height: int
    ) -> List[float]:
        """
        Normalize coordinates to [0, 1] range if they are in pixels.

        Args:
            coords: List of coordinates (alternating x, y).
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            Normalized coordinates.
        """
        # Check if already normalized (all values <= 1.0)
        if all(c <= 1.0 for c in coords):
            return coords

        normalized = []
        for i, c in enumerate(coords):
            if i % 2 == 0:  # x coordinate
                normalized.append(c / max(1, img_width))
            else:  # y coordinate
                normalized.append(c / max(1, img_height))

        return normalized

    @staticmethod
    def clamp_coords(coords: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
        """
        Clamp coordinates to a specified range.

        Args:
            coords: List of coordinates.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            Clamped coordinates.
        """
        return [min(max(c, min_val), max_val) for c in coords]

    @staticmethod
    def reorder_obb_points_clockwise(coords: List[float]) -> List[float]:
        """
        Reorder OBB points to clockwise order starting near top-left.

        Args:
            coords: 8 coordinate values (x1, y1, x2, y2, x3, y3, x4, y4).

        Returns:
            Reordered coordinates.
        """
        if len(coords) != 8:
            raise ValueError(f"OBB requires 8 coordinates, got {len(coords)}")

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

        # Calculate centroid
        cx, cy = pts.mean(axis=0)

        # Calculate angles from centroid
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

        # Sort by angle (counter-clockwise)
        order = np.argsort(angles)
        pts = pts[order]

        # Reverse to get clockwise order
        pts = pts[::-1]

        # Start from the point nearest to top-left
        idx0 = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
        pts = np.roll(pts, -idx0, axis=0)

        return pts.reshape(-1).tolist()

    # --------------------------
    # Image operations
    # --------------------------

    @staticmethod
    def get_image_size(image_path: Path) -> Tuple[int, int]:
        """
        Get image dimensions without loading the full image.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (width, height).
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            import cv2
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            h, w = img.shape[:2]
            return w, h

    @staticmethod
    def load_image_cv2(image_path: Path) -> np.ndarray:
        """
        Load an image using OpenCV.

        Args:
            image_path: Path to image file.

        Returns:
            Image as numpy array (BGR format).
        """
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return img

    @staticmethod
    def load_image_pil(image_path: Path):
        """
        Load an image using PIL.

        Args:
            image_path: Path to image file.

        Returns:
            PIL Image object in RGB mode.
        """
        from PIL import Image
        return Image.open(image_path).convert("RGB")
