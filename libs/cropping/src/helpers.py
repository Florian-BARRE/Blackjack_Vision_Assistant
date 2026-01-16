# ====== Code Summary ======
# Utility functions for handling image files: listing images from a folder,
# loading them with consistent format, and validating crop coordinates.

from __future__ import annotations

# ====== Standard Library Imports ======
import os
from pathlib import Path

# ====== Third-Party Library Imports ======
from PIL import Image

# ====== Local Project Imports ======
from .constants import CROPPING_CONSTANTS


class CROPPING_HELPERS:
    @staticmethod
    def list_images(folder: Path, exts: set[str] = CROPPING_CONSTANTS.SUPPORTED_EXTS) -> list[Path]:
        """
        List all supported image files in the specified folder.

        Args:
            folder (Path): Directory to search for image files.
            exts (set[str], optional): Allowed file extensions (lowercase, with leading dot). Defaults to SUPPORTED_EXTS.

        Raises:
            FileNotFoundError: If the provided folder does not exist.

        Returns:
            list[Path]: Sorted list of image file paths, case-insensitively sorted by name.
        """
        # 1. Check if folder exists
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder.resolve()}")

        # 2. Filter for files with supported extensions
        paths: list[Path] = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

        # 3. Return sorted list by file name (case-insensitive)
        return sorted(paths, key=lambda p: p.name.lower())

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        """
        Load an image from the given path and convert it to RGB mode.

        Args:
            path (Path): Path to the image file.

        Returns:
            Image.Image: The image loaded in RGB mode.
        """
        # 1. Open the image and convert to RGB mode
        return Image.open(path).convert("RGB")

    @staticmethod
    def validate_crop(
            left: int,
            top: int,
            right: int,
            bottom: int,
            width: int,
            height: int
    ) -> None:
        """
        Validate whether the crop box is within the image boundaries.

        Args:
            left (int): Left boundary of the crop box.
            top (int): Top boundary of the crop box.
            right (int): Right boundary of the crop box.
            bottom (int): Bottom boundary of the crop box.
            width (int): Width of the image.
            height (int): Height of the image.

        Raises:
            ValueError: If the horizontal or vertical crop bounds are invalid.
        """
        # 1. Validate horizontal bounds
        if not (0 <= left < right <= width):
            raise ValueError(
                f"Invalid horizontal crop: left={left}, right={right}, width={width}"
            )

        # 2. Validate vertical bounds
        if not (0 <= top < bottom <= height):
            raise ValueError(
                f"Invalid vertical crop: top={top}, bottom={bottom}, height={height}"
            )

    @staticmethod
    def to_extended_length_path(p: Path) -> Path:
        """
        Return a Path that supports Windows extended-length paths.

        - On non-Windows systems: returns the resolved Path unchanged
        - On Windows: prefixes with \\\\?\\ when necessary
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
