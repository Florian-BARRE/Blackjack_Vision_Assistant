# ====== Code Summary ======
# Defines the CropConfig dataclass for managing image cropping configurations.
# Includes methods to serialize/deserialize from JSON for reproducible ML workflows.

from __future__ import annotations

# ====== Standard Library Imports ======
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CropConfig:
    """Cropping configuration in pixel coordinates.

    The crop box follows the PIL convention: (left, top, right, bottom),
    where left/top are inclusive and right/bottom are exclusive.

    This object is intentionally small and serializable to ensure
    reproducibility in ML experiments.
    """

    left: int
    top: int
    right: int
    bottom: int
    mode: str = "pixels"

    def to_box(self) -> tuple[int, int, int, int]:
        """
        Convert the crop configuration to a box tuple compatible with PIL.

        Returns:
            tuple[int, int, int, int]: A 4-tuple (left, top, right, bottom).
        """
        # 1. Return the crop box in PIL format
        return self.left, self.top, self.right, self.bottom

    def to_json_dict(self) -> dict[str, Any]:
        """
        Serialize the configuration into a dictionary suitable for JSON export.

        Returns:
            dict[str, Any]: Dictionary with crop configuration and metadata.
        """
        # 1. Return a dictionary representation with metadata
        return {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "crop": {
                "left": int(self.left),
                "top": int(self.top),
                "right": int(self.right),
                "bottom": int(self.bottom),
            },
        }

    def save(self, path: Path) -> None:
        """
        Save this configuration to a JSON file.

        Args:
            path (Path): Target file path.
        """
        # 1. Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Serialize to JSON and write to file
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> CropConfig:
        """
        Load a CropConfig object from a JSON file.

        Args:
            path (Path): Path to the JSON file.

        Returns:
            CropConfig: Reconstructed configuration object.
        """
        # 1. Read and parse JSON content
        data = json.loads(path.read_text(encoding="utf-8"))

        # 2. Extract crop region and mode
        crop = data["crop"]
        mode = data.get("mode", "pixels")

        # 3. Create and return CropConfig instance
        return CropConfig(
            left=int(crop["left"]),
            top=int(crop["top"]),
            right=int(crop["right"]),
            bottom=int(crop["bottom"]),
            mode=str(mode),
        )
