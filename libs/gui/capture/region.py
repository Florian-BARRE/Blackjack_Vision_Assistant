# ====== Code Summary ======
# Provides utilities for managing screen capture regions and screen dimensions.
# Includes a data class `CaptureRegion` for bounding rectangles,
# and `ScreenInfo` with methods to query primary monitor properties and default capture area.

from __future__ import annotations

# ====== Standard Library Imports ======
from dataclasses import dataclass
from typing import Tuple

# ====== Third-Party Library Imports ======
import mss


@dataclass
class CaptureRegion:
    """
    Rectangular region definition for screen capture using pixel coordinates.

    Attributes:
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        width (int): Width of the capture region.
        height (int): Height of the capture region.
    """

    x: int
    y: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict:
        """
        Convert the region into a dict compatible with `mss` library.

        Returns:
            dict: Dictionary with `left`, `top`, `width`, and `height` keys.
        """
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }

    def clamp_to_screen(self, screen_width: int, screen_height: int) -> CaptureRegion:
        """
        Clamp the capture region to stay within screen bounds and enforce minimum size.

        Args:
            screen_width (int): Width of the screen.
            screen_height (int): Height of the screen.

        Returns:
            CaptureRegion: Clamped region constrained to screen and min size of 100×100.
        """
        # 1. Clamp position
        x = max(0, min(self.x, screen_width - self.width))
        y = max(0, min(self.y, screen_height - self.height))

        # 2. Clamp size to remaining space from (x, y)
        w = min(self.width, screen_width - x)
        h = min(self.height, screen_height - y)

        # 3. Enforce minimum size
        return CaptureRegion(x, y, max(100, w), max(100, h))


class ScreenInfo:
    """
    Utility class for retrieving screen-related information.
    """

    @staticmethod
    def get_primary_monitor_size() -> Tuple[int, int]:
        """
        Get the dimensions of the primary monitor.

        Returns:
            Tuple[int, int]: (width, height) of the primary monitor.
        """
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Index 1 = primary monitor
            return monitor["width"], monitor["height"]

    @staticmethod
    def get_default_region(height_ratio: float = 0.4) -> CaptureRegion:
        """
        Generate a default capture region anchored to the bottom of the screen.

        Args:
            height_ratio (float): Ratio of the screen height to include in the region.

        Returns:
            CaptureRegion: Bottom screen capture region with full width.
        """
        # 1. Get screen size
        sw, sh = ScreenInfo.get_primary_monitor_size()

        # 2. Calculate height based on ratio
        h = int(sh * height_ratio)

        # 3. Return region anchored to bottom
        return CaptureRegion(0, sh - h, sw, h)
