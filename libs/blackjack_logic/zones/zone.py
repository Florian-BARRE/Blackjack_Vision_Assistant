# ====== Code Summary ======
# Defines a rectangular zone representation on a blackjack table and utilities for
# spatial inclusion checks of points or detected cards within that zone.

# ====== Standard Library Imports ======
from dataclasses import dataclass
from typing import Tuple

# ====== Internal Project Imports ======
from ..detection import DetectedCard


@dataclass
class Zone:
    """
    A rectangular zone on the blackjack table used for spatial classification.

    Attributes:
        x (int): Top-left x-coordinate of the zone.
        y (int): Top-left y-coordinate of the zone.
        width (int): Width of the zone.
        height (int): Height of the zone.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """
        Return the rectangle as a 4-tuple.

        Returns:
            tuple[int, int, int, int]: (x, y, width, height)
        """
        return (self.x, self.y, self.width, self.height)

    @property
    def bottom(self) -> int:
        """
        Get the bottom boundary of the zone.

        Returns:
            int: y + height
        """
        return self.y + self.height

    @property
    def right(self) -> int:
        """
        Get the right boundary of the zone.

        Returns:
            int: x + width
        """
        return self.x + self.width

    def contains(self, px: float, py: float) -> bool:
        """
        Check if a point is within the zone boundaries.

        Args:
            px (float): X-coordinate of the point.
            py (float): Y-coordinate of the point.

        Returns:
            bool: True if the point is within the zone, else False.
        """
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def contains_point(self, px: float, py: float) -> bool:
        """
        Alias for contains(), for clarity when checking points.

        Args:
            px (float): X-coordinate.
            py (float): Y-coordinate.

        Returns:
            bool: True if inside zone.
        """
        return self.contains(px, py)

    def contains_card(self, card: DetectedCard) -> bool:
        """
        Check if a detected card's center lies within this zone.

        Args:
            card (DetectedCard): The card to check.

        Returns:
            bool: True if the card is inside the zone.
        """
        return self.contains(card.x, card.y)