# ====== Code Summary ======
# Defines color mappings for object detection overlays in OpenCV.
# Supports mapping object types (`ObbType`) and special ranks (`Rank`)
# to specific BGR colors used in rendering visual annotations.

"""Color definitions for overlay rendering."""

# ====== Standard Library Imports ======
from dataclasses import dataclass
from typing import Tuple

# ====== Internal Project Imports ======
from public_models.obb_type import ObbType
from public_models.rank import Rank


@dataclass
class OverlayColors:
    """
    Color scheme for rendering overlays using OpenCV (BGR format).

    Includes mappings for object types, zones, and special card ranks.
    """

    # OBB type colors
    CARD: Tuple[int, int, int] = (0, 255, 0)            # Green
    CARD_HOLDER: Tuple[int, int, int] = (255, 0, 255)   # Magenta
    TRAP: Tuple[int, int, int] = (0, 0, 255)            # Red
    UNKNOWN: Tuple[int, int, int] = (128, 128, 128)     # Gray

    # Special rank colors
    DEALER_CARD: Tuple[int, int, int] = (0, 165, 255)   # Orange
    SWITCH_CARD: Tuple[int, int, int] = (255, 255, 0)   # Cyan-yellow

    # Zone highlight colors
    DEALER_ZONE: Tuple[int, int, int] = (255, 100, 100) # Light red
    PLAYER_ZONE: Tuple[int, int, int] = (100, 255, 100) # Light green

    def get_obb_color(self, obb_type: ObbType) -> Tuple[int, int, int]:
        """
        Get the color used for a given object type (OBB).

        Args:
            obb_type (ObbType): The object type to render.

        Returns:
            Tuple[int, int, int]: Corresponding BGR color.
        """
        # 1. Define mapping of OBB type to color
        mapping = {
            ObbType.CARD: self.CARD,
            ObbType.CARD_HOLDER: self.CARD_HOLDER,
            ObbType.TRAP: self.TRAP,
        }

        # 2. Return mapped color or fallback to UNKNOWN
        return mapping.get(obb_type, self.UNKNOWN)

    def get_rank_color(self, rank: Rank) -> Tuple[int, int, int] | None:
        """
        Get the color used for special card ranks (DealerCard, SwitchCard).

        Args:
            rank (Rank): The card rank to render.

        Returns:
            Tuple[int, int, int] | None: Corresponding BGR color or None for normal ranks.
        """
        # 1. Match special ranks
        if rank == Rank.DEALER_CARD:
            return self.DEALER_CARD
        elif rank == Rank.SWITCH_CARD:
            return self.SWITCH_CARD

        # 2. Return None for non-special ranks
        return None
