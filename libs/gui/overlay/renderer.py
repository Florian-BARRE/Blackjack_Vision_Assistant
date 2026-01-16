# ====== Code Summary ======
# Defines a centralized color scheme for overlay rendering in OpenCV (BGR format).
# Supports consistent coloring of object types (cards, holders, traps),
# special card ranks (DealerCard, SwitchCard), and gameplay zones.

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
    Color scheme for overlay elements, using BGR tuples for OpenCV rendering.

    Includes mappings for object bounding box (OBB) types, special card ranks,
    and player/dealer zones.
    """

    # OBB type colors
    CARD: Tuple[int, int, int] = (0, 255, 0)            # Green for cards
    CARD_HOLDER: Tuple[int, int, int] = (255, 0, 255)   # Magenta for card holders
    TRAP: Tuple[int, int, int] = (0, 0, 255)            # Red for traps
    UNKNOWN: Tuple[int, int, int] = (128, 128, 128)     # Gray for unknown types

    # Special rank colors
    DEALER_CARD: Tuple[int, int, int] = (0, 165, 255)   # Orange for DealerCard
    SWITCH_CARD: Tuple[int, int, int] = (255, 255, 0)   # Cyan-yellow for SwitchCard

    # Zone highlight colors
    DEALER_ZONE: Tuple[int, int, int] = (255, 100, 100) # Light red for dealer zone
    PLAYER_ZONE: Tuple[int, int, int] = (100, 255, 100) # Light green for player zone

    def get_obb_color(self, obb_type: ObbType) -> Tuple[int, int, int]:
        """
        Get color for a given object type (OBB).

        Args:
            obb_type (ObbType): Object type from detection.

        Returns:
            Tuple[int, int, int]: Corresponding BGR color for rendering.
        """
        # 1. Define mapping from object type to color
        mapping = {
            ObbType.CARD: self.CARD,
            ObbType.CARD_HOLDER: self.CARD_HOLDER,
            ObbType.TRAP: self.TRAP,
        }

        # 2. Return matching color or fallback to UNKNOWN
        return mapping.get(obb_type, self.UNKNOWN)

    def get_rank_color(self, rank: Rank) -> Tuple[int, int, int] | None:
        """
        Get color for a special card rank if applicable.

        Args:
            rank (Rank): Card rank enum.

        Returns:
            Tuple[int, int, int] | None: Color for special rank, or None if normal rank.
        """
        # 1. Check if the rank is a known special type
        if rank == Rank.DEALER_CARD:
            return self.DEALER_CARD
        elif rank == Rank.SWITCH_CARD:
            return self.SWITCH_CARD

        # 2. No color for standard card ranks
        return None
