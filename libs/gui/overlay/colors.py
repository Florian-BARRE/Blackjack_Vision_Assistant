"""Color definitions for overlay rendering."""

from dataclasses import dataclass
from typing import Tuple

from public_models.obb_type import ObbType
from public_models.rank import Rank


@dataclass
class OverlayColors:
    """Color scheme for overlay (BGR format for OpenCV)."""

    # OBB type colors
    CARD: Tuple[int, int, int] = (0, 255, 0)
    CARD_HOLDER: Tuple[int, int, int] = (255, 0, 255)
    TRAP: Tuple[int, int, int] = (0, 0, 255)
    UNKNOWN: Tuple[int, int, int] = (128, 128, 128)

    # Special rank colors
    DEALER_CARD: Tuple[int, int, int] = (0, 165, 255)
    SWITCH_CARD: Tuple[int, int, int] = (255, 255, 0)

    # Zone colors
    DEALER_ZONE: Tuple[int, int, int] = (255, 100, 100)
    PLAYER_ZONE: Tuple[int, int, int] = (100, 255, 100)

    def get_obb_color(self, obb_type: ObbType) -> Tuple[int, int, int]:
        """Get color for OBB type."""
        mapping = {
            ObbType.CARD: self.CARD,
            ObbType.CARD_HOLDER: self.CARD_HOLDER,
            ObbType.TRAP: self.TRAP,
        }
        return mapping.get(obb_type, self.UNKNOWN)

    def get_rank_color(self, rank: Rank) -> Tuple[int, int, int]:
        """Get color for special rank types (DealerCard, SwitchCard)."""
        if rank == Rank.DEALER_CARD:
            return self.DEALER_CARD
        elif rank == Rank.SWITCH_CARD:
            return self.SWITCH_CARD
        return None