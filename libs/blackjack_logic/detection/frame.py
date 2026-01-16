# ====== Code Summary ======
# Container model representing all object and card detections for a single frame.
# Supports access to special cards (dealer/switch), player cards, and sorted lists
# based on spatial coordinates.

# ====== Standard Library Imports ======
from dataclasses import dataclass, field

# ====== Internal Project Imports ======
from public_models.rank import Rank
from .card import DetectedCard
from .object import DetectedObject


@dataclass
class FrameDetections:
    """
    All detections from a single frame, grouped by type.

    Attributes:
        cards (list[DetectedCard]): All detected cards.
        card_holders (list[DetectedObject]): Detected card holders.
        traps (list[DetectedObject]): Detected trap devices or markers.
    """

    cards: list[DetectedCard] = field(default_factory=list)
    card_holders: list[DetectedObject] = field(default_factory=list)
    traps: list[DetectedObject] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """
        Total number of detected elements in the frame.

        Returns:
            int: Sum of all card, holder, and trap detections.
        """
        return len(self.cards) + len(self.card_holders) + len(self.traps)

    @property
    def dealer_cards(self) -> list[DetectedCard]:
        """
        List of cards identified as dealer cards.

        Returns:
            list[DetectedCard]: Dealer card detections.
        """
        return [c for c in self.cards if c.rank == Rank.DEALER_CARD]

    @property
    def switch_cards(self) -> list[DetectedCard]:
        """
        List of cards identified as switch cards.

        Returns:
            list[DetectedCard]: Switch card detections.
        """
        return [c for c in self.cards if c.rank == Rank.SWITCH_CARD]

    @property
    def playing_cards(self) -> list[DetectedCard]:
        """
        List of standard (non-special) playing cards.

        Returns:
            list[DetectedCard]: Regular playing cards.
        """
        return [c for c in self.cards if not c.is_special]

    def get_cards_sorted_by_x(self) -> list[DetectedCard]:
        """
        Get all cards sorted by horizontal position.

        Returns:
            list[DetectedCard]: Cards sorted left to right.
        """
        return sorted(self.cards, key=lambda c: c.x)

    def get_cards_sorted_by_y(self) -> list[DetectedCard]:
        """
        Get all cards sorted by vertical position.

        Returns:
            list[DetectedCard]: Cards sorted top to bottom.
        """
        return sorted(self.cards, key=lambda c: c.y)
