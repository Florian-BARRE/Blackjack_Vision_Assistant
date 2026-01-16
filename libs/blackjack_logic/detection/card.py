# ====== Code Summary ======
# Defines data structures for detected card entities used in blackjack analysis.
# Includes positional metadata, detection confidence, card type classification, and derived properties.

# ====== Standard Library Imports ======
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

# ====== Internal Project Imports ======
from public_models.rank import Rank


class CardType(Enum):
    """
    Type of card based on position within the blackjack context.

    Attributes:
        PLAYER: Player-side detected card.
        DEALER: Dealer-side detected card.
        UNKNOWN: Unclassified or unassigned position.
    """
    PLAYER = auto()
    DEALER = auto()
    UNKNOWN = auto()


@dataclass
class DetectedCard:
    """
    A detected card with its classification and geometric metadata.

    Attributes:
        rank (Rank): The detected rank of the card.
        confidence (float): Confidence score from the model [0.0 - 1.0].
        box (np.ndarray): Oriented bounding box as a (4, 2) NumPy array.
        card_type (CardType): Classification of card side (player/dealer/unknown).
    """

    rank: Rank
    confidence: float
    box: np.ndarray
    card_type: CardType = CardType.UNKNOWN

    @property
    def x(self) -> float:
        """
        X-coordinate of the card center (averaged from OBB corners).

        Returns:
            float: Horizontal center.
        """
        return float(np.mean(self.box[:, 0]))

    @property
    def y(self) -> float:
        """
        Y-coordinate of the card center (averaged from OBB corners).

        Returns:
            float: Vertical center.
        """
        return float(np.mean(self.box[:, 1]))

    @property
    def is_special(self) -> bool:
        """
        Whether the card is a special non-standard rank.

        Returns:
            bool: True if card is DEALER_CARD or SWITCH_CARD.
        """
        return self.rank in (Rank.DEALER_CARD, Rank.SWITCH_CARD)
