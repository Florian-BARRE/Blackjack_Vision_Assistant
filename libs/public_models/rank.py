# ====== Code Summary ======
# This module defines the `Rank` enumeration used to represent card ranks in a card game,
# including standard playing cards and special identifiers like Dealer and Switch cards.
# It provides a utility class method to safely parse string inputs into valid Rank values.

from __future__ import annotations

# ====== Standard Library Imports ======
from enum import StrEnum, auto


class Rank(StrEnum):
    """
    Enumeration of card ranks including standard cards, special game-related values,
    and an UNKNOWN fallback for invalid or unexpected values.
    """
    UNKNOWN = auto()  # Used when an invalid or unrecognized string is passed

    # Standard card ranks
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    TEN = '10'
    JACK = 'J'
    QUEEN = 'Q'
    KING = 'K'
    ACE = 'A'

    # Special game-specific ranks
    DEALER_CARD = "DealerCard"  # Used to represent a special dealer card
    SWITCH_CARD = "SwitchCard"  # Used to represent a special switch card

    @classmethod
    def from_str(cls, value: str) -> Rank:
        """
        Safely convert a string into a corresponding Rank enum member.
        Returns UNKNOWN if the input is invalid or unrecognized.

        Args:
            value (str): Input string representing a card rank.

        Returns:
            Rank: Corresponding Rank enum value, or UNKNOWN if invalid.
        """
        # 1. Validate input type
        if not isinstance(value, str):
            return cls.UNKNOWN

        # 2. Attempt to create Rank from string, return UNKNOWN on failure
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN
