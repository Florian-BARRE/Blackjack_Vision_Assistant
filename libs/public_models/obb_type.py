# ====== Code Summary ======
# This module defines the `ObbType` enumeration to categorize object types
# (e.g., cards, card holders, traps) in a game or application.
# It includes a utility method to safely parse strings into enum values,
# falling back to UNKNOWN for invalid inputs.

from __future__ import annotations

# ====== Standard Library Imports ======
from enum import StrEnum, auto


class ObbType(StrEnum):
    """
    Enumeration of object types that may appear in a game environment,
    such as cards, card holders, and traps, with a fallback UNKNOWN type.
    """
    UNKNOWN = auto()  # Used when an invalid or unrecognized string is passed

    CARD = "Card"  # Represents a standard card object
    CARD_HOLDER = "Card_Holder"  # Represents a card holder or container
    TRAP = "Trap"  # Represents a trap object in the environment

    @classmethod
    def from_str(cls, value: str) -> ObbType:
        """
        Safely convert a string into a corresponding ObbType enum member.
        Returns UNKNOWN if the input is invalid or unrecognized.

        Args:
            value (str): Input string representing an object type.

        Returns:
            ObbType: Corresponding ObbType enum value, or UNKNOWN if invalid.
        """
        # 1. Validate input type
        if not isinstance(value, str):
            return cls.UNKNOWN

        # 2. Attempt to create ObbType from string, return UNKNOWN on failure
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN
