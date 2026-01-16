# ====== Code Summary ======
# Defines strategy actions and enumerations for Power Blackjack gameplay,
# including supported player decisions such as HIT, STAND, SPLIT, and QUADRUPLE.

# ====== Standard Library Imports ======
from __future__ import annotations

from enum import Enum


class Action(Enum):
    """
    Possible actions a player can take during a Power Blackjack hand.

    Attributes:
        HIT (str): Take another card.
        STAND (str): Hold current hand.
        SPLIT (str): Split a pair into two separate hands.
        QUADRUPLE (str): Multiply bet by 4, take one card, and stand.
    """

    HIT = "H"
    STAND = "S"
    SPLIT = "P"
    QUADRUPLE = "4"  # x4 bet, 1 card then stand
