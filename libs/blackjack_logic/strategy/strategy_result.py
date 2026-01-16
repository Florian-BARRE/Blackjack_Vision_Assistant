# ====== Code Summary ======
# Defines strategy-related enums and result data structures for Power Blackjack,
# including player actions and a structured result type for grid-based strategy lookups.

# ====== Standard Library Imports ======
from __future__ import annotations

from dataclasses import dataclass

# ====== Local Project Imports ======
from .action import Action


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """
    Result of a strategy grid lookup.

    Attributes:
        action (Action): The recommended optimal action.
        grid_type (str): Grid category ("HARD", "SOFT", "PAIRS").
        player_key (str): Row label (e.g. "16", "A,7", "8,8").
        dealer_key (str): Column label (e.g. "7", "A").
        row_index (int): Index of the row in the strategy grid.
        col_index (int): Index of the column in the strategy grid.
    """

    action: Action
    grid_type: str
    player_key: str
    dealer_key: str
    row_index: int
    col_index: int
