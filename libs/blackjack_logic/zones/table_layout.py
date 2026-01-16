# ====== Code Summary ======
# Defines a `TableLayout` dataclass representing detected spatial zones on a blackjack table,
# including dealer, player, and card holder areas, along with the current detection state.

# ====== Standard Library Imports ======
from dataclasses import dataclass
from typing import Optional

# ====== Internal Project Imports ======
from .zone import Zone
from .states import ZoneState


@dataclass
class TableLayout:
    """
    Detected table layout with spatial zones and detection state.

    Attributes:
        dealer_zone (Optional[Zone]): Zone allocated for dealer cards.
        player_zone (Optional[Zone]): Zone allocated for player cards.
        holder_zone (Optional[Zone]): Zone used to track card holder area.
        state (ZoneState): Current detection state of the layout.
    """

    dealer_zone: Optional[Zone] = None
    player_zone: Optional[Zone] = None
    holder_zone: Optional[Zone] = None  # Zone for tracking card holder position
    state: ZoneState = ZoneState.NOT_INITIALIZED