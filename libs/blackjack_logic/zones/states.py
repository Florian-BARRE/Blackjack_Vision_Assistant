# ====== Code Summary ======
# Defines the state machine for zone detection phases in the blackjack analysis system.
# Represents different lifecycle stages a detection zone may go through.

# ====== Standard Library Imports ======
from enum import Enum, auto


class ZoneState(Enum):
    """
    State of zone detection during processing lifecycle.

    Attributes:
        NOT_INITIALIZED: Zone has not been initialized or observed yet.
        DETECTING: Zone is currently in detection phase.
        INITIALIZED: Zone has been established and recognized.
        LOCKED: Zone is finalized and no longer updated.
    """
    NOT_INITIALIZED = auto()
    DETECTING = auto()
    INITIALIZED = auto()
    LOCKED = auto()
