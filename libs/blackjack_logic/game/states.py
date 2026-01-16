from enum import Enum, auto


class GamePhase(Enum):
    """Current phase of the game."""

    WAITING = auto()
    DEALING = auto()
    PLAYER_TURN = auto()
    DEALER_TURN = auto()
    FINISHED = auto()


class SpecialEvent(Enum):
    """Special events detected."""

    NONE = auto()
    DEALER_SWITCH = auto()  # DealerCard visible - dealer is switching
    SWITCH_WARNING = auto()  # SwitchCard visible - card holder will change soon
    HOLDER_CHANGING = auto()  # CardHolder moved out of zone

