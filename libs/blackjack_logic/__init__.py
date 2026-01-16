# ====== Code Summary ======
# Aggregates core blackjack game logic components: detection models, spatial zone detection,
# game state tracking, hand evaluation, and strategy recommendation utilities.

# ------------------- Detection ------------------- #
from .detection import (
    DetectedCard,
    DetectedObject,
    FrameDetections,
    CardType,
    DetectionParser,
)

# --------------------- Zones --------------------- #
from .zones import (
    Zone,
    ZoneDetector,
    TableLayout,
    ZoneState,
)

# --------------------- Hands --------------------- #
from .hand import Hand

# ------------------ Game State ------------------- #
from .game import (
    PowerBlackjackState,
    GamePhase, SpecialEvent,
    CardPair,
)

# ------------------- Strategy -------------------- #
from .strategy import (
    PowerBlackjackStrategy,
    Action,
    StrategyResult,
)

# ------------------- Public API ------------------- #
__all__ = [
    "DetectedCard",
    "DetectedObject",
    "FrameDetections",
    "CardType",
    "DetectionParser",

    "Zone",
    "ZoneDetector",
    "TableLayout",
    "ZoneState",

    "Hand",

    "PowerBlackjackState",
    "GamePhase", "SpecialEvent",
    "CardPair",

    "PowerBlackjackStrategy",
    "Action",
    "StrategyResult",
]
