# ====== Code Summary ======
# Implements Power Blackjack basic strategy using precomputed grids for HARD, SOFT, and PAIRS hands.
# Provides lookup methods that return structured strategy recommendations (`StrategyResult`)
# based on player cards and dealer upcard. Includes display utilities and rule-specific adaptations
# (e.g., no 9s/10s, quadruple down rules, soft 17 stand).

# ====== Standard Library Imports ======
from __future__ import annotations

# ====== Third-Party Library Imports ======
from loggerplusplus import LoggerClass

# ====== Internal Project Imports ======
from public_models.rank import Rank

# ====== Local Project Imports ======
from .strategy_result import StrategyResult
from .action import Action


class PowerBlackjackStrategy(LoggerClass):
    """
    Power Blackjack Basic Strategy (no 9 & 10).

    Rules:
        - 8 decks
        - 9 & 10 removed (dealer "T" represents 10/J/Q/K)
        - Dealer stands on soft 17

    This class is a pure lookup utility backed by static grids.
    It is implemented as a class with classmethods for compatibility.
    """

    # Dealer upcard columns: 2, 3, 4, 5, 6, 7, 8, T, A
    DEALER_CARDS: list[str] = ["2", "3", "4", "5", "6", "7", "8", "T", "A"]

    # --------------------- HARD Totals --------------------- #
    HARD_ROWS: list[str] = [str(n) for n in range(5, 21)]
    HARD_GRID: list[list[str]] = [
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 5
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 6
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 7
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 8
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 9
        ["4", "4", "4", "4", "4", "4", "4", "4", "H"],  # 10
        ["4", "4", "4", "4", "4", "4", "4", "4", "H"],  # 11
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 12
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 13
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],  # 14
        ["H", "H", "H", "S", "H", "H", "H", "H", "H"],  # 15
        ["S", "H", "S", "S", "S", "H", "H", "H", "H"],  # 16
        ["S", "S", "S", "S", "S", "S", "S", "S", "H"],  # 17
        ["S", "S", "S", "S", "S", "S", "S", "S", "S"],  # 18
        ["S", "S", "S", "S", "S", "S", "S", "S", "S"],  # 19
        ["S", "S", "S", "S", "S", "S", "S", "S", "S"],  # 20
    ]

    # --------------------- SOFT Totals --------------------- #
    SOFT_ROWS: list[str] = ["A,2", "A,3", "A,4", "A,5", "A,6", "A,7", "A,8", "A,T"]
    SOFT_GRID: list[list[str]] = [
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],
        ["H", "H", "H", "H", "H", "H", "H", "H", "H"],
        ["H", "H", "H", "H", "4", "H", "H", "H", "H"],
        ["H", "H", "H", "H", "4", "H", "H", "H", "H"],
        ["H", "H", "H", "4", "4", "H", "H", "H", "H"],
        ["4", "S", "4", "4", "4", "S", "H", "H", "H"],
        ["S", "S", "S", "S", "4", "S", "S", "S", "S"],
        ["S", "S", "S", "S", "S", "S", "S", "S", "S"],
    ]

    # ----------------------- PAIRS ------------------------ #
    PAIRS_ROWS: list[str] = ["A,A", "2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "T,T"]
    PAIRS_GRID: list[list[str]] = [
        ["P", "P", "P", "P", "P", "H", "P", "P", "H"],
        ["H", "H", "H", "H", "P", "P", "H", "H", "H"],
        ["H", "H", "P", "P", "P", "P", "P", "H", "H"],
        ["H", "H", "H", "H", "P", "P", "H", "H", "H"],
        ["4", "4", "4", "4", "4", "4", "4", "H", "H"],
        ["P", "P", "P", "P", "P", "P", "H", "H", "H"],
        ["P", "P", "P", "P", "P", "P", "H", "H", "H"],
        ["P", "P", "P", "P", "P", "P", "P", "P", "P"],
        ["S", "S", "S", "S", "S", "S", "S", "S", "S"],
    ]

    def __init__(self) -> None:
        """
        Initialize the strategy helper.

        This class is primarily used via classmethods, but instantiation is supported
        to comply with projects that expect LoggerClass inheritance.
        """
        super().__init__()
        self.logger.debug("START initializing PowerBlackjackStrategy")
        self.logger.debug("END initializing PowerBlackjackStrategy")

    @classmethod
    def get_dealer_col(cls, rank: Rank) -> int:
        """
        Get column index for dealer upcard.

        Args:
            rank (Rank): Dealer upcard rank.

        Returns:
            int: Column index in the strategy grids, or -1 if unsupported.
        """
        mapping: dict[Rank, int] = {
            Rank.TWO: 0,
            Rank.THREE: 1,
            Rank.FOUR: 2,
            Rank.FIVE: 3,
            Rank.SIX: 4,
            Rank.SEVEN: 5,
            Rank.EIGHT: 6,
            Rank.JACK: 7,
            Rank.QUEEN: 7,
            Rank.KING: 7,
            Rank.TEN: 7,
            Rank.ACE: 8,
        }
        return mapping.get(rank, -1)

    @classmethod
    def get_dealer_display(cls, rank: Rank) -> str:
        """
        Get display string for dealer card.

        Args:
            rank (Rank): Dealer rank.

        Returns:
            str: Display string ("2"-"8", "T", "A", or "?").
        """
        if rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
            return "T"
        return rank.value if rank != Rank.UNKNOWN else "?"

    @classmethod
    def rank_value(cls, rank: Rank) -> int:
        """
        Get numeric value of a rank for blackjack totals.

        Args:
            rank (Rank): Rank enum.

        Returns:
            int: Numeric blackjack value.
        """
        if rank == Rank.ACE:
            return 11
        if rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
            return 10
        try:
            return int(rank.value)
        except ValueError:
            return 0

    @classmethod
    def lookup(cls, player_ranks: list[Rank], dealer_rank: Rank) -> StrategyResult | None:
        """
        Look up the optimal action from player and dealer cards.

        Args:
            player_ranks (list[Rank]): Player card ranks.
            dealer_rank (Rank): Dealer's visible card.

        Returns:
            StrategyResult | None: Recommended strategy, or None if indeterminate.
        """
        # 1. Sanitize inputs
        if not player_ranks or dealer_rank == Rank.UNKNOWN:
            return None

        ranks = [r for r in player_ranks if r != Rank.UNKNOWN]
        if not ranks:
            return None

        dealer_col = cls.get_dealer_col(dealer_rank)
        if dealer_col < 0:
            return None

        dealer_key = cls.DEALER_CARDS[dealer_col]

        # 2. Check for Pairs
        if len(ranks) == 2:
            v1 = cls.rank_value(ranks[0])
            v2 = cls.rank_value(ranks[1])
            if v1 == v2:
                result = cls._lookup_pairs(ranks, dealer_col, dealer_key)
                if result:
                    return result

        # 3. Check for Soft Hand
        if any(r == Rank.ACE for r in ranks) and len(ranks) == 2:
            result = cls._lookup_soft(ranks, dealer_col, dealer_key)
            if result:
                return result

        # 4. Default to Hard Hand
        return cls._lookup_hard(ranks, dealer_col, dealer_key)

    @classmethod
    def _lookup_hard(cls, ranks: list[Rank], dealer_col: int, dealer_key: str) -> StrategyResult | None:
        """
        Lookup action in HARD totals grid.

        Args:
            ranks (list[Rank]): Player cards.
            dealer_col (int): Dealer column index.
            dealer_key (str): Dealer label.

        Returns:
            StrategyResult | None
        """
        # 1. Calculate total, adjusting Aces from 11 to 1 if needed
        total = sum(cls.rank_value(r) for r in ranks)
        aces = sum(1 for r in ranks if r == Rank.ACE)

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        # 2. Clamp total within supported range
        total = max(5, min(20, total))
        row_key = str(total)

        if row_key not in cls.HARD_ROWS:
            return None

        row_idx = cls.HARD_ROWS.index(row_key)
        action_str = cls.HARD_GRID[row_idx][dealer_col]

        return StrategyResult(
            action=Action(action_str),
            grid_type="HARD",
            player_key=row_key,
            dealer_key=dealer_key,
            row_index=row_idx,
            col_index=dealer_col,
        )

    @classmethod
    def _lookup_soft(cls, ranks: list[Rank], dealer_col: int, dealer_key: str) -> StrategyResult | None:
        """
        Lookup action in SOFT totals grid.

        Args:
            ranks (list[Rank]): Two-card hand including an Ace.
            dealer_col (int): Dealer column index.
            dealer_key (str): Dealer label.

        Returns:
            StrategyResult | None
        """
        other = next((r for r in ranks if r != Rank.ACE), None)
        if other is None:
            return None

        val = cls.rank_value(other)
        row_key = "A,T" if val >= 10 else f"A,{val}"

        if row_key not in cls.SOFT_ROWS:
            return None

        row_idx = cls.SOFT_ROWS.index(row_key)
        action_str = cls.SOFT_GRID[row_idx][dealer_col]

        return StrategyResult(
            action=Action(action_str),
            grid_type="SOFT",
            player_key=row_key,
            dealer_key=dealer_key,
            row_index=row_idx,
            col_index=dealer_col,
        )

    @classmethod
    def _lookup_pairs(cls, ranks: list[Rank], dealer_col: int, dealer_key: str) -> StrategyResult | None:
        """
        Lookup action in PAIRS grid.

        Args:
            ranks (list[Rank]): Two-card pair.
            dealer_col (int): Dealer column index.
            dealer_key (str): Dealer label.

        Returns:
            StrategyResult | None
        """
        rank = ranks[0]
        val = cls.rank_value(rank)

        if rank == Rank.ACE:
            row_key = "A,A"
        elif val >= 10:
            row_key = "T,T"
        else:
            row_key = f"{val},{val}"

        if row_key not in cls.PAIRS_ROWS:
            return None

        row_idx = cls.PAIRS_ROWS.index(row_key)
        action_str = cls.PAIRS_GRID[row_idx][dealer_col]

        return StrategyResult(
            action=Action(action_str),
            grid_type="PAIRS",
            player_key=row_key,
            dealer_key=dealer_key,
            row_index=row_idx,
            col_index=dealer_col,
        )

    @classmethod
    def get_action_display(cls, action: Action) -> tuple[str, str]:
        """
        Get human-readable label and display color for an action.

        Args:
            action (Action): Strategy action.

        Returns:
            tuple[str, str]: (label, hex_color)
        """
        if action == Action.HIT:
            return "HIT", "#a6e3a1"
        if action == Action.STAND:
            return "STAND", "#89b4fa"
        if action == Action.SPLIT:
            return "SPLIT", "#cba6f7"
        if action == Action.QUADRUPLE:
            return "x4", "#f9e2af"
        return "?", "#6c7086"
