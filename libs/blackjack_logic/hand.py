"""Hand representation and value calculation."""

from dataclasses import dataclass, field
from typing import List

from public_models.rank import Rank
from .detection import DetectedCard


@dataclass
class Hand:
    """A blackjack hand with value calculation."""
    cards: List[DetectedCard] = field(default_factory=list)

    @property
    def ranks(self) -> List[str]:
        return [c.rank.value for c in self.cards if not c.is_special and c.rank != Rank.UNKNOWN]

    @property
    def value(self) -> int:
        """Calculate hand value using blackjack rules."""
        total = 0
        aces = 0

        for card in self.cards:
            if card.is_special or card.rank == Rank.UNKNOWN:
                continue

            rank = card.rank
            if rank == Rank.ACE:
                aces += 1
                total += 11
            elif rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
                total += 10
            else:
                try:
                    total += int(rank.value)
                except ValueError:
                    pass

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21

    @property
    def is_busted(self) -> bool:
        return self.value > 21

    @property
    def is_soft(self) -> bool:
        """Check if hand has a usable ace (counted as 11)."""
        total_hard = 0
        aces = 0
        for card in self.cards:
            if card.is_special or card.rank == Rank.UNKNOWN:
                continue
            if card.rank == Rank.ACE:
                aces += 1
            elif card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
                total_hard += 10
            else:
                try:
                    total_hard += int(card.rank.value)
                except ValueError:
                    pass
        return aces > 0 and total_hard + 11 + (aces - 1) <= 21

    def display(self) -> str:
        if not self.cards:
            return "--"
        ranks = self.ranks
        if not ranks:
            return "--"
        return " ".join(ranks) + f" = {self.value}"
