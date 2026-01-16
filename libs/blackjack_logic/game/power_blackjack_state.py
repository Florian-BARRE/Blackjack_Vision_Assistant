"""Blackjack game state manager.

This module maintains the game state for a blackjack table based on per-frame detections:
- Tracks table zones (dealer/player/holder) via ZoneDetector.
- Classifies detected cards into dealer/player hands.
- Pairs dealer scanned/real cards into consolidated CardPair objects.
- Detects special events (dealer switch, holder changing, switch warning).
"""

# ====== Standard Library Imports ======
from __future__ import annotations

# ====== Third-Party Imports ======
from loggerplusplus import LoggerClass

# ====== Internal Project Imports ======
from public_models.rank import Rank

from ..detection import CardType, DetectedCard, DetectionParser, FrameDetections
from ..zones import TableLayout, Zone, ZoneDetector, ZoneState
from .states import SpecialEvent, GamePhase
from .card_pair import CardPair


class PowerBlackjackState(LoggerClass):
    """Main blackjack game state manager.

    This class updates its state on every frame:
    - Parses raw inference outputs into detections.
    - Updates zone layout via ZoneDetector.
    - Detects special events and card-holder transitions.
    - When initialized and no special event blocks it, classifies/pairs cards and updates phase.
    """

    _zone_detector: ZoneDetector
    _layout: TableLayout | None
    _phase: GamePhase

    _dealer_pairs: list[CardPair]
    _dealer_scanned: list[DetectedCard]
    _dealer_real: list[DetectedCard]
    _player_cards: list[DetectedCard]

    _cards_count: int
    _holders_count: int
    _traps_count: int

    _special_event: SpecialEvent
    _has_dealer_card: bool
    _has_switch_card: bool

    _holder_zone: Zone | None
    _holder_in_zone: bool

    def __init__(self) -> None:
        """Initialize the blackjack state manager."""
        super().__init__()

        self.logger.info("START initializing BlackjackState")
        try:
            self._zone_detector = ZoneDetector()
            self._layout = None
            self._phase = GamePhase.WAITING

            self._dealer_pairs = []
            self._dealer_scanned = []
            self._dealer_real = []

            self._player_cards = []

            self._cards_count = 0
            self._holders_count = 0
            self._traps_count = 0

            self._special_event = SpecialEvent.NONE
            self._has_dealer_card = False
            self._has_switch_card = False

            self._holder_zone = None
            self._holder_in_zone = True
        finally:
            self.logger.info("END initializing BlackjackState")

    @property
    def is_initialized(self) -> bool:
        """Whether layout is ready and stable enough to classify cards.

        Returns:
            True if layout exists and is initialized or locked; otherwise False.
        """
        return self._layout is not None and self._layout.state in (
            ZoneState.INITIALIZED,
            ZoneState.LOCKED,
        )

    @property
    def dealer_zone(self) -> Zone | None:
        """Dealer zone if available.

        Returns:
            Dealer Zone or None if not available.
        """
        return self._layout.dealer_zone if self._layout is not None else None

    @property
    def player_zone(self) -> Zone | None:
        """Player zone if available.

        Returns:
            Player Zone or None if not available.
        """
        return self._layout.player_zone if self._layout is not None else None

    @property
    def holder_zone(self) -> Zone | None:
        """Currently tracked holder zone (when locked).

        Returns:
            Holder Zone or None if not tracked.
        """
        return self._holder_zone

    @property
    def phase(self) -> GamePhase:
        """Current game phase.

        Returns:
            GamePhase.
        """
        return self._phase

    @property
    def cards_count(self) -> int:
        """Detected cards count in last update.

        Returns:
            Number of detected cards.
        """
        return self._cards_count

    @property
    def holders_count(self) -> int:
        """Detected card holders count in last update.

        Returns:
            Number of detected holders.
        """
        return self._holders_count

    @property
    def traps_count(self) -> int:
        """Detected traps count in last update.

        Returns:
            Number of detected traps.
        """
        return self._traps_count

    @property
    def is_locked(self) -> bool:
        """Whether ZoneDetector is locked.

        Returns:
            True if locked; otherwise False.
        """
        return self._zone_detector.is_locked

    @property
    def zone_detector(self) -> ZoneDetector:
        """Zone detector instance.

        Returns:
            ZoneDetector.
        """
        return self._zone_detector

    @property
    def dealer_pairs(self) -> list[CardPair]:
        """Consolidated dealer card pairs.

        Returns:
            List of CardPair.
        """
        return self._dealer_pairs

    @property
    def dealer_scanned(self) -> list[DetectedCard]:
        """Dealer scanned cards slice (left half, ordered by x).

        Returns:
            List of DetectedCard.
        """
        return self._dealer_scanned

    @property
    def dealer_real(self) -> list[DetectedCard]:
        """Dealer real cards slice (right half, ordered by x).

        Returns:
            List of DetectedCard.
        """
        return self._dealer_real

    @property
    def player_cards(self) -> list[DetectedCard]:
        """Player cards ordered by x.

        Returns:
            List of DetectedCard.
        """
        return self._player_cards

    @property
    def dealer_card_count(self) -> int:
        """Number of consolidated dealer cards.

        Returns:
            Dealer card count.
        """
        return len(self._dealer_pairs)

    @property
    def player_card_count(self) -> int:
        """Number of player cards.

        Returns:
            Player card count.
        """
        return len(self._player_cards)

    @property
    def special_event(self) -> SpecialEvent:
        """Current special event.

        Returns:
            SpecialEvent.
        """
        return self._special_event

    @property
    def has_dealer_card(self) -> bool:
        """Whether a dealer-switch card was detected.

        Returns:
            True if a dealer card is present; otherwise False.
        """
        return self._has_dealer_card

    @property
    def has_switch_card(self) -> bool:
        """Whether a switch-warning card was detected.

        Returns:
            True if a switch card is present; otherwise False.
        """
        return self._has_switch_card

    @property
    def holder_in_zone(self) -> bool:
        """Whether the tracked holder is still in its registered zone.

        Returns:
            True if in zone (or not tracked); otherwise False.
        """
        return self._holder_in_zone

    def lock_zones(self) -> None:
        """Lock zones and save holder zone for tracking."""
        self.logger.info("START locking zones")
        try:
            self._zone_detector.lock()
            if self._layout is not None and self._layout.holder_zone is not None:
                self._holder_zone = self._layout.holder_zone
                self.logger.debug("Holder zone saved during lock")
            else:
                self._holder_zone = None
                self.logger.debug("No holder zone available to save during lock")
        finally:
            self.logger.info("END locking zones")

    def unlock_zones(self) -> None:
        """Unlock zones and reset holder tracking."""
        self.logger.info("START unlocking zones")
        try:
            self._zone_detector.unlock()
            self._holder_zone = None
            self._holder_in_zone = True
        finally:
            self.logger.info("END unlocking zones")

    def update(self, inferences: list, frame_shape: tuple[int, int]) -> FrameDetections:
        """Update game state with new detections.

        Args:
            inferences: Raw model outputs for the current frame.
            frame_shape: Frame (height, width).

        Returns:
            Parsed FrameDetections.

        Raises:
            Exception: Re-raises any parsing/detection errors after logging once.
        """
        self.logger.info(
            f"START update frame (frame_shape={frame_shape}, inferences_count={len(inferences)})"
        )

        try:
            detections = DetectionParser.parse(inferences)

            self._cards_count = len(detections.cards)
            self._holders_count = len(detections.card_holders)
            self._traps_count = len(detections.traps)

            self.logger.debug(
                f"Parsed detections (cards={self._cards_count}, holders={self._holders_count}, "
                f"traps={self._traps_count})"
            )

            self._layout = self._zone_detector.detect(detections, frame_shape)
            self.logger.debug(
                f"Zone detection completed (initialized={self.is_initialized}, locked={self.is_locked})"
            )

            self._has_dealer_card = len(detections.dealer_cards) > 0
            self._has_switch_card = len(detections.switch_cards) > 0

            self._check_holder_position(detections)
            self._update_special_event()

            self.logger.debug(
                f"Special event evaluated (event={self._special_event.name}, "
                f"has_dealer_card={self._has_dealer_card}, has_switch_card={self._has_switch_card}, "
                f"holder_in_zone={self._holder_in_zone})"
            )

            if self.is_initialized and self._special_event == SpecialEvent.NONE:
                self._classify_and_pair_cards(detections)
                self._update_phase()

                self.logger.info(
                    f"Game state updated (phase={self._phase.name}, dealer_cards={self.dealer_card_count}, "
                    f"player_cards={self.player_card_count})"
                )

            return detections

        except Exception as exc:
            self.logger.error(
                f"ERROR during update frame (exc_type={type(exc).__name__}, message={exc})"
            )
            raise

        finally:
            self.logger.info("END update frame")

    def _check_holder_position(self, detections: FrameDetections) -> None:
        """Check if card holder is still in its registered zone.

        Args:
            detections: Parsed detections for the current frame.
        """
        if self._holder_zone is None or not detections.card_holders:
            self._holder_in_zone = True
            self.logger.debug(
                "Holder position check skipped (no tracked holder zone or no holders detected)"
            )
            return

        holder = max(detections.card_holders, key=lambda h: h.x)
        self._holder_in_zone = self._holder_zone.contains_point(holder.x, holder.y)
        self.logger.debug(
            f"Holder position checked (holder_x={holder.x}, holder_y={holder.y}, in_zone={self._holder_in_zone})"
        )

    def _update_special_event(self) -> None:
        """Update the current special event based on priority."""
        previous_event = self._special_event

        if self._has_dealer_card:
            self._special_event = SpecialEvent.DEALER_SWITCH
        elif not self._holder_in_zone and self._holder_zone is not None:
            self._special_event = SpecialEvent.HOLDER_CHANGING
        elif self._has_switch_card:
            self._special_event = SpecialEvent.SWITCH_WARNING
        else:
            self._special_event = SpecialEvent.NONE

        if self._special_event != previous_event:
            self.logger.info(
                f"Special event transition (from={previous_event.name}, to={self._special_event.name})"
            )

    def _classify_and_pair_cards(self, detections: FrameDetections) -> None:
        """Classify cards and create dealer pairs.

        Args:
            detections: Parsed detections for the current frame.
        """
        dealer_zone = self.dealer_zone
        player_zone = self.player_zone

        if dealer_zone is None:
            self.logger.warning("Dealer zone unavailable; skipping card classification")
            return

        dealer_cards_all: list[DetectedCard] = []
        self._player_cards = []

        for card in detections.cards:
            if card.is_special:
                continue

            if dealer_zone.contains_card(card):
                dealer_cards_all.append(card)
            elif player_zone is not None and player_zone.contains_card(card):
                self._player_cards.append(card)
                card.card_type = CardType.PLAYER

        dealer_cards_all.sort(key=lambda c: c.x)

        n_cards = len(dealer_cards_all)
        if n_cards == 0:
            self._dealer_scanned = []
            self._dealer_real = []
            self._dealer_pairs = []
        elif n_cards == 1:
            self._dealer_scanned = [dealer_cards_all[0]]
            self._dealer_real = []
            self._dealer_pairs = [CardPair(scanned=dealer_cards_all[0], real=None)]
        else:
            mid = n_cards // 2
            self._dealer_scanned = dealer_cards_all[:mid]
            self._dealer_real = dealer_cards_all[mid:]

            self._dealer_pairs = []
            max_len = max(len(self._dealer_scanned), len(self._dealer_real))
            for i in range(max_len):
                scanned = self._dealer_scanned[i] if i < len(self._dealer_scanned) else None
                real = self._dealer_real[i] if i < len(self._dealer_real) else None
                self._dealer_pairs.append(CardPair(scanned=scanned, real=real))

        self._player_cards.sort(key=lambda c: c.x)

        self.logger.debug(
            f"Card classification completed (dealer_total={n_cards}, dealer_pairs={len(self._dealer_pairs)}, "
            f"player_cards={len(self._player_cards)})"
        )

    def _update_phase(self) -> None:
        """Update game phase based on card counts."""
        previous_phase = self._phase

        dealer_count = len(self._dealer_pairs)
        player_count = len(self._player_cards)

        if dealer_count == 0 and player_count == 0:
            self._phase = GamePhase.WAITING
        elif dealer_count < 2 or player_count < 2:
            self._phase = GamePhase.DEALING
        else:
            self._phase = GamePhase.PLAYER_TURN

        if self._phase != previous_phase:
            self.logger.info(
                f"Phase transition (from={previous_phase.name}, to={self._phase.name}, "
                f"dealer_cards={dealer_count}, player_cards={player_count})"
            )
        else:
            self.logger.debug(f"Phase unchanged (phase={self._phase.name})")

    def get_dealer_value(self) -> int:
        """Calculate dealer hand value from consolidated dealer pairs.

        Returns:
            Dealer hand value as blackjack total.
        """
        return _calculate_blackjack_total([pair.consolidated_rank for pair in self._dealer_pairs])

    def get_player_value(self) -> int:
        """Calculate player hand value.

        Returns:
            Player hand value as blackjack total.
        """
        return _calculate_blackjack_total([card.rank for card in self._player_cards])

    def get_recommendation(self) -> tuple[str, str]:
        """Get a simple strategy recommendation with a UI color.

        Returns:
            Tuple of (message, hex_color).
        """
        if not self.is_initialized:
            return "Waiting...", "#6c7086"

        if self._phase == GamePhase.WAITING:
            return "Waiting", "#6c7086"

        if self._phase == GamePhase.DEALING:
            return "Dealing...", "#f9e2af"

        player_val = self.get_player_value()

        if player_val == 21 and len(self._player_cards) == 2:
            return "BLACKJACK!", "#a6e3a1"

        if player_val > 21:
            return "BUSTED", "#f38ba8"

        if player_val < 12:
            return "HIT", "#a6e3a1"
        if player_val < 17:
            return "HIT?", "#f9e2af"
        return "STAND", "#89b4fa"

    def get_status(self) -> str:
        """Get current status message.

        Returns:
            Zone detector status when not initialized; otherwise current phase name.
        """
        if not self.is_initialized:
            return self._zone_detector.get_status()
        return self._phase.name

    def get_dealer_display(self) -> str:
        """Get dealer cards display string.

        Returns:
            A space-separated string of dealer card ranks (consolidated), or "--" if empty.
        """
        if not self._dealer_pairs:
            return "--"
        return " ".join(pair.display_rank for pair in self._dealer_pairs)

    def get_player_display(self) -> str:
        """Get player cards display string.

        Returns:
            A space-separated string of player ranks, or "--" if empty.
        """
        if not self._player_cards:
            return "--"
        return " ".join(
            card.rank.value if card.rank != Rank.UNKNOWN else "?" for card in self._player_cards
        )

    def get_special_event_message(self) -> tuple[str, str, str]:
        """Get special event display info.

        Returns:
            Tuple of (title, message, hex_color).
        """
        if self._special_event == SpecialEvent.DEALER_SWITCH:
            return "🔄 DEALER SWITCH", "Dealer is switching cards...", "#f9e2af"
        if self._special_event == SpecialEvent.HOLDER_CHANGING:
            return "📦 HOLDER CHANGE", "Card holder is being changed!", "#f38ba8"
        if self._special_event == SpecialEvent.SWITCH_WARNING:
            return "⚠️ SWITCH SOON", "Holder will change at end of round", "#fab387"
        return "", "", "#6c7086"


def _calculate_blackjack_total(ranks: list[Rank]) -> int:
    """Compute blackjack total from a list of ranks, handling Aces as 11 or 1.

    Args:
        ranks: List of Rank values.

    Returns:
        Best blackjack total (<=21 when possible).
    """
    total = 0
    aces = 0

    for rank in ranks:
        if rank == Rank.UNKNOWN:
            continue

        if rank == Rank.ACE:
            aces += 1
            total += 11
        elif rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
            total += 10
        else:
            try:
                total += int(rank.value)
            except ValueError:
                continue

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total
