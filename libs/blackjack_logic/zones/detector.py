# ====== Code Summary ======
# Implements a spatial ZoneDetector for identifying blackjack table layout zones
# (dealer, player, and card holder) from a stream of FrameDetections. Supports
# stabilizing detections across frames, state tracking, and optional locking for
# fixed-zone overlays.

# ====== Standard Library Imports ======
from __future__ import annotations

# ====== Third-Party Library Imports ======
from loggerplusplus import LoggerClass

# ====== Internal Project Imports ======
from ..detection import FrameDetections
from .states import ZoneState
from .table_layout import TableLayout
from .zone import Zone


class ZoneDetector(LoggerClass):
    """
    Detect table zones based on detected object positions.

    Pattern:
        - DealerCard (left) + middle cards + CardHolder (right) = dealer row
        - Cards below Trap = player zone

    The detector can stabilize over multiple frames before declaring the layout initialized.
    It can also be locked to freeze a layout.
    """

    _tolerance: int
    _stability_frames: int
    _stable_count: int
    _last_layout: TableLayout | None
    _locked_layout: TableLayout | None

    dealer_aligned: bool
    player_aligned: bool
    holder_found: bool
    trap_found: bool

    def __init__(self, tolerance: int = 100, stability_frames: int = 5) -> None:
        """
        Initialize the zone detector.

        Args:
            tolerance (int): Pixel tolerance for alignment and stability checks.
            stability_frames (int): Number of consecutive stable frames needed to initialize zones.
        """
        super().__init__()
        self.logger.info(
            f"START initializing ZoneDetector (tolerance={tolerance}, stability_frames={stability_frames})"
        )
        try:
            self._tolerance = tolerance
            self._stability_frames = stability_frames
            self._stable_count = 0
            self._last_layout = None
            self._locked_layout = None

            self.dealer_aligned = False
            self.player_aligned = False
            self.holder_found = False
            self.trap_found = False
        finally:
            self.logger.info("END initializing ZoneDetector")

    @property
    def is_locked(self) -> bool:
        """
        Whether a layout is currently locked.

        Returns:
            bool: True if locked; otherwise False.
        """
        return self._locked_layout is not None

    def lock(self) -> None:
        """
        Lock current zones if the latest layout is initialized.
        """
        self.logger.info("START locking zones")
        try:
            if self._last_layout is None:
                self.logger.warning("Lock skipped (no last layout available)")
                return

            if self._last_layout.state != ZoneState.INITIALIZED:
                self.logger.warning(
                    f"Lock skipped (last_layout_state={self._last_layout.state.name})"
                )
                return

            self._locked_layout = self._last_layout
            self._locked_layout.state = ZoneState.LOCKED
            self.logger.info("Zones locked (state=LOCKED)")
        finally:
            self.logger.info("END locking zones")

    def unlock(self) -> None:
        """
        Unlock zones and reset stability counters.
        """
        self.logger.info("START unlocking zones")
        try:
            self._locked_layout = None
            self._stable_count = 0
            self.logger.info("Zones unlocked (stability reset)")
        finally:
            self.logger.info("END unlocking zones")

    def detect(self, detections: FrameDetections, frame_shape: tuple[int, int]) -> TableLayout:
        """
        Attempt to detect table zones.

        Args:
            detections (FrameDetections): Current frame detections.
            frame_shape (tuple[int, int]): Frame (height, width).

        Returns:
            TableLayout: A layout object with computed zones and state.

        Raises:
            ValueError: If frame_shape is invalid.
        """
        if len(frame_shape) != 2:
            raise ValueError(f"frame_shape must be (height, width), got={frame_shape}")

        self.logger.debug(
            f"START detect zones (locked={self.is_locked}, frame_shape={frame_shape})"
        )

        if self._locked_layout is not None:
            self.logger.debug("END detect zones (returning locked layout)")
            return self._locked_layout

        frame_h, frame_w = frame_shape
        layout = TableLayout()

        # 1. Reset state flags
        self.holder_found = len(detections.card_holders) > 0
        self.trap_found = len(detections.traps) > 0
        self.dealer_aligned = False
        self.player_aligned = False

        if not self.holder_found or not self.trap_found or len(detections.cards) < 2:
            layout.state = ZoneState.DETECTING
            self._stable_count = 0
            self.logger.debug(
                "END detect zones (insufficient detections: "
                f"holder_found={self.holder_found}, trap_found={self.trap_found}, "
                f"cards={len(detections.cards)})"
            )
            return layout

        # 2. Analyze layout structure
        holder = detections.card_holders[0]
        trap = detections.traps[0]
        cards_by_x = detections.get_cards_sorted_by_x()
        cards_by_y = detections.get_cards_sorted_by_y()

        # 3. Check dealer alignment
        cards_left_of_holder = [c for c in cards_by_x if c.x < holder.x - 50]
        if cards_left_of_holder:
            holder_y = holder.y
            cards_near_holder_y = [
                c for c in cards_left_of_holder if abs(c.y - holder_y) < self._tolerance
            ]
            self.dealer_aligned = len(cards_near_holder_y) >= 2

        # 4. Check player alignment
        player_cards = [c for c in cards_by_y if c.y > trap.y + 20]
        self.player_aligned = len(player_cards) >= 1

        self.logger.debug(
            f"Alignment checks completed (dealer_aligned={self.dealer_aligned}, "
            f"player_aligned={self.player_aligned}, holder_found={self.holder_found}, "
            f"trap_found={self.trap_found})"
        )

        # 5. Build layout if both alignments succeed
        if self.dealer_aligned and self.player_aligned:
            dealer_cards = [
                c for c in cards_by_x if c.x < holder.x - 50 and abs(c.y - holder.y) < self._tolerance
            ]

            if dealer_cards and player_cards:
                dealer_left = min(c.x for c in dealer_cards) - 30
                rightmost_card_x = max(c.x for c in dealer_cards)
                dealer_right = (rightmost_card_x + holder.x) / 2.0
                leftmost_bottom = min(dealer_cards, key=lambda c: c.x).y + 60
                dealer_top = min(c.y for c in dealer_cards) - 30

                safe_left = max(0.0, float(dealer_left))
                safe_top = max(0.0, float(dealer_top))

                layout.dealer_zone = Zone(
                    x=int(safe_left),
                    y=int(safe_top),
                    width=int(dealer_right - safe_left),
                    height=int(leftmost_bottom - safe_top),
                )

                trap_bottom = trap.y + 40
                player_bottom = max(c.y for c in player_cards) + 60

                layout.player_zone = Zone(
                    x=int(safe_left),
                    y=int(trap_bottom),
                    width=int(dealer_right - safe_left),
                    height=int(min(frame_h, player_bottom) - trap_bottom),
                )

                holder_margin = 80
                layout.holder_zone = Zone(
                    x=int(holder.x - holder_margin),
                    y=int(holder.y - holder_margin),
                    width=int(holder_margin * 2),
                    height=int(holder_margin * 2),
                )

            if layout.dealer_zone and layout.player_zone:
                if self._is_stable(layout):
                    self._stable_count += 1
                else:
                    self._stable_count = 1

                self._last_layout = layout

                layout.state = (
                    ZoneState.INITIALIZED
                    if self._stable_count >= self._stability_frames
                    else ZoneState.DETECTING
                )
            else:
                layout.state = ZoneState.DETECTING
                self._stable_count = 0
        else:
            layout.state = ZoneState.DETECTING
            self._stable_count = 0

        self.logger.debug(
            f"END detect zones (state={layout.state.name}, stable_count={self._stable_count}, "
            f"stability_frames={self._stability_frames})"
        )
        return layout

    def _is_stable(self, layout: TableLayout) -> bool:
        """
        Check whether the layout is stable compared to the last layout.

        Args:
            layout (TableLayout): The newly computed layout.

        Returns:
            bool: True if layout is stable; otherwise False.
        """
        if self._last_layout is None or self._last_layout.dealer_zone is None or layout.dealer_zone is None:
            return False

        d1 = self._last_layout.dealer_zone
        d2 = layout.dealer_zone
        drift = abs(d1.x - d2.x) + abs(d1.y - d2.y)
        return drift < self._tolerance

    def get_status(self) -> str:
        """
        Return a user-friendly status string describing detection state.

        Returns:
            str: Status message for UI or logging.
        """
        if self._locked_layout is not None:
            return "Zones locked"

        parts: list[str] = []
        if not self.holder_found:
            parts.append("No holder")
        if not self.trap_found:
            parts.append("No trap")
        if not self.dealer_aligned:
            parts.append("Dealer not aligned")
        if not self.player_aligned:
            parts.append("Player not aligned")

        if not parts:
            return f"Stabilizing... ({self._stable_count}/{self._stability_frames})"
        return ", ".join(parts)
