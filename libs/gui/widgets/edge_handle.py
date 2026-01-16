# ====== Code Summary ======
# Defines a custom `tk.Canvas` widget used as an interactive arrow-based handle
# for resizing the screen capture region in a GUI. The handle supports mouse interaction
# (left-click to resize, right-click to toggle direction) and auto-repeat while held.

"""Edge handle widget for resizing capture region."""

# ====== Standard Library Imports ======
import tkinter as tk
from typing import Callable


class EdgeHandle(tk.Canvas):
    """
    Interactive arrow-shaped handle used to expand/shrink a capture region from a specific edge.

    Supports mouse events:
    - Left click: Resize by step amount (with auto-repeat if held).
    - Right click: Toggle expansion direction.
    """

    COLORS = {
        'normal': '#6c7086',   # Default color
        'hover': '#89b4fa',    # On mouse hover
        'active': '#f38ba8',   # While clicking
    }

    def __init__(
        self,
        parent: tk.Misc,
        edge: str,
        on_adjust: Callable[[str, int], None],
        step: int = 25,
    ) -> None:
        """
        Initialize the edge handle.

        Args:
            parent (tk.Misc): Parent tkinter widget.
            edge (str): One of 'left', 'right', 'top', 'bottom'.
            on_adjust (Callable[[str, int], None]): Callback to adjust region size.
            step (int): Amount to adjust per click/hold.
        """
        width, height = (16, 40) if edge in ("left", "right") else (40, 16)
        super().__init__(
            parent,
            width=width,
            height=height,
            bg="#1e1e2e",
            highlightthickness=0,
            cursor="hand2",
        )

        self._edge: str = edge
        self._on_adjust: Callable[[str, int], None] = on_adjust
        self._step: int = step
        self._expanding: bool = True
        self._holding: bool = False
        self._hold_job: int | None = None

        self._draw_arrow()
        self._bind_events()

    def _bind_events(self) -> None:
        """Bind mouse events to the handle."""
        self.bind("<Enter>", lambda _: self._draw_arrow(self.COLORS["hover"]))
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._toggle_direction)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _get_arrow_points(self) -> list[int]:
        """
        Calculate polygon points for the arrow shape.

        Returns:
            list[int]: Flattened list of x, y coordinates.
        """
        w, h = int(self["width"]), int(self["height"])

        # Two states per edge: expanding vs contracting
        arrows = {
            "left": (
                [w - 2, 4, 4, h // 2, w - 2, h - 4],
                [4, 4, w - 4, h // 2, 4, h - 4],
            ),
            "right": (
                [2, 4, w - 4, h // 2, 2, h - 4],
                [w - 4, 4, 4, h // 2, w - 4, h - 4],
            ),
            "top": (
                [4, h - 2, w // 2, 4, w - 4, h - 2],
                [4, 4, w // 2, h - 4, w - 4, 4],
            ),
            "bottom": (
                [4, 2, w // 2, h - 4, w - 4, 2],
                [4, h - 4, w // 2, 4, w - 4, h - 4],
            ),
        }

        return arrows[self._edge][0 if self._expanding else 1]

    def _draw_arrow(self, color: str = None) -> None:
        """
        Draw or redraw the arrow polygon.

        Args:
            color (str, optional): Fill and outline color. Defaults to normal color.
        """
        self.delete("all")
        self.create_polygon(
            self._get_arrow_points(),
            fill=color or self.COLORS["normal"],
            outline=color or self.COLORS["normal"],
        )

    def _toggle_direction(self, event=None) -> None:
        """Right-click handler to toggle arrow direction."""
        self._expanding = not self._expanding
        self._draw_arrow(self.COLORS["hover"])

    def _on_leave(self, event) -> None:
        """Mouse leave handler; cancel repeat and reset appearance."""
        self._holding = False
        if self._hold_job:
            self.after_cancel(self._hold_job)
            self._hold_job = None
        self._draw_arrow()

    def _on_click(self, event) -> None:
        """Mouse down handler; start adjusting and repeating."""
        self._holding = True
        self._draw_arrow(self.COLORS["active"])
        self._do_adjust()

    def _on_release(self, event) -> None:
        """Mouse release handler; stop adjusting."""
        self._holding = False
        if self._hold_job:
            self.after_cancel(self._hold_job)
            self._hold_job = None
        self._draw_arrow(self.COLORS["hover"])

    def _do_adjust(self) -> None:
        """Perform a single adjustment and schedule the next if still holding."""
        if not self._holding:
            return

        delta = self._step if self._expanding else -self._step
        self._on_adjust(self._edge, delta)

        self._hold_job = self.after(80, self._do_adjust)
