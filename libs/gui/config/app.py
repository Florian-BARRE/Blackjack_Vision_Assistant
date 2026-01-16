# ====== Code Summary ======
# Defines a data class `AppConfig` used to store persistent GUI and capture settings
# for the Blackjack Analyzer application. This includes window geometry,
# screen capture region, scaling factor, and UI panel widths.

"""Configuration manager for persistent settings."""

# ====== Standard Library Imports ======
from dataclasses import dataclass


@dataclass
class AppConfig:
    """
    Application configuration container for UI and capture settings.

    Attributes:
        window_x (int): X-coordinate of the application window.
        window_y (int): Y-coordinate of the application window.
        window_width (int): Width of the application window.
        window_height (int): Height of the application window.

        capture_x (int): X-coordinate of the screen capture region.
        capture_y (int): Y-coordinate of the screen capture region.
        capture_width (int): Width of the screen capture region.
        capture_height (int): Height of the screen capture region.

        scale (float): Scaling factor for rendering preview.

        info_panel_width (int): Width of the info panel in the UI.
        strategy_panel_width (int): Width of the strategy panel in the UI.
    """
    # Window position and size
    window_x: int = 100
    window_y: int = 100
    window_width: int = 1400
    window_height: int = 600

    # Capture region
    capture_x: int = 0
    capture_y: int = 0
    capture_width: int = 800
    capture_height: int = 450

    # Display scale
    scale: float = 0.6

    # Panel widths
    info_panel_width: int = 280
    strategy_panel_width: int = 260
