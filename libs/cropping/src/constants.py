# ====== Code Summary ======
# Defines constants related to image cropping configuration and file handling,
# including supported image formats and default configuration file paths.

# ====== Standard Library Imports ======
from pathlib import Path


class CROPPING_CONSTANTS:
    """
    Constants used for cropping operations, configuration management, and supported image formats.
    """

    # Supported image extensions (must be lowercase and include dot prefix)
    SUPPORTED_EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }

    # Default filename for storing crop configuration JSON
    DEFAULT_CONFIG_FILENAME: str = "crop_config.json"

    # Default path where configuration files are saved or loaded from
    DEFAULT_CONFIG_PATH: Path = Path("../libs/cropping/configs")
