# ====== Code Summary ======
# Configuration manager responsible for loading, saving, and dynamically updating
# application-level settings using a persistent JSON file.

# ====== Standard Library Imports ======
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# ====== Third-Party Library Imports ======
from loggerplusplus import LoggerClass

# ====== Local Project Imports ======
from .app import AppConfig


class GuiConfigManager(LoggerClass):
    """
    Manages loading, saving, and updating application configuration.

    This class is responsible for handling persistent application configuration stored
    in a JSON file, including default initialization, serialization, and selective updates.
    """

    DEFAULT_PATH: Path = Path(__file__).resolve().parent / ".blackjack_analyzer" / "config.json"

    _path: Path
    _config: AppConfig

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_path (Optional[Path]): Optional custom path to the configuration file.
        """
        super().__init__()
        self._path = config_path or self.DEFAULT_PATH
        self._config = AppConfig()

        self.logger.debug(
            f"INIT ConfigManager (path='{self._path}')"
        )

    @property
    def config(self) -> AppConfig:
        """
        Return the active application configuration.

        Returns:
            AppConfig: Current configuration instance.
        """
        return self._config

    def load(self) -> AppConfig:
        """
        Load configuration from disk.

        Returns:
            AppConfig: Loaded or default configuration.
        """
        self.logger.info(f"START loading configuration (path='{self._path}')")

        try:
            # 1. Check if config file exists
            if not self._path.exists():
                self.logger.info("END loading configuration (file not found, using defaults)")
                return self._config

            # 2. Read and deserialize JSON from file
            with self._path.open("r", encoding="utf-8") as file:
                raw_data: dict[str, object] = json.load(file)

            # 3. Construct AppConfig from deserialized data
            self._config = AppConfig(**raw_data)

            self.logger.info("END loading configuration (success)")

        except Exception as exc:  # noqa: BLE001
            # 4. Fallback to default config on failure
            self.logger.warning(
                f"END loading configuration (failed, using defaults, error='{type(exc).__name__}')"
            )
            self._config = AppConfig()

        return self._config

    def save(self) -> None:
        """
        Persist configuration to disk.
        """
        self.logger.info(f"START saving configuration (path='{self._path}')")

        try:
            # 1. Ensure config directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # 2. Serialize config to JSON and write to file
            with self._path.open("w", encoding="utf-8") as file:
                json.dump(asdict(self._config), file, indent=2)

            self.logger.info("END saving configuration (success)")

        except Exception as exc:  # noqa: BLE001
            # 3. Raise error after logging failure
            self.logger.error(
                f"END saving configuration (failed, error='{type(exc).__name__}')"
            )
            raise

    def update(self, **kwargs: object) -> None:
        """
        Update configuration values dynamically.

        Only existing configuration attributes will be updated.

        Args:
            **kwargs (object): Configuration field names and their new values.
        """
        self.logger.debug(
            f"START updating configuration (fields={list(kwargs.keys())})"
        )

        # 1. Iterate and apply valid updates
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                self.logger.debug(f"UPDATED config field '{key}'")

        self.logger.debug("END updating configuration")
