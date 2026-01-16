# ====== Code Summary ======
# Abstract base class for YOLO-based models.
# Handles model loading, device and image size config, logging, and class ID resolution.

# ====== Standard Library Imports ======
from abc import ABC, abstractmethod
from pathlib import Path

# ====== Third-Party Library Imports ======
import torch
import numpy as np
from ultralytics import YOLO
from loggerplusplus import LoggerClass

# ====== Internal Project Imports ======
from ..structs import ModelType


class BaseModel(ABC, LoggerClass):
    """
    Abstract base class for all YOLO-based model wrappers.

    Responsibilities:
    - Loads YOLO model from checkpoint.
    - Manages device, image size, and verbosity.
    - Provides utility for class name lookup.
    - Requires `infer()` to be implemented by subclasses.
    """

    type: ModelType  # Must be defined by concrete subclass

    def __init__(
            self,
            path: str | Path,
            imgsz: int | tuple[int, int] | None,
            device: str | int | torch.device,
            verbose: bool,
    ) -> None:
        """
        Initialize model base with common inference configuration.

        Args:
            path (str | Path): Path to YOLO model checkpoint (.pt file).
            imgsz (int | tuple | None): Input image size.
            device (str | int | torch.device): Torch-compatible device descriptor.
            verbose (bool): Whether to enable verbose output.
        """
        LoggerClass.__init__(self)

        self.path: Path = Path(path) if isinstance(path, str) else path
        self._imgsz: int | tuple[int, int] | None = imgsz
        self._device: str | int | torch.device = device
        self._verbose: bool = verbose

        # Load YOLO model
        self._model: YOLO = self.__load_yolo_model()

    def __load_yolo_model(self) -> YOLO:
        """
        Load a YOLO model instance from file.

        Returns:
            YOLO: Loaded YOLO model.
        """
        try:
            model = YOLO(self.path, task=str(self.type.value))
            self.logger.info(f"[{self.type.value}] YOLO model loaded: {self.path}")
            return model
        except Exception as e:
            self.logger.error("Failed to load YOLO model", exc_info=True)
            raise e

    def _get_class_name_from_id(self, cid: str | int) -> str | None:
        """
        Map class ID to class name using YOLO's internal label map.

        Args:
            cid (str | int): Class ID to resolve.

        Returns:
            str | None: Class name, or None if not found.
        """
        cname: str | None = self._model.model.names.get(int(cid), None)

        if cname is None:
            self.logger.warning(
                f"No {self.type} name found for {cid}. "
                f"Model cname mapping: {self._model.model.names}"
            )

        return cname

    @abstractmethod
    def infer(
            self,
            source: str | Path | np.ndarray | list[np.ndarray],
    ):
        """
        Abstract inference method to be implemented by subclasses.

        Args:
            source (str | Path | np.ndarray | list[np.ndarray]): Image input(s).

        Returns:
            Any: Inference result specific to the model type.
        """
        ...
