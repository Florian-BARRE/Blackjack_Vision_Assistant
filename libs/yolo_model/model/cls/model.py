# ====== Code Summary ======
# Model class for Rank classification tasks (e.g., jersey numbers, ranks).
# Wraps a YOLO-based classifier and returns structured RankInference outputs.

# ====== Standard Library Imports ======
from pathlib import Path

# ====== Third-Party Library Imports ======
import numpy as np
import torch

# ====== Internal Project Imports ======
from ..base import BaseModel
from ..structs import ModelType
from .inference import RankInference
from public_models.rank import Rank


class RankModel(BaseModel):
    """
    Classification model for predicting ranks.

    Converts raw model predictions into structured RankInference objects
    including top-1 and top-5 ranked predictions with confidences.
    """

    type: ModelType = ModelType.CLS

    def __init__(
        self,
        path: str | Path,
        imgsz: int | tuple[int, int] | None,
        device: str | int | torch.device,
        verbose: bool,
        conf: float,
    ) -> None:
        """
        Initialize the Rank classification model.

        Args:
            path (str | Path): Path to model weights.
            imgsz (int | tuple | None): Input image size.
            device (str | int | torch.device): Device to run inference.
            verbose (bool): Verbosity flag.
            conf (float): Minimum confidence threshold to return a result.
        """
        self._conf: float = conf

        super().__init__(
            path=path,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
        )

    def __get_rank_type_from_class_id(self, cid: str | int) -> Rank:
        """
        Map a class ID to a Rank enum using the class name.

        Args:
            cid (str | int): Class index or label.

        Returns:
            Rank: Enum corresponding to the class name.
        """
        return Rank.from_str(
            value=self._get_class_name_from_id(cid)
        )

    def infer(
        self,
        source: str | Path | np.ndarray | list[np.ndarray],
    ) -> RankInference | None:
        """
        Perform inference and return a RankInference result.

        Args:
            source (str | Path | np.ndarray | list[np.ndarray]):
                Image input(s) for inference.

        Returns:
            RankInference | None: Inference result, or None if confidence is too low.
        """
        # 1. Optional argument forwarding
        optional_args = {"imgsz": self._imgsz} if self._imgsz is not None else {}

        # 2. Run prediction and extract probability outputs
        result = self._model.predict(
            source=source,
            device=self._device,
            verbose=self._verbose,
            **optional_args
        )[0].probs

        # 3. Construct inference object
        rank_inference = RankInference(
            confidence=float(result.top1conf),
            rank=self.__get_rank_type_from_class_id(result.top1),
            top5_confidence=[float(x) for x in result.top5conf],
            top5_rank=[self.__get_rank_type_from_class_id(x) for x in result.top5],
        )

        # 4. Filter by confidence threshold
        if rank_inference.confidence < self._conf:
            return None

        return rank_inference
