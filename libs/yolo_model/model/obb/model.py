# ====== Code Summary ======
# Implementation of a YOLO-based model for detecting Oriented Bounding Boxes (OBB).
# Handles inference execution and structured conversion to `ObbInference` instances.

# ====== Standard Library Imports ======
from pathlib import Path

# ====== Third-Party Library Imports ======
import numpy as np
import torch

# ====== Internal Project Imports ======
from ..base import BaseModel
from ..structs import ModelType
from public_models.obb_type import ObbType
from .inference import ObbInference


class ObbModel(BaseModel):
    """
    YOLO-based model class for Oriented Bounding Box (OBB) detection.

    Performs model inference and processes results into normalized, typed outputs.
    """

    type: ModelType = ModelType.OBB

    def __init__(
        self,
        path: str | Path,
        imgsz: int | tuple[int, int],
        device: str | int | torch.device,
        verbose: bool,
        conf: float,
        iou: float,
        max_det: int,
    ) -> None:
        """
        Initialize the OBB model with configuration parameters.

        Args:
            path (str | Path): Path to model weights.
            imgsz (int | tuple[int, int]): Image size.
            device (str | int | torch.device): Device to run inference on.
            verbose (bool): Verbosity flag.
            conf (float): Confidence threshold.
            iou (float): IoU threshold.
            max_det (int): Maximum detections per image.
        """
        self._conf: float = conf
        self._iou: float = iou
        self._max_det: int = max_det

        super().__init__(
            path=path,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
        )

    def __get_obb_type_from_class_id(self, cid: str | int) -> ObbType:
        """
        Map class ID to ObbType enum.

        Args:
            cid (str | int): Class identifier.

        Returns:
            ObbType: Corresponding object type.
        """
        cname = self._get_class_name_from_id(cid)
        if cname is None:
            return ObbType.UNKNOWN
        return ObbType(cname)

    def infer(
        self,
        source: str | Path | np.ndarray | list[np.ndarray]
    ) -> list[ObbInference]:
        """
        Run inference and return structured OBB results.

        Args:
            source (str | Path | np.ndarray | list[np.ndarray]):
                Input image or batch of images.

        Returns:
            list[ObbInference]: Structured predictions with boxes and metadata.
        """
        # 1. Perform model prediction
        results = self._model.predict(
            source=source,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=self._verbose,
            max_det=self._max_det,
        )[0].obb

        # 2. Handle no detection
        if results is None or len(results) == 0:
            return []

        # 3. Extract result tensors
        polys = results.xyxyxyxy.detach().cpu().numpy()
        polygons = polys.reshape(-1, 4, 2)
        cls_ids = results.cls.detach().cpu().numpy().astype(int)
        confs = results.conf.detach().cpu().numpy()

        obb_inferences: list[ObbInference] = []

        # 4. Process each polygon box
        for i in range(len(polygons)):
            box = polygons[i]

            # 4.1 Sort points by y-coordinate to identify top/bottom
            y_sorted = box[np.argsort(box[:, 1])]
            top_two = y_sorted[:2]
            bottom_two = y_sorted[2:]

            # 4.2 Sort by x within each group to get corners
            tl, tr = top_two[np.argsort(top_two[:, 0])]
            bl, br = bottom_two[np.argsort(bottom_two[:, 0])]

            # 4.3 Reorder consistently as [tl, tr, br, bl]
            sorted_box = np.array([tl, tr, br, bl], dtype=np.float32)

            # 4.4 Construct inference object
            obb_inferences.append(
                ObbInference(
                    box=sorted_box,
                    obb_type=self.__get_obb_type_from_class_id(cls_ids[i]),
                    confidence=confs[i],
                )
            )

        return obb_inferences
