# ====== Code Summary ======
# MetaModel combines OBB detection and classification (rank prediction).
# For each detected card, it optionally runs a second model to infer its rank.

# ====== Standard Library Imports ======
from pathlib import Path

# ====== Third-Party Library Imports ======
from loggerplusplus import LoggerClass
import numpy as np
import cv2

# ====== Internal Project Imports ======
from ..model import (
    ObbModel,
    RankModel,
    ObbInference,
    RankInference
)
from .inference import MetaModelInference
from public_models.obb_type import ObbType


class MetaModel(LoggerClass):
    """
    Composite model that performs OBB detection followed by optional classification.

    Workflow:
    1. Runs OBB model to detect card-like objects.
    2. If detected object is a card, extracts and classifies its rank.
    3. Wraps results in a MetaModelInference object.
    """

    def __init__(
            self,
            obb_model: ObbModel,
            rank_model: RankModel,
    ):
        """
        Initialize the MetaModel with both OBB and Rank models.

        Args:
            obb_model (ObbModel): Model for oriented bounding box detection.
            rank_model (RankModel): Model for rank classification.
        """
        LoggerClass.__init__(self)

        self._obb_model = obb_model
        self._rank_model = rank_model

    def infer(
            self,
            source: str | Path | np.ndarray | list[np.ndarray]
    ) -> list[MetaModelInference]:
        """
        Run inference on an image using both OBB and classification models.

        Args:
            source (Union[str, Path, np.ndarray, List[np.ndarray]]): Input image or path.

        Returns:
            List[MetaModelInference]: Inference results for each detected object.
        """
        meta_model_inferences: list[MetaModelInference] = []

        # 1. Infer OBB model
        obb_inferences = self._obb_model.infer(source=source)

        # 2. Infer with Rank or Suit if obb inference type correspond
        for obb_inference in obb_inferences:
            # 2.a check obb type
            # 2.a.i Card-holder or Trap -> add to output detection no more computation needed
            if obb_inference.obb_type == ObbType.CARD_HOLDER or obb_inference.obb_type == ObbType.TRAP:
                meta_model_inferences.append(
                    MetaModelInference(
                        obb_inference=obb_inference,
                        rank_inference=RankInference.default(),
                    )
                )

            # 2.a.ii Card type -> need to compute rank value
            elif obb_inference.obb_type == ObbType.CARD:
                # 2.a.ii.1 Extract the obb detection image
                cropped_image = self._crop_obb_from_image(image=source, obb_inference=obb_inference)

                # 2.a.ii.2 Compute rank of the card
                rank_inference = self._rank_model.infer(source=cropped_image)
                meta_model_inferences.append(
                    MetaModelInference(
                        obb_inference=obb_inference,
                        rank_inference=rank_inference
                    )
                )

            # 2.a.iii Unknown type -> warning
            else:
                self.logger.warning(f"Unknown obb type: {obb_inference}")

        return meta_model_inferences

    @staticmethod
    def _crop_obb_from_image(image: np.ndarray, obb_inference: ObbInference) -> np.ndarray:
        """
        Extract and warp a cropped image of the detected OBB region.

        Args:
            image (np.ndarray): Original image.
            obb_inference (ObbInference): OBB detection result.

        Returns:
            np.ndarray: Cropped and rectified image patch.
        """
        (tl, tr, br, bl) = obb_inference.box

        # Compute the width and height of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Ensure minimal crop size
        maxWidth = max(maxWidth, 2)
        maxHeight = max(maxHeight, 2)

        # Define destination box for warping
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(obb_inference.box, dst)

        # Warp the image
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped
