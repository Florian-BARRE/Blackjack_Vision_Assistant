# ====== Code Summary ======
# Base class for inference result objects, capturing confidence and timestamp.
# Intended to be extended by task-specific inference dataclasses.

# ====== Standard Library Imports ======
from dataclasses import dataclass, field
import time


@dataclass
class BaseInference:
    """
    Base class for model inference outputs.

    Attributes:
        confidence (float): Confidence score of the prediction.
        time (float): Time when the inference was created (epoch time).
    """
    confidence: float
    time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """
        Automatically set the inference timestamp after initialization.
        """
        self.inference_time = time.time()
