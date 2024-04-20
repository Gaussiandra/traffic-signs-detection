from typing import Any

from .model import Yolov8


def infer(batch_to_infer: Any, checkpoint_path: str):
    """Provides functionality to model inference."""

    model = Yolov8.load_from_checkpoint(checkpoint_path=checkpoint_path)
    results = model(batch_to_infer)

    return results
