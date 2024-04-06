from typing import Any

from ultralytics import YOLO


def infer(images_to_infer: Any, checkpoint_path: str, **kwargs):
    """Provides functionality to model inference.

    Args:
        images_to_infer: any object described in
            https://docs.ultralytics.com/modes/predict/#inference-sources.
        checkpoint_path: absolute/relative path to checkpoint.
    """
    model = YOLO(checkpoint_path)

    # return a generator of Results objects
    results = model(images_to_infer, stream=True, **kwargs)

    return results
