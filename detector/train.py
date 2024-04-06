from ultralytics import YOLO


def train(
    dataset_path: str,
    epochs: int = 1,
    imgsz: int = 224,
    model_ckeckpoint: str = "yolov8n.pt",
    **kwargs,
) -> None:
    """Provides functionality to train model."""

    model = YOLO(model_ckeckpoint)

    model.train(data=dataset_path, epochs=epochs, imgsz=imgsz, **kwargs)
