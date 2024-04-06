from ultralytics import YOLO


def validate(dataset_path: str, checkpoint_path: str, **kwargs):
    """Validates model using config from ../runs/detect/train/args.yaml."""

    model = YOLO(checkpoint_path)

    metrics = model.val(data=dataset_path, **kwargs)

    return metrics
