import torch
from PIL import Image
from torchvision import transforms

from .model import Yolov8


def infer(target_sz: tuple[int, int], checkpoint_path: str, image_path: str):
    """Provides functionality for model inference."""

    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize(target_sz)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    processed_image = transform(resized_image).unsqueeze(0)

    input_dict = {
        "im_file": ("aboba",),
        "img": processed_image,
        "ori_shape": ((image.size[0], image.size[1]),),
        "batch_idx": torch.tensor(1),
        "cls": torch.tensor(1),
        "bboxes": torch.tensor([[1]]),
    }

    model = Yolov8.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()

    with torch.no_grad():
        results = model.forward(input_dict)

    return results
