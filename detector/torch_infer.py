import torch
from omegaconf.dictconfig import DictConfig

from .model import Yolov8


class TorchInfer:
    def __init__(self, cfg: DictConfig, checkpoint_path: str):
        self.cfg = cfg

        self.model = Yolov8.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.model.to(cfg.train.accelerator)
        self.model.eval()

    def do_torch_inference(self, input_dict: dict):
        with torch.no_grad():
            return self.model.forward(input_dict)
