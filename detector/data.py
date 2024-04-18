import torch

import lightning.pytorch as pl

from ultralytics.models.yolo.detect.train import DetectionTrainer
from omegaconf.dictconfig import DictConfig
from easydict import EasyDict

class TrafficSigns(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.detection_trainer = DetectionTrainer(overrides=EasyDict(cfg.yolo_args))
        self.detection_trainer.model = None

        self.train_set = self.cfg.data.train
        self.val_set = self.cfg.data.val

    def _get_dataloader(self, path, is_train):
        ds = self.detection_trainer.build_dataset(path, batch=self.cfg.train.batch_size)

        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=is_train,
            num_workers=self.cfg.train.dataloader_num_wokers,
            pin_memory=True,
            collate_fn=ds.collate_fn
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_set, is_train=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_set, is_train=False)