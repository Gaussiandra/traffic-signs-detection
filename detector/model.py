import lightning.pytorch as pl
from easydict import EasyDict
from omegaconf.dictconfig import DictConfig
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils.downloads import attempt_download_asset


class Yolov8(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.detection_trainer = DetectionTrainer(overrides=EasyDict(cfg.yolo_args))

        # model init
        self.model = DetectionModel(cfg=cfg.model.cfg, nc=cfg.model.nc)
        attempt_download_asset(f"weights/{cfg.model.name}")
        weights, ckpt = attempt_load_one_weight(f"weights/{cfg.model.name}")
        self.model.load(weights)

        # loss settings
        self.model.args = EasyDict()
        self.model.args["box"] = cfg.loss.box
        self.model.args["cls"] = cfg.loss.cls
        self.model.args["dfl"] = cfg.loss.dfl

        self.save_hyperparameters()

    def forward(self, input):
        input = self.detection_trainer.preprocess_batch(input)
        loss, loss_items = self.model(input)

        return loss, loss_items

    def training_step(self, batch, batch_idx, dalaloader_idx=0):
        loss, loss_items = self(batch)

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["img"].shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_items = self(batch)

        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["img"].shape[0],
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.detection_trainer.build_optimizer(
            model=self.model,
            lr=self.cfg.train.learning_rate,
            decay=self.cfg.train.weight_decay,
            iterations=self.cfg.train.num_training_steps,
        )

        return [optimizer]
