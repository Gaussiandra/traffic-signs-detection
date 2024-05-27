from pathlib import Path

import hydra
import lightning.pytorch as pl
from omegaconf.dictconfig import DictConfig

from .data import TrafficSigns
from .model import Yolov8


@hydra.main(config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.artifacts.checkpoint.dirpath)
            / cfg.artifacts.experiment_name,
            filename=cfg.artifacts.checkpoint.filename,
            monitor=cfg.artifacts.checkpoint.monitor,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
        )
    ]

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=cfg.artifacts.tracking_uri,
        )
    ]

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=cfg.train.accelerator,
        max_steps=cfg.train.num_training_steps,
        log_every_n_steps=cfg.train.log_every_n_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        val_check_interval=cfg.train.val_check_interval,
        precision=cfg.train.precision,
        callbacks=callbacks,
    )

    model = Yolov8(cfg)
    dataset = TrafficSigns(cfg)

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    main()
