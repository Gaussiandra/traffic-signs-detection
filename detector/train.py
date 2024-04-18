import hydra

import lightning.pytorch as pl
from omegaconf.dictconfig import DictConfig
from pathlib import Path

from model import Yolov8
from data import TrafficSigns

@hydra.main(config_path="configs", config_name="base_config_64", version_base="1.3")
def main(cfg: DictConfig):
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.artifacts.checkpoint.dirpath) / 
                         cfg.artifacts.experiment_name,
            filename=cfg.artifacts.checkpoint.filename,
            monitor=cfg.artifacts.checkpoint.monitor,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
        )
    ]

    loggers = [
        # pl.loggers.TensorBoardLogger("./tensorboard_logs"),
        # mlflowlogger
    ]

    trainer = pl.Trainer(
        logger=loggers, 
        accelerator=cfg.train.accelerator,
        max_steps=cfg.train.num_training_steps, 
        log_every_n_steps=cfg.train.log_every_n_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        val_check_interval=cfg.train.val_check_interval,
        precision=cfg.train.precision,
        callbacks=callbacks
    )

    model = Yolov8(cfg)
    dm = TrafficSigns(cfg)

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()