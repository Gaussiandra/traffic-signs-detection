import fire
import omegaconf

from detector.infer import infer as infer_model
from detector.train import main as train_model


def train(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    train_model(cfg)


def infer(cfg_path, checkpoint_path, image_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)

    results = infer_model(
        target_sz=(cfg.yolo_args.imgsz, cfg.yolo_args.imgsz),
        checkpoint_path=checkpoint_path,
        image_path=image_path,
    )

    return results


if __name__ == "__main__":
    fire.Fire()
