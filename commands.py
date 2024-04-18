import fire
import omegaconf

from detector.infer import infer as infer_model
from detector.train import main as train_model


def train(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    train_model(cfg)


def infer(batch_to_infer, checkpoint_path):
    infer_model(batch_to_infer, checkpoint_path)


if __name__ == "__main__":
    fire.Fire()
