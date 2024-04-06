import logging

import fire

from detector.infer import infer as infer_model
from detector.train import train as train_model
from detector.validate import validate as validate_model


logger = logging.getLogger(__name__)


def train(dataset_path, epochs):
    train_model(dataset_path, epochs)


def infer(images_to_infer, checkpoint_path):
    infer_model(images_to_infer, checkpoint_path)


def validate(dataset_path, checkpoint_path):
    metrics = validate_model(dataset_path, checkpoint_path)
    logger.info(metrics)


if __name__ == "__main__":
    fire.Fire()
