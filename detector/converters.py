import onnx
import tensorrt as trt
import torch
from omegaconf.dictconfig import DictConfig

from .model import Yolov8


def convert_to_onnx(cfg: DictConfig, checkpoint_path: str, output_name: str):
    model = Yolov8.load_from_checkpoint(checkpoint_path=checkpoint_path)
    input_sample = torch.randn((1, 3, cfg.yolo_args.imgsz, cfg.yolo_args.imgsz))

    torch.onnx.export(
        model.model,
        input_sample,
        output_name,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
    )

    onnx_model = onnx.load(output_name)
    onnx.checker.check_model(onnx_model)


def build_trt_engine(onnx_path: str, trt_engine_path: str):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    assert parser.parse_from_file(onnx_path)

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)
