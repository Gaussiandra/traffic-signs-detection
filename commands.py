import fire
import omegaconf

from detector import benchmark, converters, model, torch_infer, train, trt_infer


def train_model(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    train.train(cfg)


def benchmark_torch(cfg_path, checkpoint_path, image_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)

    infer_obj = torch_infer.TorchInfer(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
    )

    input = model.Yolov8.preprocess_input(
        image_path, cfg.yolo_args.imgsz, cfg.yolo_args.imgsz, cfg.data.mean, cfg.data.std
    )

    benchmark.benchmark_model(infer_obj.do_torch_inference, model_args=(input,))


def benchmark_trt(cfg_path, engine_path, image_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)

    infer_obj = trt_infer.TRTEngineInfer(engine_path)
    input = model.Yolov8.preprocess_input(
        image_path, cfg.yolo_args.imgsz, cfg.yolo_args.imgsz, cfg.data.mean, cfg.data.std
    )["img"]

    infer_obj.load_images_to_buffer(input, infer_obj.host_input)
    benchmark.benchmark_model(
        infer_obj.do_trt_inference,
        model_args=(),
        n_warmups=cfg.inference.benchmark_warmups,
        n_tests=cfg.inference.benchmark_tests,
    )


def convert_to_onnx(cfg_path, checkpoint_path, output_name):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    converters.convert_to_onnx(cfg, checkpoint_path, output_name)


def convert_to_trt(onnx_path: str, trt_engine_path: str):
    converters.build_trt_engine(onnx_path, trt_engine_path)


if __name__ == "__main__":
    fire.Fire()
