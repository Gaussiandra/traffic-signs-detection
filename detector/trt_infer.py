import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class TRTEngineInfer:
    # Motivated by https://developer.nvidia.cn/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/#running_inference_from_the_tensorrt_engine

    def __init__(
        self, trt_path: str, input_tensor_name="input", output_tensor_name="output"
    ):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(trt_path, "rb") as f:
            serialized_engine = f.read()

        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        # Create pinned memory buffers
        self.host_input = cuda.pagelocked_empty(
            trt.volume(self.engine.get_tensor_shape(input_tensor_name)), dtype=np.float32
        )
        self.host_output = cuda.pagelocked_empty(
            trt.volume(self.engine.get_tensor_shape(output_tensor_name)), dtype=np.float32
        )

        # Allocate device memory for inputs and outputs.
        self.device_buffers = []
        for tensor_idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(tensor_idx)
            tensor_dtype = self.engine.get_tensor_dtype(name)
            buf = cuda.mem_alloc(
                trt.volume(self.engine.get_tensor_shape(name)) * tensor_dtype.itemsize
            )
            self.device_buffers.append(buf)

            self.context.set_tensor_address(name, buf)

        assert self.context.all_shape_inputs_specified
        assert self.context.all_binding_shapes_specified

        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

    @staticmethod
    def load_images_to_buffer(pics, pagelocked_buffer):
        preprocessed = np.asarray(pics).ravel()
        np.copyto(pagelocked_buffer, preprocessed)

    def do_trt_inference(self, use_profiler=False):
        # Transfer input data to the GPU.
        assert self.engine.get_tensor_name(0) == "input"

        cuda.memcpy_htod_async(self.device_buffers[0], self.host_input, self.stream)

        # Run inference.
        if use_profiler:
            self.context.profiler = trt.Profiler()

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back from the GPU.
        assert self.engine.get_tensor_name(1) == "output"
        cuda.memcpy_dtoh_async(self.host_output, self.device_buffers[1], self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        return self.host_output
