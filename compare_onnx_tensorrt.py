#!/usr/bin/env python3
"""
Compare original and quantized ONNX models using TensorRT on Jetson Nano TX2.
Measures inference latency, throughput, and memory usage for both encoder and depth models.
"""
import os
import time
import numpy as np
import cv2
import psutil
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Paths to models
ORIG_ENCODER = 'models/encoder.onnx'
ORIG_DEPTH = 'models/depth.onnx'
QUANT_ENCODER = 'models/encoder_quantized.onnx'
QUANT_DEPTH = 'models/depth_quantized.onnx'

# Sample image path (change if needed)
SAMPLE_IMAGE = 'kitti_data/2011_09_26_drive_0035_sync/image_00/data/0000000000.png'

# Helper to build TensorRT engine from ONNX
def build_engine(onnx_path):
    with open(onnx_path, 'rb') as f:
        onnx_model = f.read()
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_model):
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28  # 256MB
    engine = builder.build_engine(network, config)
    return engine

# Helper to run inference and measure time
def run_inference(engine, input_data, n_runs=50):
    context = engine.create_execution_context()
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_mem = cuda.mem_alloc(input_data.nbytes)
            inputs.append(input_mem)
            bindings.append(int(input_mem))
        else:
            output_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            outputs.append(output_mem)
            bindings.append(int(output_mem))
    # Warmup
    cuda.memcpy_htod(inputs[0], input_data)
    context.execute_v2(bindings)
    # Timed runs
    start = time.time()
    for _ in range(n_runs):
        cuda.memcpy_htod(inputs[0], input_data)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(np.empty_like(input_data), outputs[0])
    end = time.time()
    avg_latency = (end - start) / n_runs
    throughput = n_runs / (end - start)
    return avg_latency, throughput

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    return mem

# Jetson hardware metrics
def get_jetson_metrics():
    if not JTOP_AVAILABLE:
        return {'gpu': None, 'ram': None}
    with jtop() as jetson:
        jetson.update()
        gpu = jetson.gpu['val'] if 'gpu' in jetson.gpu else None
        ram = jetson.memory['ram'] if 'ram' in jetson.memory else None
        return {'gpu': gpu, 'ram': ram}

# Preprocess image for model
def preprocess_image(img_path, input_shape):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Main comparison function
def compare_models():
    print("Building engines...")
    orig_encoder_engine = build_engine(ORIG_ENCODER)
    orig_depth_engine = build_engine(ORIG_DEPTH)
    quant_encoder_engine = build_engine(QUANT_ENCODER)
    quant_depth_engine = build_engine(QUANT_DEPTH)

    # Get input shape from encoder
    input_shape = orig_encoder_engine.get_binding_shape(0)
    img = preprocess_image(SAMPLE_IMAGE, input_shape)

    print("Evaluating original models...")
    mem_before = get_memory_usage()
    jetson_before = get_jetson_metrics()
    enc_lat, enc_thr = run_inference(orig_encoder_engine, img)
    depth_lat, depth_thr = run_inference(orig_depth_engine, img)
    mem_after = get_memory_usage()
    jetson_after = get_jetson_metrics()
    print(f"Original Encoder: Latency={enc_lat*1000:.2f}ms, FPS={enc_thr:.2f}, Throughput={enc_thr:.2f}fps, Mem={mem_after-mem_before:.2f}MB, Jetson GPU={jetson_after['gpu']}%, Jetson RAM={jetson_after['ram']}MB")
    print(f"Original Depth: Latency={depth_lat*1000:.2f}ms, FPS={depth_thr:.2f}, Throughput={depth_thr:.2f}fps, Mem={mem_after-mem_before:.2f}MB, Jetson GPU={jetson_after['gpu']}%, Jetson RAM={jetson_after['ram']}MB")

    print("Evaluating quantized models...")
    mem_before = get_memory_usage()
    jetson_before = get_jetson_metrics()
    enc_lat, enc_thr = run_inference(quant_encoder_engine, img)
    depth_lat, depth_thr = run_inference(quant_depth_engine, img)
    mem_after = get_memory_usage()
    jetson_after = get_jetson_metrics()
    print(f"Quantized Encoder: Latency={enc_lat*1000:.2f}ms, FPS={enc_thr:.2f}, Throughput={enc_thr:.2f}fps, Mem={mem_after-mem_before:.2f}MB, Jetson GPU={jetson_after['gpu']}%, Jetson RAM={jetson_after['ram']}MB")
    print(f"Quantized Depth: Latency={depth_lat*1000:.2f}ms, FPS={depth_thr:.2f}, Throughput={depth_thr:.2f}fps, Mem={mem_after-mem_before:.2f}MB, Jetson GPU={jetson_after['gpu']}%, Jetson RAM={jetson_after['ram']}MB")

    if not JTOP_AVAILABLE:
        print("Note: jtop is not installed. Jetson GPU/RAM metrics will not be shown. Install with 'pip install jetson-stats' on your Jetson device.")

if __name__ == "__main__":
    compare_models()
