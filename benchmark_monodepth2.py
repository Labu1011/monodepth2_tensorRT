#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script for Monodepth2 (original vs quantized) on Jetson Nano TX2
Compatible with Python 3.6

Features:
- Loads ONNX models (original & quantized) from models/
- Runs inference on KITTI test dataset (eigen splits)
- Measures FPS, latency, Jetson GPU/memory usage, throughput
- Saves metrics and visualizations (matplotlib)
"""
import os
import sys
import time
import csv
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

# --- CONFIG ---
MODELS = {
    'original': {
        'encoder': 'models/encoder.onnx',
        'depth': 'models/depth.onnx',
    },
    'quantized': {
        'encoder': 'models/encoder_quantized.onnx',
        'depth': 'models/depth_quantized.onnx',
    }
}
KITTI_IMAGE_DIR = 'kitti_data/2011_09_26_drive_0035_sync/image_00/data'
EIGEN_SPLIT = 'splits/eigen/test_files.txt'
RESULTS_DIR = 'benchmark_results'
BATCH_SIZE = 1

# --- UTILS ---
def load_image(path, resize=(640, 192)):
    img = Image.open(path).convert('RGB')
    img = img.resize(resize, Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    img_np = np.expand_dims(img_np, 0)  # BCHW
    return img_np

def get_tegrastats():
    """Run tegrastats and parse output for GPU/mem usage."""
    try:
        output = subprocess.check_output(['tegrastats'], timeout=2)
        output = output.decode('utf-8')
        # Example parse: RAM 1234/4096MB (lfb 123x4MB) CPU [1%@1234, ...] GPU 12%@1234
        ram = None
        gpu = None
        for part in output.split():
            if 'RAM' in part:
                ram = part
            if 'GPU' in part:
                gpu = part
        return {'ram': ram, 'gpu': gpu}
    except Exception:
        return {'ram': None, 'gpu': None}

def load_eigen_split(split_path):
    with open(split_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files

def load_trt_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def trt_infer(context, bindings, inputs, outputs, stream):
    # Transfer input data to device
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

def run_inference(encoder_engine_path, depth_engine_path, image_paths):
    encoder_engine = load_trt_engine(encoder_engine_path)
    depth_engine = load_trt_engine(depth_engine_path)
    encoder_context = encoder_engine.create_execution_context()
    depth_context = depth_engine.create_execution_context()
    encoder_inputs, encoder_outputs, encoder_bindings, encoder_stream = allocate_buffers(encoder_engine)
    depth_inputs, depth_outputs, depth_bindings, depth_stream = allocate_buffers(depth_engine)
    times = []
    gpu_stats = []
    for img_path in image_paths:
        img = load_image(img_path)
        encoder_inputs[0]['host'][:] = img.flatten()
        start = time.time()
        enc_out = trt_infer(encoder_context, encoder_bindings, encoder_inputs, encoder_outputs, encoder_stream)
        depth_inputs[0]['host'][:] = enc_out.flatten()
        depth_out = trt_infer(depth_context, depth_bindings, depth_inputs, depth_outputs, depth_stream)
        end = time.time()
        times.append(end - start)
        stats = get_tegrastats()
        gpu_stats.append(stats)
    return times, gpu_stats

def benchmark_model(model_type, image_paths):
    encoder_path = MODELS[model_type]['encoder']
    depth_path = MODELS[model_type]['depth']
    print('Benchmarking {} model...'.format(model_type))
    times, gpu_stats = run_inference(encoder_path, depth_path, image_paths)
    fps = len(times) / sum(times)
    latency = np.mean(times)
    # Parse GPU/mem usage
    gpu_usages = [s['gpu'] for s in gpu_stats if s['gpu']]
    ram_usages = [s['ram'] for s in gpu_stats if s['ram']]
    return {
        'fps': fps,
        'latency': latency,
        'gpu_usages': gpu_usages,
        'ram_usages': ram_usages,
        'times': times
    }

def save_results(results, out_path):
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

def plot_comparison(results, out_dir):
    # FPS & Latency
    labels = list(results.keys())
    fps = [results[k]['fps'] for k in labels]
    latency = [results[k]['latency'] for k in labels]
    plt.figure()
    plt.bar(labels, fps)
    plt.ylabel('FPS')
    plt.title('FPS Comparison')
    plt.savefig(os.path.join(out_dir, 'fps_comparison.png'))
    plt.close()
    plt.figure()
    plt.bar(labels, latency)
    plt.ylabel('Latency (s)')
    plt.title('Latency Comparison')
    plt.savefig(os.path.join(out_dir, 'latency_comparison.png'))
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    # Load test images from eigen split
    eigen_files = load_eigen_split(EIGEN_SPLIT)
    image_paths = [os.path.join(KITTI_IMAGE_DIR, os.path.basename(f).split()[0]) for f in eigen_files]
    # Only keep images that exist
    image_paths = [p for p in image_paths if os.path.exists(p)]
    # Run benchmarks
    results = {}
    for model_type in MODELS.keys():
        res = benchmark_model(model_type, image_paths)
        results[model_type] = res
        save_results(res, os.path.join(RESULTS_DIR, '{}_results.json'.format(model_type)))
    # Plot comparison
    plot_comparison(results, RESULTS_DIR)
    print('Benchmarking complete. Results saved in {}'.format(RESULTS_DIR))

if __name__ == '__main__':
    main()
