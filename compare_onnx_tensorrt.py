#!/usr/bin/env python3
"""
Compare original and quantized ONNX models using TensorRT on Jetson Nano TX2.
Measures inference latency, throughput, and memory usage for both encoder and depth models.
"""
import matplotlib.pyplot as plt
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
import os
import time
import numpy as np
import cv2
import psutil
import onnxruntime as ort

# Paths to models
ORIG_ENCODER = 'models/encoder.onnx'
ORIG_DEPTH = 'models/depth.onnx'
QUANT_ENCODER = 'models/encoder_quantized.onnx'
QUANT_DEPTH = 'models/depth_quantized.onnx'

# Sample image path (change if needed)
SAMPLE_IMAGE = 'kitti_data/2011_09_26_drive_0035_sync/image_00/data/0000000000.png'

def preprocess_image(img_path, input_shape):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def run_onnx_inference(model_path, input_data, n_runs=50):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # Warmup
    session.run(None, {input_name: input_data})
    start = time.time()
    for _ in range(n_runs):
        session.run(None, {input_name: input_data})
    end = time.time()
    avg_latency = (end - start) / n_runs
    fps = n_runs / (end - start)
    return avg_latency, fps

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
def compare_models():
    # Get input shape from encoder ONNX
    session = ort.InferenceSession(ORIG_ENCODER)
    input_shape = session.get_inputs()[0].shape
    img = preprocess_image(SAMPLE_IMAGE, input_shape)


    metrics = {}
    for name, path in [
        ('Original Encoder', ORIG_ENCODER),
        ('Original Depth', ORIG_DEPTH),
        ('Quantized Encoder', QUANT_ENCODER),
        ('Quantized Depth', QUANT_DEPTH)
    ]:
        mem_before = get_memory_usage()
        jetson_before = get_jetson_metrics() if JTOP_AVAILABLE else {'gpu': None, 'ram': None}
        latency, fps = run_onnx_inference(path, img)
        mem_after = get_memory_usage()
        jetson_after = get_jetson_metrics() if JTOP_AVAILABLE else {'gpu': None, 'ram': None}
        metrics[name] = {
            'Latency': latency * 1000,
            'FPS': fps,
            'Mem': mem_after - mem_before,
            'GPU': jetson_after['gpu'],
            'RAM': jetson_after['ram'],
            'Throughput': fps  # FPS is equivalent to throughput here
        }
        print(f"{name}: Latency={latency*1000:.2f}ms, FPS={fps:.2f}, Mem={mem_after-mem_before:.2f}MB, Jetson GPU={jetson_after['gpu']}%, Jetson RAM={jetson_after['ram']}MB")

    # Visualization
    labels = list(metrics.keys())
    latency = [metrics[k]['Latency'] for k in labels]
    fps = [metrics[k]['FPS'] for k in labels]
    mem = [metrics[k]['Mem'] for k in labels]
    gpu = [metrics[k]['GPU'] if metrics[k]['GPU'] is not None else 0 for k in labels]
    ram = [metrics[k]['RAM'] if metrics[k]['RAM'] is not None else 0 for k in labels]
    throughput = [metrics[k]['Throughput'] for k in labels]

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.bar(labels, latency, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('Latency (ms)')
    plt.title('Model Latency Comparison')

    plt.subplot(2, 3, 2)
    plt.bar(labels, fps, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('FPS')
    plt.title('Model FPS Comparison')

    plt.subplot(2, 3, 3)
    plt.bar(labels, throughput, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('Throughput (fps)')
    plt.title('Model Throughput Comparison')

    plt.subplot(2, 3, 4)
    plt.bar(labels, mem, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('Memory Usage (MB)')
    plt.title('Model Memory Usage')

    plt.subplot(2, 3, 5)
    plt.bar(labels, gpu, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('Jetson GPU Usage (%)')
    plt.title('Jetson GPU Usage')

    plt.subplot(2, 3, 6)
    plt.bar(labels, ram, color=['blue', 'cyan', 'orange', 'red'])
    plt.ylabel('Jetson RAM Usage (MB)')
    plt.title('Jetson RAM Usage')

    plt.tight_layout()
    plt.savefig('onnxruntime_model_comparison.png')
    plt.show()












if __name__ == "__main__":
    compare_models()
