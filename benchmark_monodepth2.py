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
import onnxruntime as ort
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
KITTI_ROOT = 'kitti_data'
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

def run_inference(encoder_onnx_path, depth_onnx_path, image_paths):
    print(f"Loading ONNX models: {encoder_onnx_path}, {depth_onnx_path}")
    encoder_session = ort.InferenceSession(encoder_onnx_path)
    depth_session = ort.InferenceSession(depth_onnx_path)
    times = []
    gpu_stats = []
    
    print(f"Processing {len(image_paths)} images...")
    for i, img_path in enumerate(image_paths):
        if i % 50 == 0:  # Progress indicator
            print(f"Processed {i}/{len(image_paths)} images...")
        
        try:
            img = load_image(img_path)
            start = time.time()
            enc_out = encoder_session.run(None, {'input': img})
            depth_out = depth_session.run(None, {'input': enc_out[0]})
            end = time.time()
            times.append(end - start)
            stats = get_tegrastats()
            gpu_stats.append(stats)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully processed {len(times)} images")
    return times, gpu_stats

def benchmark_model(model_type, image_paths):
    encoder_path = MODELS[model_type]['encoder']
    depth_path = MODELS[model_type]['depth']
    print('Benchmarking {} model...'.format(model_type))
    times, gpu_stats = run_inference(encoder_path, depth_path, image_paths)
    
    # Handle case where no valid times were recorded
    if not times or sum(times) == 0:
        print("Warning: No valid inference times recorded for {} model".format(model_type))
        return {
            'fps': 0.0,
            'latency': 0.0,
            'gpu_usages': [],
            'ram_usages': [],
            'times': [],
            'total_images': len(image_paths),
            'processed_images': 0
        }
    
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
        'times': times,
        'total_images': len(image_paths),
        'processed_images': len(times)
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


def get_image_path(kitti_root, split_line):
    # Handles lines like "2011_09_26_drive_0035_sync 0000000000 l"
    parts = split_line.strip().split()
    if len(parts) >= 2:
        drive, frame = parts[0], parts[1]
        full_path = os.path.join(kitti_root, drive, 'image_02', 'data', frame + '.png')
        return full_path
    return None

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load test images from eigen split
    print(f"Loading test files from {EIGEN_SPLIT}")
    eigen_files = load_eigen_split(EIGEN_SPLIT)
    print(f"Found {len(eigen_files)} test files in split")
    
    image_paths = [get_image_path(KITTI_ROOT, f) for f in eigen_files]
    image_paths = [p for p in image_paths if p and os.path.exists(p)]
    print(f"Found {len(image_paths)} valid image paths")
    
    if len(image_paths) == 0:
        print("Error: No valid images found! Check KITTI data path and format.")
        return
    
    # Show some example paths for debugging
    print("Example image paths:")
    for i, path in enumerate(image_paths[:5]):
        print(f"  {i+1}: {path}")
    
    # Run benchmarks
    results = {}
    for model_type in MODELS.keys():
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_type} model")
        print(f"{'='*50}")
        
        res = benchmark_model(model_type, image_paths)
        results[model_type] = res
        
        # Print results
        print(f"Results for {model_type}:")
        print(f"  Processed images: {res['processed_images']}/{res['total_images']}")
        print(f"  FPS: {res['fps']:.2f}")
        print(f"  Average latency: {res['latency']:.4f}s")
        
        save_results(res, os.path.join(RESULTS_DIR, '{}_results.json'.format(model_type)))
    
    # Plot comparison
    plot_comparison(results, RESULTS_DIR)
    print(f'\nBenchmarking complete. Results saved in {RESULTS_DIR}')
    
    # Print summary
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    for model_type, res in results.items():
        print(f"{model_type.upper()} MODEL:")
        print(f"  FPS: {res['fps']:.2f}")
        print(f"  Latency: {res['latency']:.4f}s")
        print(f"  Images processed: {res['processed_images']}/{res['total_images']}")
        print()

if __name__ == '__main__':
    main()
