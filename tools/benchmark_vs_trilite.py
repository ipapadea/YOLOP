"""
Benchmark script to compare TwinLiteNet2Scaled against TriLiteNet
Author: Automated Analysis
Date: January 2026

This script measures:
1. Model parameters and FLOPs
2. Inference speed (FPS) at different batch sizes
3. Memory consumption
4. Comparison with TriLiteNet benchmarks
"""

import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.models import get_net
from lib.config import cfg

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")
    print("Continuing without FLOPs calculation...")


class ModelBenchmark:
    def __init__(self, model, device='cuda', input_size=(384, 640)):
        self.model = model.to(device)
        self.device = device
        self.input_size = input_size
        self.model.eval()
        
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        }
    
    def calculate_flops(self, batch_size=1):
        """Calculate FLOPs using thop"""
        if not HAS_THOP:
            return None
        
        input_tensor = torch.randn(batch_size, 3, self.input_size[0], self.input_size[1]).to(self.device)
        flops, params = profile(self.model, inputs=(input_tensor,), verbose=False)
        
        return {
            'flops': flops,
            'params': params,
            'flops_G': flops / 1e9,
            'params_M': params / 1e6
        }
    
    def measure_inference_speed(self, batch_size=1, num_warmup=100, num_iterations=1000):
        """Measure inference speed (FPS)"""
        input_tensor = torch.randn(batch_size, 3, self.input_size[0], self.input_size[1]).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_tensor)
        
        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = (num_iterations * batch_size) / total_time
        latency = (total_time / num_iterations) * 1000  # ms
        
        return {
            'fps': fps,
            'latency_ms': latency,
            'total_time_s': total_time,
            'batch_size': batch_size
        }
    
    def measure_memory(self, batch_size=1):
        """Measure GPU memory consumption"""
        if self.device != 'cuda':
            return None
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        input_tensor = torch.randn(batch_size, 3, self.input_size[0], self.input_size[1]).to(self.device)
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        
        return {
            'allocated_MB': memory_allocated,
            'reserved_MB': memory_reserved,
            'batch_size': batch_size
        }


def print_comparison_table(our_results, trilite_benchmarks):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("COMPARISON WITH TRILITE-NET BASE")
    print("="*80)
    
    print("\n{:<30} {:<20} {:<20} {:<15}".format(
        "Metric", "TriLiteNet-base", "Your Model", "Difference"
    ))
    print("-"*80)
    
    # Parameters
    trilite_params = trilite_benchmarks['params_M']
    our_params = our_results['params']['total_M']
    param_diff = ((our_params - trilite_params) / trilite_params) * 100
    print("{:<30} {:<20} {:<20} {:<15}".format(
        "Parameters (M)",
        f"{trilite_params:.2f}",
        f"{our_params:.2f}",
        f"{param_diff:+.1f}%"
    ))
    
    # FLOPs
    if our_results.get('flops'):
        trilite_flops = trilite_benchmarks['flops_G']
        our_flops = our_results['flops']['flops_G']
        flops_diff = ((our_flops - trilite_flops) / trilite_flops) * 100
        print("{:<30} {:<20} {:<20} {:<15}".format(
            "FLOPs (G)",
            f"{trilite_flops:.2f}",
            f"{our_flops:.2f}",
            f"{flops_diff:+.1f}%"
        ))
    
    print("-"*80)
    
    # Speed comparisons
    for batch_size in [1, 8, 32]:
        trilite_fps = trilite_benchmarks['fps'].get(f'batch_{batch_size}', 'N/A')
        our_fps = our_results['speed'].get(f'batch_{batch_size}', {}).get('fps', 'N/A')
        
        if trilite_fps != 'N/A' and our_fps != 'N/A':
            fps_diff = ((our_fps - trilite_fps) / trilite_fps) * 100
            print("{:<30} {:<20} {:<20} {:<15}".format(
                f"FPS (batch={batch_size})",
                f"{trilite_fps:.0f}",
                f"{our_fps:.0f}",
                f"{fps_diff:+.1f}%"
            ))
    
    print("="*80)
    
    # Summary
    print("\nSUMMARY:")
    if our_results.get('flops'):
        if our_params < trilite_params and our_flops < trilite_flops:
            print("✅ Your model is MORE EFFICIENT (fewer params & FLOPs)")
        elif our_params > trilite_params * 1.5 or our_flops > trilite_flops * 1.5:
            print("⚠️  Your model is SIGNIFICANTLY LARGER - consider optimization")
        else:
            print("➡️  Your model has similar efficiency")
    
    print("\nNOTE: Performance metrics (Recall, mAP, mIoU) need to be evaluated on BDD100K")
    print("="*80 + "\n")


def main():
    print("="*80)
    print("TwinLiteNet2Scaled vs TriLiteNet Benchmark")
    print("="*80)
    
    # TriLiteNet-base benchmarks (from paper)
    trilite_benchmarks = {
        'params_M': 2.35,
        'flops_G': 7.72,
        'fps': {
            'batch_1': 151,
            'batch_8': 1081,
            'batch_32': 1641
        },
        'performance': {
            'detection_recall': 85.6,
            'detection_map50': 72.3,
            'da_miou': 92.4,
            'll_acc': 82.3,
            'll_iou': 29.8
        }
    }
    
    # Load your model
    print("\n1. Loading TwinLiteNet2Scaled model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    model = get_net(cfg)
    benchmark = ModelBenchmark(model, device=device, input_size=(384, 640))
    
    # Count parameters
    print("\n2. Counting parameters...")
    params = benchmark.count_parameters()
    print(f"   Total parameters: {params['total_M']:.2f}M")
    print(f"   Trainable parameters: {params['trainable_M']:.2f}M")
    
    # Calculate FLOPs
    flops_result = None
    if HAS_THOP:
        print("\n3. Calculating FLOPs...")
        flops_result = benchmark.calculate_flops(batch_size=1)
        print(f"   FLOPs: {flops_result['flops_G']:.2f}G")
    
    # Measure inference speed
    print("\n4. Measuring inference speed...")
    speed_results = {}
    for batch_size in [1, 8, 32]:
        print(f"\n   Batch size {batch_size}:")
        speed = benchmark.measure_inference_speed(batch_size=batch_size, num_iterations=500)
        speed_results[f'batch_{batch_size}'] = speed
        print(f"     FPS: {speed['fps']:.1f}")
        print(f"     Latency: {speed['latency_ms']:.2f} ms")
    
    # Measure memory
    if device == 'cuda':
        print("\n5. Measuring GPU memory...")
        for batch_size in [1, 8, 32]:
            memory = benchmark.measure_memory(batch_size=batch_size)
            print(f"   Batch {batch_size}: {memory['allocated_MB']:.1f} MB allocated")
    
    # Store all results
    our_results = {
        'params': params,
        'flops': flops_result,
        'speed': speed_results
    }
    
    # Print comparison table
    print_comparison_table(our_results, trilite_benchmarks)
    
    # Save results
    results_file = Path(BASE_DIR) / "benchmark_results.txt"
    with open(results_file, 'w') as f:
        f.write("TwinLiteNet2Scaled Benchmark Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Parameters: {params['total_M']:.2f}M\n")
        if flops_result:
            f.write(f"FLOPs: {flops_result['flops_G']:.2f}G\n")
        f.write("\nInference Speed:\n")
        for batch_name, speed in speed_results.items():
            f.write(f"  {batch_name}: {speed['fps']:.1f} FPS\n")
        
        f.write("\n\nComparison with TriLiteNet-base:\n")
        f.write(f"  TriLiteNet params: {trilite_benchmarks['params_M']}M\n")
        f.write(f"  TriLiteNet FLOPs: {trilite_benchmarks['flops_G']}G\n")
        f.write(f"  TriLiteNet FPS (batch=1): {trilite_benchmarks['fps']['batch_1']}\n")
    
    print(f"Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Evaluate your model on BDD100K dataset to get performance metrics")
    print("2. Compare detection Recall, mAP with TriLiteNet (85.6%, 72.3%)")
    print("3. Compare segmentation mIoU, Acc with TriLiteNet (92.4%, 82.3%)")
    print("4. If needed, optimize model size while maintaining performance")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
