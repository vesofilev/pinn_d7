#!/usr/bin/env python3
"""
Check parallel computing capabilities and recommend optimal settings.
"""
import os
# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.multiprocessing as mp
try:
    import psutil
except ImportError:
    psutil = None

def check_cpu_info():
    """Check CPU information and threading capabilities."""
    print("=" * 60)
    print("CPU INFORMATION")
    print("=" * 60)
    
    cpu_count = mp.cpu_count()
    if psutil:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"Physical CPU cores: {physical_cores}")
        print(f"Logical CPU cores (with hyperthreading): {logical_cores}")
        
        # Check memory
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    else:
        print("Install psutil for detailed CPU info: pip install psutil")
        physical_cores = cpu_count // 2  # rough estimate
        logical_cores = cpu_count
    
    print(f"PyTorch reports: {cpu_count} cores")
    print(f"Current PyTorch threads: {torch.get_num_threads()}")
    
    return physical_cores, logical_cores

def check_gpu_info():
    """Check GPU information."""
    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - using CPU only")
        return 0
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")
    
    return gpu_count

def check_backends():
    """Check parallel backends."""
    print("\n" + "=" * 60)
    print("PARALLEL BACKENDS")
    print("=" * 60)
    
    backends = {
        "OpenMP": "OMP_NUM_THREADS",
        "MKL": "MKL_NUM_THREADS", 
        "VECLIB": "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR": "NUMEXPR_NUM_THREADS"
    }
    
    for name, env_var in backends.items():
        value = os.environ.get(env_var, "Not set")
        print(f"{name}: {value}")

def recommend_settings(physical_cores, logical_cores, gpu_count):
    """Recommend optimal parallel settings."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if gpu_count > 0:
        print("🚀 GPU TRAINING (Recommended):")
        print(f"   python run_mass_cond.py --device cuda")
        if gpu_count > 1:
            print(f"   python run_mass_cond.py --data_parallel  # Use {gpu_count} GPUs")
    
    print("\n💻 CPU TRAINING:")
    print(f"   Recommended threads: {physical_cores} (physical cores)")
    print(f"   Maximum threads: {logical_cores} (with hyperthreading)")
    print(f"   python run_mass_cond.py --num_threads {physical_cores}")
    
    print("\n⚡ PERFORMANCE TIPS:")
    print("   1. Use physical cores for CPU (avoid hyperthreading overhead)")
    print("   2. Larger batch sizes benefit more from parallelization")
    print("   3. GPU is typically 5-20x faster than CPU for neural networks")
    print("   4. Use smaller models (--hidden 8 --depth 2) for faster iteration")

def test_parallel_performance():
    """Quick parallel performance test."""
    print("\n" + "=" * 60)
    print("QUICK PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    
    # Test matrix multiplication scaling
    size = 1000
    x = torch.randn(size, size, dtype=torch.float64)
    y = torch.randn(size, size, dtype=torch.float64)
    
    # Single thread
    torch.set_num_threads(1)
    start = time.time()
    for _ in range(10):
        _ = torch.mm(x, y)
    single_time = time.time() - start
    
    # Multi thread
    torch.set_num_threads(torch.multiprocessing.cpu_count())
    start = time.time()
    for _ in range(10):
        _ = torch.mm(x, y)
    multi_time = time.time() - start
    
    speedup = single_time / multi_time
    print(f"Single thread: {single_time:.3f}s")
    print(f"Multi thread: {multi_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("✅ Good parallel scaling!")
    else:
        print("⚠️  Limited parallel scaling - check your setup")

def main():
    print("PyTorch Parallel Computing Capabilities Check")
    
    physical_cores, logical_cores = check_cpu_info()
    gpu_count = check_gpu_info()
    check_backends()
    recommend_settings(physical_cores, logical_cores, gpu_count)
    test_parallel_performance()
    
    print("\n" + "=" * 60)
    print("Copy and paste the recommended command above! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    main()
