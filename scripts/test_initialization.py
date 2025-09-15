#!/usr/bin/env python3
"""
Test different initialization strategies for PINN convergence.
"""
import sys
import pathlib
import subprocess
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

def test_initialization_strategy(strategy, seed=42, epochs=5000):
    """Test a specific initialization strategy."""
    print(f"\n{'='*60}")
    print(f"Testing initialization strategy: {strategy}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "run_mass_cond.py",
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--init_strategy", strategy,
        "--potential", "hot",
        "--m_min", "0.1",
        "--m_max", "0.5",
        "--lr", "1e-4",
        "--N_grid", "500",  # Smaller grid for faster testing
        "--hidden", "16",
        "--depth", "2"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            # Extract MAE from output
            lines = result.stdout.split('\n')
            mae_line = [line for line in lines if 'MAE:' in line]
            if mae_line:
                print(f"✓ Success! {mae_line[0]}")
                print(f"  Runtime: {end_time - start_time:.1f}s")
            else:
                print("✓ Success! (No MAE reported)")
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout (>5 minutes)")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    strategies = ['physics_informed', 'small_xavier', 'kaiming_small', 'uniform_small']
    
    print("Testing different weight initialization strategies for PINN convergence")
    print("Each test runs for 5000 epochs with the same random seed")
    
    for strategy in strategies:
        test_initialization_strategy(strategy)
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print("- 'physics_informed': Scaled Xavier + small final layer (recommended)")
    print("- 'small_xavier': Standard Xavier scaled down by 0.1")  
    print("- 'kaiming_small': Kaiming normal scaled down by 0.3")
    print("- 'uniform_small': Small uniform initialization")
    print("Try with --seed for reproducible results!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
