import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt
import os
import time
import threading
import multiprocessing
import psutil

def solve_profile_and_free_energy(L0, eps=1e-6, rhomax=20.0, H=1.0, R=1.0,
                                   rtol=1e-10, atol=1e-12, method='DOP853'):
    """
    Solve the ODE for L(rho) and compute the free energy integral I7
    for a given L0, using initial conditions from the small-rho expansion.
    Returns (mass, free_energy) = (L(rhomax), I7).
    """

    # Constants
    A = (H**2) * (R**4)  # equals 1 for H=R=1

    # Initial conditions from the given series at rho = eps
    denom = 2.0 * L0 * (L0**4 + 1.0)
    L_eps = L0 - (eps**2) / (2.0 * denom)  # since 1/(4 L0 (L0^4+1)) = 1/(2*denom)
    p_eps = -(eps) / (denom)               # -(1/(2 L0 (L0^4+1))) * eps
    z_eps = 0.0

    y0 = np.array([L_eps, p_eps, z_eps], dtype=float)

    # Define the ODE system:
    # y[0] = L, y[1] = p = L', y[2] = z = integral I7
    def rhs(rho, y):
        L = y[0]
        p = y[1]

        L2 = L * L
        L4 = L2 * L2
        L6 = L4 * L2
        p2 = p * p
        rho2 = rho * rho
        rho4 = rho2 * rho2

        # Avoid division-by-zero: we never call at rho=0 (start at eps>0)
        denom_main = rho * (rho2 + L2) * (A + rho4 + 2.0 * rho2 * L2 + L4)

        term1 = 2.0 * A * rho * L
        term2 = rho2 * (A + 3.0 * rho4) * p
        term3 = 3.0 * (A + 3.0 * rho4) * L2 * p
        term4 = 9.0 * rho2 * L4 * p
        term5 = 3.0 * L6 * p

        numer = (term1 + term2 + term3 + term4 + term5) * (1.0 + p2)

        F = numer / denom_main  # L'' + F = 0 -> L'' = -F
        Ldd = -F

        # Integrand for I7:
        t1 = math.sqrt(1.0 + 1.0 / ((rho2 + L2) ** 2))
        t2 = math.sqrt(1.0 + p2)
        integrand = (rho**3) * t1 * t2 - rho * math.sqrt(1.0 + rho4)

        return np.array([p, Ldd, integrand], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=(eps, rhomax),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        vectorized=False,
        max_step=np.inf
    )

    if sol.status < 0:
        raise RuntimeError(f"ODE solver failed for L0={L0}: {sol.message}")

    L_rhomax = sol.y[0, -1]
    I7_val = sol.y[2, -1]
    return L_rhomax, I7_val
def compute_Fs(
    L0_min=0.476802,
    L0_max=1.1449862,
    n_points=251,
    eps=1e-6,
    rhomax=20.0,
    H=1.0,
    R=1.0,
    rtol=1e-10,
    atol=1e-12,
    method='DOP853',
    verbose=True
):
    """
    Compute the Fs array: [[L(rhomax), I7], ...] over a grid of L0 values.
    """
    L0_values = np.linspace(L0_min, L0_max, n_points)
    Fs = np.empty((n_points, 2), dtype=float)
    
    # Timing arrays for individual computations
    individual_times = np.empty(n_points, dtype=float)
    
    if verbose:
        print(f"Processing {n_points} L0 values from {L0_min:.6f} to {L0_max:.6f}")
        print("Progress: ", end="", flush=True)

    for i, L0 in enumerate(L0_values):
        # Time each individual ODE solution
        start_time = time.perf_counter()
        
        mass, free_energy = solve_profile_and_free_energy(
            L0, eps=eps, rhomax=rhomax, H=H, R=R, rtol=rtol, atol=atol, method=method
        )
        
        end_time = time.perf_counter()
        individual_times[i] = end_time - start_time
        
        Fs[i, 0] = mass
        Fs[i, 1] = free_energy
        
        # Progress indicator
        if verbose:
            if (i + 1) % max(1, n_points // 20) == 0 or i == n_points - 1:
                progress = (i + 1) / n_points * 100
                print(f"{progress:.1f}%", end=" ", flush=True)

    if verbose:
        print()  # New line after progress
        
        # Print timing statistics
        mean_time = np.mean(individual_times)
        std_time = np.std(individual_times)
        min_time = np.min(individual_times)
        max_time = np.max(individual_times)
        
        print(f"\nPer-processor timing statistics:")
        print(f"  Mean time per ODE solution: {mean_time:.4f} ± {std_time:.4f} seconds")
        print(f"  Min time: {min_time:.4f} seconds")
        print(f"  Max time: {max_time:.4f} seconds")
        print(f"  Total computational time: {np.sum(individual_times):.3f} seconds")
        print(f"  Average rate: {1/mean_time:.1f} solutions per second")

    return Fs


if __name__ == "__main__":
    # System diagnostics ----------------------------------------------------------------
    print(f"{'='*60}")
    print(f"SYSTEM DIAGNOSTICS")
    print(f"{'='*60}")
    
    # CPU information
    cpu_count_logical = multiprocessing.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    current_process = psutil.Process()
    
    print(f"CPU cores (logical): {cpu_count_logical}")
    print(f"CPU cores (physical): {cpu_count_physical}")
    print(f"Current process PID: {current_process.pid}")
    print(f"Current thread count: {threading.active_count()}")
    print(f"Main thread: {threading.current_thread().name}")
    
    # Check CPU affinity if available
    try:
        cpu_affinity = current_process.cpu_affinity()
        print(f"CPU affinity: {cpu_affinity}")
    except (AttributeError, OSError):
        print("CPU affinity: Not available on this system")
    
    # Check NumPy/SciPy threading
    print(f"\nLibrary threading info:")
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        # Check for BLAS threading
        try:
            config = np.__config__.show()
            print("NumPy config available (check output above for BLAS threading)")
        except:
            print("NumPy config not accessible")
    except:
        pass
    
    # Check environment variables that control threading
    threading_env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS']
    print(f"\nThreading environment variables:")
    for var in threading_env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    print(f"{'='*60}\n")

    # Parameters matching the Mathematica code
    R = 1.0
    H = 1.0
    rhomax = 20.0
    eps = 1e-6

    # Monitor system resources before computation
    cpu_percent_before = psutil.cpu_percent(interval=1)
    memory_before = current_process.memory_info()
    threads_before = current_process.num_threads()
    
    print(f"Pre-computation system state:")
    print(f"  CPU usage: {cpu_percent_before:.1f}%")
    print(f"  Memory usage: {memory_before.rss / 1024 / 1024:.1f} MB")
    print(f"  Thread count: {threads_before}")

    # Start timing -------------------------------------------------------------------
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    # Compute Fs with 251 points from 0.476802 to 1.1449862
    print("\nComputing ODE solution for magnetic free energy...")
    Fs = compute_Fs(
        L0_min=0.476802,
        L0_max=1.1449862,
        n_points=251,
        eps=eps,
        rhomax=rhomax,
        H=H,
        R=R,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853'  # You can also try 'LSODA' or 'Radau' if needed
    )

    # End timing ---------------------------------------------------------------------
    cpu_end = time.process_time()
    wall_end = time.perf_counter()
    
    cpu_time = cpu_end - cpu_start
    wall_time = wall_end - wall_start

    # Monitor system resources after computation
    cpu_percent_after = psutil.cpu_percent(interval=1)
    memory_after = current_process.memory_info()
    threads_after = current_process.num_threads()
    
    print(f"\nPost-computation system state:")
    print(f"  CPU usage: {cpu_percent_after:.1f}%")
    print(f"  Memory usage: {memory_after.rss / 1024 / 1024:.1f} MB")
    print(f"  Thread count: {threads_after}")
    print(f"  Thread count change: {threads_after - threads_before}")
    
    # Check for potential multithreading indicators
    print(f"\nMultithreading analysis:")
    efficiency_ratio = cpu_time / wall_time
    print(f"  CPU/Wall time ratio: {efficiency_ratio:.4f}")
    if efficiency_ratio > 1.1:
        print(f"  → LIKELY MULTITHREADED (ratio > 1.1)")
    elif efficiency_ratio > 0.9:
        print(f"  → SINGLE-THREADED (ratio ≈ 1.0)")
    else:
        print(f"  → POSSIBLE I/O WAIT OR SYSTEM OVERHEAD (ratio < 0.9)")
    
    if threads_after > threads_before:
        print(f"  → Thread count increased during computation: {threads_before} → {threads_after}")
    else:
        print(f"  → No thread count change detected")

    # Print and save results
    np.set_printoptions(precision=10, suppress=False)
    print("Fs array (columns: [mass = L(rhomax), free_energy = I7]):")
    print(Fs)

    # Save to CSV
    np.savetxt("Fs_ode.csv", Fs, delimiter=",", header="mass,free_energy", comments="")
    
    # Load the magnetic reference data for comparison
    mag_data_path = "data/MagFree.csv"
    if os.path.exists(mag_data_path):
        mag_data = np.loadtxt(mag_data_path, delimiter=",")
        mag_mass = mag_data[:, 0]
        mag_free_energy = mag_data[:, 1]
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot ODE results
        plt.plot(Fs[:, 0], Fs[:, 1], 'b-', linewidth=2, label='ODE Solution (I7 integral)')
        
        # Plot magnetic reference data
        plt.plot(mag_mass, mag_free_energy, 'r--', linewidth=2, label='Magnetic Reference Data')
        
        plt.xlabel('Mass (m)')
        plt.ylabel('Free Energy (F)')
        plt.title('Free Energy vs Mass: ODE Solution vs Magnetic Reference Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig("free_energy_comparison_magnetic.png", dpi=300, bbox_inches="tight")
        print("\nComparison plot saved as 'free_energy_comparison_magnetic.png'")
        
        # Show basic statistics comparison
        print(f"\nODE Solution Statistics:")
        print(f"  Mass range: {Fs[:, 0].min():.6f} to {Fs[:, 0].max():.6f}")
        print(f"  Free energy range: {Fs[:, 1].min():.6f} to {Fs[:, 1].max():.6f}")
        
        print(f"\nMagnetic Reference Data Statistics:")
        print(f"  Mass range: {mag_mass.min():.6f} to {mag_mass.max():.6f}")
        print(f"  Free energy range: {mag_free_energy.min():.6f} to {mag_free_energy.max():.6f}")
        
        # Find overlapping mass range for better comparison
        mass_min = max(Fs[:, 0].min(), mag_mass.min())
        mass_max = min(Fs[:, 0].max(), mag_mass.max())
        
        # Interpolate both datasets to common mass points for comparison
        
        # Common mass points in overlapping range
        common_mass = np.linspace(mass_min, mass_max, 100)
        
        # Interpolate ODE results
        f_ode = interp1d(Fs[:, 0], Fs[:, 1], kind='linear', bounds_error=False, fill_value=np.nan)
        ode_interp = f_ode(common_mass)
        
        # Interpolate magnetic data
        f_mag = interp1d(mag_mass, mag_free_energy, kind='linear', bounds_error=False, fill_value=np.nan)
        mag_interp = f_mag(common_mass)
        
        # Calculate difference where both are valid
        valid_mask = ~(np.isnan(ode_interp) | np.isnan(mag_interp))
        if np.any(valid_mask):
            diff = ode_interp[valid_mask] - mag_interp[valid_mask]
            rmse = np.sqrt(np.mean(diff**2))
            max_abs_diff = np.max(np.abs(diff))
            
            print(f"\nComparison in overlapping mass range [{mass_min:.6f}, {mass_max:.6f}]:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Max absolute difference: {max_abs_diff:.6f}")
        
        plt.show()
    else:
        print(f"\nWarning: Could not find magnetic reference data at {mag_data_path}")
        print("Saving ODE results only...")
        
        # Plot ODE results only
        plt.figure(figsize=(10, 6))
        plt.plot(Fs[:, 0], Fs[:, 1], 'b-', linewidth=2, label='ODE Solution (I7 integral)')
        plt.xlabel('Mass (m)')
        plt.ylabel('Free Energy (F)')
        plt.title('Free Energy vs Mass: ODE Solution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("free_energy_ode_only.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Print timing information -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ODE COMPUTATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total CPU time: {cpu_time:.3f} seconds ({cpu_time/60:.2f} minutes)")
    print(f"Total wall time: {wall_time:.3f} seconds ({wall_time/60:.2f} minutes)")
    print(f"CPU efficiency: {cpu_time/wall_time:.2f}x")
    print(f"{'='*60}")