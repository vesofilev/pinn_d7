#!/usr/bin/env python3
"""
Analyze HotFreeZoom data to find multi-valued region and phase boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_phase_boundaries(data_file):
    """
    Analyze the free energy data to find phase boundaries.
    Returns: (m_min, m_bh_max, m_min_min, m_max)
    """
    data = np.loadtxt(data_file, delimiter=",")
    masses = data[:, 0]
    free_energies = data[:, 1]
    
    # Find the minimum point (critical point)
    min_idx = np.argmin(free_energies)
    m_critical = masses[min_idx]
    
    print(f"Critical mass (minimum F): {m_critical:.6f}")
    print(f"Overall mass range: [{masses.min():.6f}, {masses.max():.6f}]")
    
    # The multi-valued region is around the critical point
    # Black hole phase: from m_min to some point before critical
    # Minkowski phase: from m_max down to some point after critical
    
    # For a more sophisticated analysis, we could look at the derivative
    # but for now, let's use the structure of the data
    
    # Find where the curve starts to turn back (derivative changes sign significantly)
    dF_dm = np.gradient(free_energies, masses)
    
    # Find the transition points where derivative changes significantly
    # Black hole phase ends where derivative becomes very negative (before min)
    # Minkowski phase starts where derivative becomes very positive (after min)
    
    # Simple approach: use points where derivative magnitude is large
    large_deriv_threshold = np.std(dF_dm) * 2
    
    # Find last point before critical where derivative is not too negative
    before_critical = masses < m_critical
    bh_candidates = masses[before_critical]
    if len(bh_candidates) > 10:
        m_bh_max = bh_candidates[-10]  # Conservative estimate
    else:
        m_bh_max = bh_candidates[-1]
    
    # Find first point after critical where derivative is not too positive  
    after_critical = masses > m_critical
    min_candidates = masses[after_critical]
    if len(min_candidates) > 10:
        m_min_min = min_candidates[10]  # Conservative estimate
    else:
        m_min_min = min_candidates[0]
    
    return masses.min(), m_bh_max, m_min_min, masses.max(), m_critical

if __name__ == "__main__":
    m_min, m_bh_max, m_min_min, m_max, m_crit = analyze_phase_boundaries("data/HotFreeZoom.csv")
    
    print(f"\nPhase boundaries:")
    print(f"Black hole phase: [{m_min:.6f}, {m_bh_max:.6f}]")
    print(f"Multi-valued region: [{m_bh_max:.6f}, {m_min_min:.6f}]") 
    print(f"Minkowski phase: [{m_min_min:.6f}, {m_max:.6f}]")
    print(f"Critical mass: {m_crit:.6f}")
