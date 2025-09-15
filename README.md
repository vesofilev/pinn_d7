# Probe D7-brane neural solver (pinn_d7)

Physics-informed neural networks (PINNs) for solving probe D7-brane
embeddings in holographic backgrounds and extrac<pre>
flag	default	description:  
--potential	hot	only hot is officially supported †  
--epochs	20000	number of training epochs  
--lr_L	1e-4	learning rate for the L-network  
--lr_V	5e-4	learning rate for the V-network  
--batch	1	masses per SGD step  
--rho_max	20.0	UV cutoff ρ_max  
--N_grid	2000	total collocation points  
--cut	5.0	boundary between dense & coarse zones  
--m_min	0.0	minimum mass  
--m_max	0.8	maximum mass  
--checkpoint_dir	checkpoints_inverse	directory for checkpoints  
--resume	None	resume from checkpoint file  
--hidden	16	hidden layer size  
--depth	2	number of hidden layers  
--live	off	live plots of F(m) and V(r) during training  
--plot	off	after training, save static plots for F(m) and V(r)  
--wait	off	keep the script alive until Enter is pressed  

† Magnetic potentials may be trained if the ANN's small-rho behaviour is adjusted accordingly.
</pre>ree-energy
and condensate curves. The solver can map phase transitions in the
holographic dual theory, including hysteresis loops and branch tracking
near critical points.

The repository now contains three complementary training pipelines

| Script                      | Task                                      | Network Learns                                  | Potentials Supported      |
|----------------------------|-------------------------------------------|--------------------------------------------------|---------------------------|
| `scripts/run_mass_scan.py` | one network per mass (independent training) | the profile L(ρ) for a single bare mass *m*     | magnetic, hot, hot-zoom   |
| `scripts/run_mass_cond.py` | one conditional network (joint training)   | the full function L(ρ, *m*) for all *m* in [*m*ₘᵢₙ, *m*ₘₐₓ] | magnetic, hot             |
| `scripts/run_inverse.py`   | inverse problem (joint reconstruction)     | both L(ρ, *m*) and the radial potential V(*r*)  | hot†                      |

† Magnetic potentials could be supported provided the asymptotic
behaviour of the ANN near the origin (\rho \to 0) is modified to
match the magnetic-brane boundary conditions.

⸻

## 1. Quick start

### clone & enter the repo
git clone https://github.com/your-username/pinn_d7.git
cd pinn_d7

### (optional but recommended) create a virtual env
python -m venv .venv
source .venv/bin/activate

### install Python dependencies (CPU); add + `torch-cuda` if you have a GPU
pip install torch matplotlib numpy scipy psutil

# Or install from requirements.txt
pip install -r requirements.txt

Tip  CUDA/GPU usage is automatic; otherwise the code silently falls
back to CPU.
macOS + OpenMP quirk is handled internally (KMP_DUPLICATE_LIB_OK=TRUE).

⸻

## 2. Mass-scan training (independent networks)

Train individual networks for a list of masses and compare their profiles:
<pre>
python scripts/run_mass_scan.py 0.05 0.1 0.2 0.3 \
       --epochs 40000 \
       --lr 1e-4 \
       --potential magnetic \
       --skip 0
</pre>

### Phase transition tracking with hysteresis capture

For studying phase transitions near critical points, use closely spaced masses around the transition region. The `hot-zoom` potential is specifically designed for this purpose:

<pre>
python scripts/run_mass_scan.py 0.9 0.9125 0.915 0.9175 0.920 0.9256 0.935 0.9325 0.930 0.9275 0.925 0.9235 0.921 0.91865 0.9163 \
       --epochs 40000 \
       --potential hot-zoom \
       --lr 5e-4 \
       --skip 1
</pre>

This command trains networks at 15 carefully chosen mass points that span the phase transition region (m ≈ 0.916 - 0.935). The non-monotonic ordering of masses helps capture both the black hole branch (stable for lower masses) and the Minkowski branch (stable for higher masses), revealing the hysteresis loop characteristic of first-order phase transitions. The resulting free energy curve F(m) will show the characteristic "S-curve" with multiple solutions in the transition region.

<pre>
flag	default	meaning:  
--potential	magnetic	choose magnetic, hot or hot-zoom  
--epochs	30000	optimisation steps per network  
--lr	1e-4	Adam learning rate  
--sort	False	train masses in descending order (warm-start)  
--skip	0	skip the first n masses when plotting  
--no-display	off	save plots without displaying them  
</pre>
Outputs  
	•	plots/probe_d7_profiles_<epochs>_<potential>.png – profiles L(\rho)  
	•	plots/F_vs_m_<epochs>_<potential>.png      – free energy F(m)  

⸻

## 3. Conditional training (single network for all masses)

The conditional trainer learnt to speak both the hot-plasma and magnetic
dialects.
It can live-stream the free-energy curve and (for the magnetic case) the
condensate c(m)=\partial_mF.

### HOT plasma, live plot + final static plot
<pre>
python scripts/run_mass_cond.py \
       --epochs 30000 \
       --lr 1e-5 \
       --batch 2 \
       --m_min 0.0 --m_max 0.9 \
       --potential hot \
       --live --plot --wait
</pre>
### Magnetic field, plus condensate evaluation
<pre>
python scripts/run_mass_cond.py \
       --epochs 120000 \
       --lr 1e-4 \
       --batch 1 \
       --m_min 0.0 --m_max 1.0 \
       --potential magnetic \
       --plot          # <- also produces the condensate plot
</pre>
Key flags (new & old)

<pre>
flag	default	description:  
--potential	hot	choose hot or magnetic  
--epochs	30000	number of training epochs  
--lr	1e-5	Adam learning rate  
--batch	1	masses per SGD step (mini-batch size)  
--m_min	0.0	minimum mass in training range  
--m_max	0.8	maximum mass in training range  
--rho_max	20.0	UV cutoff ρ_max  
--N_grid	2000	total collocation points  
--cut	5.0	boundary between dense & coarse zones  
--hidden	16	hidden layer size  
--depth	2	number of hidden layers  
--seed	None	random seed for reproducible results  
--init_strategy	physics_informed	weight initialization (physics_informed, small_xavier, kaiming_small, uniform_small)  
--checkpoint_dir	checkpoints	directory where checkpoints are saved  
--checkpoint_every	1000	epochs between checkpoints  
--resume	—	path to a checkpoint (checkpoints_<pot>/ckpt_epoch_XXXXXX.pt) to resume  
--live	off	open an interactive F(m) plot that refreshes every diagnostics interval  
--plot	off	after training, save static plots for F(m) (both potentials) and dF/dm (magnetic only)  
--eval_mae	on	evaluate MAE against reference data  
--mae_linear	off	use linear scale for MAE plot instead of log scale  
--wait	off	keep the script alive until Enter is pressed (gives you time to inspect plots)  
</pre>

Outputs  
	•	Free energy plots/F_vs_m_<epochs>_<potential>_conditional.png  
	•	Condensate (magnetic only) plots/c_vs_m_magnetic_<epochs>.png  
	•	Checkpoints every 1k epochs, written to checkpoints_<potential>/ckpt_epoch_XXXXXX.pt  

⸻

## 4. Inverse-problem training (joint recovery of L(\rho,m) and V(r))

The inverse trainer simultaneously learns the D7-brane embedding and the underlying radial potential.
Training alternates between updating the embedding network (L) and the potential network (V) every other gradient step, guided by the reference free-energy curve.

Currently only the hot-plasma potential is fully supported.
Magnetic backgrounds would require modifying the ANN’s near-origin
asymptotics to satisfy the magnetic-brane boundary conditions.

<pre>
python scripts/run_inverse.py --potential hot \
       --epochs 50000 \
       --checkpoint_dir checkpoints_inverse_hot \
       --lr_L 1e-4 --lr_V 5e-4 \
       --live --wait
</pre>
<pre>
flag	default	description:  
--potential	hot	only hot is officially supported †  
--lr_L	1e-4	learning rate for the L-network  
--lr_V	5e-4	learning rate for the V-network  
--batch	1	masses per SGD step  
--live	off	live plots of F(m) and V(r) during training  
--plot	off	after training, save static plots for F(m) and V(r)  
--wait	off	keep the script alive until Enter is pressed  

† Magnetic potentials may be trained if the ANN’s small-rho behaviour is adjusted accordingly.
</pre>

Outputs  
	•	plots/F_and_V_<epochs>_inverse.png – side-by-side F(m) and V(r)  
	•	Checkpoints every 1 k epochs, written to checkpoints_inverse/ckpt_epoch_XXXXXX.pt  

⸻

## 5. Directory structure (data + code)
<pre>
pinn_d7/  
├── data/                          # reference ODE solutions (CSV)  
│   ├── HotFree.csv                # F(m) for AdS-BH background  
│   ├── HotFreeZoom.csv            # F(m) for AdS-BH background near phase transition (m ≈ 0.91-0.94)  
│   ├── HotFree0_0_9.csv           # F(m) for AdS-BH background (m ∈ [0,0.9])  
│   ├── MagFree.csv                # F(m) for SYM with magnetic field  
│   └── MagCondensate.csv          # c(m)=∂mF reference curve for SYM with magnetic field  
├── pinn_d7/                       # Python package  
│   ├── models.py                  # neural-network definitions (LNetwork, LNetworkM)  
│   ├── losses.py                  # DBI free-energy functionals  
│   ├── train_conditional.py       # ★ new conditional training loop (hot & magnetic)  
│   ├── train_inverse.py           # inverse-problem training loop (joint L & V)  
│   └── trainer.py                 # utilities for mass-scan training  
└── scripts/                       # CLI entry points  
    ├── run_mass_scan.py           # independent (per-mass) training  
    ├── run_inverse.py             # inverse-problem training  
    ├── run_mass_cond.py           # conditional (multi-mass) training  
    ├── ode_free_energy_magnetic.py # ODE reference solutions for validation  
    └── analyze_phases.py          # phase boundary analysis tools  

</pre>

The `data/` directory contains reference solutions obtained by solving the D7-brane embedding ODEs directly using high-precision numerical integration (scipy's DOP853 method). These serve as ground truth for validating the PINN results. The `HotFreeZoom.csv` dataset specifically covers the phase transition region where the holographic dual theory exhibits a first-order transition between black hole and Minkowski embeddings.

⸻

## 6. Reproducibility & tips  
	•	Determinism: set torch.manual_seed(seed) before training.  
	•	Apple Silicon: to enable the Metal backend pass pinn_d7.trainer.get_default_device(allow_mps=True).  
	•	OpenMP on macOS/conda: already patched via os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" in run_mass_cond.py.  
	•	Phase transitions: use `hot-zoom` potential with non-monotonic mass ordering to capture hysteresis loops and multiple stable branches near critical points (m ≈ 0.916-0.935).  
	•	Convergence: for phase transition studies, increase epochs to 40k+ and use learning rates around 5e-4 for better accuracy near critical regions.  

⸻

## 7. License

MIT License – see LICENSE for details.
