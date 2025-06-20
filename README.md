# Probe D7‑brane neural solver (`pinn_d7`)
Physics‑informed neural networks (PINNs) for solving probe D7‑brane
embeddings in holographic backgrounds and extracting their free–energy
curves.

The repository contains **two training pipelines**

| script | task | network learns |
| ------ | ---- | -------------- |
| `scripts/run_mass_scan.py` | *one network per mass* (**independent training**) | the profile \(L(\rho)\) for a single bare mass \(m\) |
| `scripts/run_mass_cond.py` | *one conditional network* (**joint training**) | the full function \(L(\rho,m)\) for all \(m\in[m_\text{min},m_\text{max}]\) |

Both workflows support the magnetic‑field potential (“magnetic”) and the hot
plasma potential (“hot” or “hot‑zoom”).

---

## 1. Quick start

```bash
# clone & enter the repo
git clone https://github.com/your‑username/pinn_d7.git
cd pinn_d7

# (optional but recommended) create a virtual env
python -m venv .venv
source .venv/bin/activate

# install Python dependencies
pip install torch matplotlib numpy
```

> **Note** : CUDA is used automatically if available; otherwise the code falls
> back to CPU.

---

## 2. Mass‑scan training (independent networks)

Train separate networks for a list of masses and plot the resulting
profiles together:

```bash
python scripts/run_mass_scan.py 0.05 0.1 0.2 0.3 \
       --epochs 40000 \
       --lr 1e-4 \
       --potential magnetic \
       --skip 0
```

*Relevant options*

| flag | default | meaning |
| ---- | ------- | ------- |
| `--potential` | `magnetic` | choose `magnetic`, `hot` or `hot-zoom` |
| `--epochs` | `30000` | optimisation steps per network |
| `--lr` | `1e-4` | Adam learning rate |
| `--sort` | `False` | train masses in descending order (warm‑start) |
| `--skip` | `0` | skip the first *n* masses when plotting |

Outputs:

* `plots/probe_d7_profiles_<epochs>_<potential>.png`  – profiles \(L(\rho)\)
* `plots/F_vs_m_<epochs>_<potential>.png`             – free energy \(F(m)\)

---

## 3. Conditional training (single network for **all** masses)

Learn a single network \(L(\rho,m)\) on the hot potential and stream the
free‑energy curve live during training:

```bash
python scripts/run_mass_cond.py \
       --epochs 30000 \
       --lr 1e-5 \
       --batch 2 \
       --m_min 0.0 \
       --m_max 0.8 \
       --live \
       --plot \
       --wait
```

*Key flags*

| flag | default | description |
| ---- | ------- | ----------- |
| `--batch` | `1` | masses per SGD step |
| `--live`  | off | open an interactive plot that updates every diagnostics interval |
| `--plot`  | off | after training, plot the final \(F(m)\) curve vs. the reference ODE result |
| `--wait`  | off | keep the script alive until **Enter** is pressed (lets you inspect plots) |
| `--resume` | —  | path to a checkpoint (`checkpoints_hot/ckpt_epoch_XXXXXX.pt`) to resume |

Checkpoints are written to `checkpoints_hot/` every `--checkpoint_every`
epochs and at the end of training.

---

## 4. Directory structure (important files)

```
pinn_d7/
├── data/                    # reference ODE solutions (CSV)
│   ├── HotFree.csv
│   └── MagFree.csv
├── pinn_d7/                 # Python package
│   ├── models.py            # neural‑network definitions (LNetwork, LNetworkM)
│   ├── losses.py            # DBI free‑energy functionals
│   ├── trainer.py           # utilities for mass‑scan training
│   └── train_conditional.py # conditional training loop
└── scripts/                 # CLI entry points
    ├── run_mass_scan.py
    └── run_mass_cond.py
```

---

## 5. Reproducibility & tips

* Fix random seeds (`torch.manual_seed(seed)`) for deterministic runs.
* For **Apple Silicon** (M‑series) you can allow Metal backend with
  `pinn_d7.trainer.get_default_device(allow_mps=True)`.
* On macOS + Anaconda you might need  
  `export KMP_DUPLICATE_LIB_OK=TRUE` (already set in `run_mass_cond.py`).

---

## 6. License

MIT License – see `LICENSE` for details.