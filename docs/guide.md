# User Guide & Technical Documentation

## 1. Project Overview

This repository implements a **Physics-Informed Neural Network (PINN)** for solving the two-dimensional steady-state incompressible Navier-Stokes equations in a channel flow configuration. The work is part of a multi-phase academic project in computational fluid dynamics.

### Governing Equations (Non-dimensional)

```
∂u/∂x + ∂v/∂y = 0                         (Continuity)
u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)   (x-momentum)
u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)   (y-momentum)
```

Where `u, v` are velocity components, `p` is pressure, and `ν` is kinematic viscosity (set to 1.0 in this work).

### Domain Configuration

- Channel geometry: x ∈ [-1.0, 1.0], y ∈ [-0.5, 0.5]
- Inlet (x = -1.0): uniform velocity u = 1.0, v = 0.0
- Walls (y = ±0.5): no-slip, u = v = 0.0
- Outlet (x = 1.0): zero-pressure condition p = 0.0

---

## 2. Getting Started

### 2.1 Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Dependencies

| Package | Purpose | Minimum Version |
|---------|---------|-----------------|
| `torch` | Neural network framework with autograd for PDE residuals | 2.0.0 |
| `numpy` | Numerical arrays for data generation | 1.21.0 |
| `matplotlib` | Visualization of results and loss histories | 3.5.0 |

---

## 3. Running the Project

### 3.1 Full Training Pipeline

```bash
# Standard run (Adam + LBFGS)
python train.py

# With custom hyperparameters
python train.py --hidden-sizes 64 64 64 64 64 --lr 5e-4 --epochs-adam 10000

# Resume from existing checkpoint
python train.py --resume

# Skip LBFGS refinement (Adam only)
python train.py --no-lbfgs
```

### 3.2 Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs-adam` | 5000 | Number of Adam optimizer epochs |
| `--epochs-lbfgs` | 1000 | Number of LBFGS refinement epochs |
| `--lr` | 1e-3 | Initial learning rate for Adam |
| `--hidden-sizes` | [50,50,50,50,50] | Neurons per hidden layer |
| `--checkpoint` | models/pinn_checkpoint.pth | Path to save checkpoints |
| `--device` | cuda/cpu | Compute device |
| `--resume` | False | Resume training from checkpoint |
| `--no-lbfgs` | False | Skip second-order optimization |

### 3.3 Output Files

Training produces the following artifacts:

- `models/pinn_checkpoint.pth` — optimizer state, epoch, loss (for resuming)
- `models/pinn_model.pth` — final model weights (for inference)
- `results/figures/loss_history.png` — training loss convergence plots
- `results/figures/velocity_profile.png` — PINN vs analytical parabolic profile
- `results/contours/u_x-velocity.png`, `v_y-velocity.png`, `pressure.png` — contour plots

---

## 4. Source Code Reference

### `src/data.py`

Handles stochastic sampling of:
- **Collocation points**: Uniform random across the domain interior (~80,000 points)
- **Boundary points**: Inlet (200 pts), upper/lower walls (800 pts each)
- **Outlet points**: Random sampling at x = 1.0

All points are returned as PyTorch tensors on the target device.

### `src/model.py`

#### `PINNModel(nn.Module)`

The core feedforward network:

```python
class PINNModel(nn.Module):
    def __init__(self, hidden_sizes, lb, ub, activation=nn.Tanh())
    def forward(self, x) -> torch.Tensor   # Returns (u, v, p)
    def predict(self, x) -> Tuple[u, v, p]
```

- Normalizes input coordinates to [0, 1] using min/max bounds
- Xavier uniform weight initialization, zero bias initialization
- Configurable hidden layer depth and width

#### `ResidualLoss`

Computes automatic-differentiation-based PDE residuals. Gradients are computed with `torch.autograd.grad` with `create_graph=True` to enable higher-order derivatives.

#### `PINNTrainer`

Orchestrates training:

- `train_adam(...)` — first-order stochastic optimization with periodic checkpointing
- `train_lbfgs(...)` — second-order quasi-Newton refinement
- `save_checkpoint(...)`, `load_checkpoint(...)` — robust training resumption

---

## 5. Troubleshooting

### CUDA out of memory
Reduce batch sizes in `data.py`: lower `N_b`, `N_w`, or `N_c`. Training was validated on T4 GPUs (Google Colab).

### Loss diverges or NaN
- Reduce learning rate: `python train.py --lr 1e-4`
- Check that `torch` and `numpy` versions meet minimum requirements
- Ensure GPU drivers are up to date

### LBFGS is slow
LBFGS requires full-batch gradient evaluation. For faster convergence, ensure Adam phase completes with loss < 0.01 before switching to LBFGS.

### Checkpoint loading fails
If `weights_only=True` raises an error (older PyTorch checkpoints), set `weights_only=False` in `PINNTrainer.load_checkpoint()`.

---

## 6. Extending the Project

Ideas for follow-up work:
- Unsteady (transient) Navier-Stokes with time-dependent boundary conditions
- Variable viscosity formulations (non-Newtonian fluids)
- Adaptive sampling for collocation points (e.g., residual-based refinement)
- Mixed-precision training for larger networks
- Comparison with CFD solvers (OpenFOAM, SU2) for validation

## 7. Workflow Summary

```
notebooks/  →  developmental notebook (archived)
     ↓
src/    →  modular, tested source code
models/ →  trained checkpoints & final weights
results/ →  plots and figures
docs/   →  this guide
```
