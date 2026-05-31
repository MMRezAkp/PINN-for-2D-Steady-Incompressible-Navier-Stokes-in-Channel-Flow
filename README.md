# PINN Channel Flow Solver

Physics-Informed Neural Network (PINN) implementation for solving the 2D steady incompressible Navier-Stokes equations in a channel flow geometry. Network architecture supports a 5-hidden-layer feedforward neural network with tanh activations and Xavier initialization, combined with a two-phase optimization strategy (Adam + LBFGS).

## Directory Structure

```
2nd pahse/
├── src/
│   ├── __init__.py
│   ├── model.py          # PINN_Net, PINNTrainer, loss functions, plotting utilities
│   └── data.py           # Collocation / boundary / outlet point sampling
├── train.py              # Entry point: full training pipeline
├── notebooks/
│   └── "Pinn Project 2nd phase.ipynb"   # Original development notebook (archived)
├── models/
│   ├── pinn_checkpoint.pth        # Created during training
│   └── pinn_model.pth             # Created during training
├── results/
│   ├── figures/
│   │   └── [velocity_profile.png, loss_history.png]
│   └── contours/
│       └── [u_x-velocity.png, v_y-velocity.png, pressure.png, velocity_magnitude.png]
├── data/
├── docs/
│   └── guide.md
├── tests/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train from scratch:
```bash
python train.py
```

Resume from a saved checkpoint:
```bash
python train.py --resume
```

Skip LBFGS refinement (Adam only):
```bash
python train.py --no-lbfgs
```

Adjust model depth or learning rate:
```bash
python train.py --hidden-sizes 64 64 64 --lr 5e-4 --epochs-adam 10000
```

## Model Architecture

- **Input**: (x, y) spatial coordinates, normalized to [-1, 1]
- **Output**: (u, v, p) — streamwise velocity, transverse velocity, pressure
- **Hidden layers**: 5 × 50 units (configurable), tanh activation
- **Optimizers**: Adam (first 5000 epochs) → LBFGS (1000 epochs)

## Physics Formulation

The network minimizes a composite loss:

```
L_total = L_PDE + λ_bc * L_BC + λ_outlet * L_outlet
```

| Component | Description |
|-----------------|-----------------------------------------------|
| L_PDE | Navier-Stokes residual (momentum x, y + continuity) |
| L_BC | Dirichlet conditions: u=1 at inlet, no-slip walls |
| L_outlet | Zero-pressure outlet condition |

## Key Results

- Adam phase: ~5k epochs, final loss ~2.5e-2
- LBFGS phase: ~1k epochs, final loss ~3.0e-3
- Parabolic velocity profile recovered with good agreement vs analytical solution

## Contributing

This is an academic project. Feel free to fork for research extensions (variable viscosity, transient simulations, turbulence closures).

## License

MIT License — see [LICENSE](LICENSE) for details.
