"""Training script for the PINN channel flow solver."""
import argparse
import torch
from pathlib import Path

from src.model import PINNModel, PINNTrainer, plot_loss_history, plot_velocity_profile, plot_contours
from src.data import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train PINN for 2D channel flow")
    parser.add_argument("--epochs-adam", type=int, default=5000)
    parser.add_argument("--epochs-lbfgs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[50, 50, 50, 50, 50])
    parser.add_argument("--checkpoint", type=str, default="models/pinn_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-lbfgs", action="store_true", help="Skip LBFGS refinement")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    
    domain = {"lb": [-1.0, -0.5], "ub": [1.0, 0.5]}
    device = torch.device(args.device)
    
    model = PINNModel(
        hidden_sizes=args.hidden_sizes,
        lb=domain["lb"],
        ub=domain["ub"],
        activation=torch.tanh,
    ).to(device)
    
    trainer = PINNTrainer(model, lr=args.lr, device=args.device)
    get_data = prepare_data(device=args.device)
    xy_col, xy_bc, uv_bc, xy_outlet = [d.to(device) for d in get_data()]
    
    start_epoch = 0
    if args.resume and Path(args.checkpoint).exists():
        start_epoch = trainer.load_checkpoint(args.checkpoint, map_location=args.device)
        print(f"Resumed from epoch {start_epoch}")
    
    print("Phase 1: Adam optimization")
    losses_total, losses_pde, losses_bc, losses_outlet = trainer.train_adam(
        xy_col=xy_col,
        xy_bc=xy_bc,
        uv_bc=uv_bc,
        xy_outlet=xy_outlet,
        epochs=args.epochs_adam,
        checkpoint_path=args.checkpoint,
    )
    
    plot_loss_history(
        losses_total, losses_pde, losses_bc, losses_outlet,
        save_path="results/figures/loss_history.png",
    )
    
    if not args.no_lbfgs:
        print("\nPhase 2: LBFGS refinement")
        trainer.train_lbfgs(
            xy_col=xy_col,
            xy_bc=xy_bc,
            uv_bc=uv_bc,
            xy_outlet=xy_outlet,
            epochs=args.epochs_lbfgs,
            checkpoint_path=args.checkpoint,
        )
    
    torch.save(model.state_dict(), "models/pinn_model.pth")
    print("Model saved to models/pinn_model.pth")
    
    print("\nGenerating plots...")
    plot_velocity_profile(model, device=args.device, save_path="results/figures/velocity_profile.png")
    plot_contours(model, device=args.device, save_dir="results/contours")
    print("Done.")


if __name__ == "__main__":
    main()
