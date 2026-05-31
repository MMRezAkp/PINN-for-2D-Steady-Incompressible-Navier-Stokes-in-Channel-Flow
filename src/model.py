import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PINNModel(nn.Module):
    """Physics-Informed Neural Network for 2D steady Navier-Stokes equations.
    
    Predicts velocity components (u, v) and pressure (p) from spatial coordinates (x, y).
    The network uses tanh activations and Xavier uniform initialization.
    """
    
    def __init__(
        self,
        hidden_sizes: List[int],
        lb: np.ndarray,
        ub: np.ndarray,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 3  # u, v, p
        self.hidden_sizes = hidden_sizes
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        self.activation = activation
        
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_sizes[0]))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
        
        layers.append(nn.Linear(hidden_sizes[-1], self.output_dim))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.lb.to(x.device)) / (self.ub.to(x.device) - self.lb.to(x.device))
        out = self.network(x_norm)
        return out
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(x)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p


class ResidualLoss:
    """Computes PDE residual loss for 2D steady incompressible Navier-Stokes.
    
    Governing equations (non-dimensional form):
        - u*du/dx + v*du/dy + dp/dx - nu*(d2u/dx2 + d2u/dy2) = 0
        - u*dv/dx + v*dv/dy + dp/dy - nu*(d2v/dx2 + d2v/dy2) = 0
        - du/dx + dv/dy = 0  (continuity)
    """
    
    def __init__(self, rho: float = 1.0, mu: float = 1.0):
        self.rho = rho
        self.mu = mu
        self.nu = mu / rho
    
    def __call__(self, model: PINNModel, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x.requires_grad_(True)
        u, v, p = model.predict(x)
        
        grads_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        grads_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        grads_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        
        u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
        v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
        p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]
        
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 1:2]
        
        f0 = u * u_x + v * u_y + (p_x / self.rho) - self.nu * (u_xx + u_yy)
        f1 = u * v_x + v * v_y + (p_y / self.rho) - self.nu * (v_xx + v_yy)
        f2 = u_x + v_y
        
        mse_f0 = torch.mean(f0 ** 2)
        mse_f1 = torch.mean(f1 ** 2)
        mse_f2 = torch.mean(f2 ** 2)
        
        return mse_f0, mse_f1, mse_f2


class BoundaryLoss:
    """Computes boundary condition losses."""
    
    @staticmethod
    def dirichlet(model: PINNModel, x_bc: torch.Tensor, u_bc: torch.Tensor) -> torch.Tensor:
        u_pred, v_pred, _ = model.predict(x_bc)
        mse_u = torch.mean((u_pred - u_bc[:, 0:1]) ** 2)
        mse_v = torch.mean((v_pred - u_bc[:, 1:2]) ** 2)
        return mse_u + mse_v
    
    @staticmethod
    def outlet_pressure(model: PINNModel, x_outlet: torch.Tensor) -> torch.Tensor:
        _, _, p_pred = model.predict(x_outlet)
        return torch.mean(p_pred ** 2)


class PINNTrainer:
    """Handles training, checkpointing, and visualization for the PINN model."""
    
    def __init__(self, model: PINNModel, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.residual_loss = ResidualLoss()
    
    def train_adam(
        self,
        xy_col: torch.Tensor,
        xy_bc: torch.Tensor,
        uv_bc: torch.Tensor,
        xy_outlet: torch.Tensor,
        epochs: int,
        checkpoint_path: str,
        lambda_bc: float = 1.0,
        lambda_outlet: float = 1.0,
        log_interval: int = 100,
        threshold: float = 1e-20,
    ) -> List[float]:
        losses_total, losses_pde, losses_bc, losses_outlet = [], [], [], []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            mse_pde = sum(self.residual_loss(self.model, xy_col))
            mse_bc = BoundaryLoss.dirichlet(self.model, xy_bc, uv_bc)
            mse_outlet = BoundaryLoss.outlet_pressure(self.model, xy_outlet)
            
            loss = mse_pde + lambda_bc * mse_bc + lambda_outlet * mse_outlet
            loss.backward()
            self.optimizer.step()
            
            losses_total.append(loss.item())
            losses_pde.append(mse_pde.item())
            losses_bc.append(mse_bc.item())
            losses_outlet.append(mse_outlet.item())
            
            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:5d} | Total: {loss.item():.6e} | "
                    f"PDE: {mse_pde.item():.6e} | BC: {mse_bc.item():.6e} | "
                    f"Outlet: {mse_outlet.item():.6e}"
                )
                self.save_checkpoint(epoch, checkpoint_path)
            
            if loss.item() < threshold:
                print(f"Early stopping at epoch {epoch} with loss {loss.item():.6e}")
                break
        
        return losses_total, losses_pde, losses_bc, losses_outlet
    
    def train_lbfgs(
        self,
        xy_col: torch.Tensor,
        xy_bc: torch.Tensor,
        uv_bc: torch.Tensor,
        xy_outlet: torch.Tensor,
        epochs: int,
        checkpoint_path: str,
        max_iter: int = 20,
    ) -> List[float]:
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=max_iter
        )
        losses = []
        
        for epoch in range(epochs):
            def closure():
                self.optimizer.zero_grad()
                mse_pde = sum(self.residual_loss(self.model, xy_col))
                mse_bc = BoundaryLoss.dirichlet(self.model, xy_bc, uv_bc)
                mse_outlet = BoundaryLoss.outlet_pressure(self.model, xy_outlet)
                loss = mse_pde + mse_bc + mse_outlet
                loss.backward()
                return loss
            
            loss = self.optimizer.step(closure)
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"LBFGS Epoch {epoch:4d} | Loss: {loss.item():.6e}")
                self.save_checkpoint(epoch, checkpoint_path)
        
        return losses
    
    def save_checkpoint(self, epoch: int, path: str):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> int:
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]


def plot_loss_history(
    losses_total: List[float],
    losses_pde: List[float],
    losses_bc: List[float],
    losses_outlet: List[float],
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(losses_total)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(losses_pde)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("PDE Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(losses_bc)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Boundary Condition Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(losses_outlet)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Outlet Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_velocity_profile(
    model: PINNModel,
    device: str,
    L: float = 2.0,
    D: float = 1.0,
    u_in: float = 1.0,
    save_path: Optional[str] = None,
):
    y = np.linspace(-D / 2, D / 2, 1000).reshape(-1, 1).astype(np.float32)
    analytic = u_in * 1.5 * (1 - (y / (D / 2)) ** 2)
    
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    locations = [
        (torch.zeros_like(y_tensor), y_tensor, "PINN x=0"),
        (torch.ones_like(y_tensor), y_tensor, "PINN x=1"),
        (-torch.ones_like(y_tensor), y_tensor, "PINN x=-1"),
    ]
    
    plt.figure(figsize=(10, 8))
    plt.plot(y, analytic, "k--", label="Analytic (Parabolic)", linewidth=2)
    
    for x_val, y_val, label in locations:
        coords = torch.cat([x_val, y_val], dim=1).to(device)
        with torch.no_grad():
            pred = model(coords).cpu().numpy()[:, 0]
        plt.plot(y, pred, linewidth=2, label=label)
    
    plt.xlabel("y")
    plt.ylabel("u (x-velocity)")
    plt.title("Velocity Profile: PINN vs Analytical Solution")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_contours(
    model: PINNModel,
    device: str,
    L: float = 2.0,
    D: float = 1.0,
    resolution: int = 500,
    save_dir: Optional[str] = None,
):
    x = np.linspace(-L / 2, L / 2, resolution)
    y = np.linspace(-D / 2, D / 2, resolution)
    X, Y = np.meshgrid(x, y)
    
    coords = torch.tensor(
        np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32
    ).to(device)
    
    with torch.no_grad():
        result = model(coords).cpu().numpy()
    
    components = ["u (x-velocity)", "v (y-velocity)", "Pressure"]
    colormaps = ["jet", "jet", "coolwarm"]
    clims = [[0, 1.5], [-0.3, 0.3], [0, 35]]
    
    for idx, (name, cmap, clim) in enumerate(zip(components, colormaps, clims)):
        Z = result[:, idx].reshape(resolution, resolution)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        contour = ax.contourf(X, Y, Z, levels=60, cmap=cmap)
        contour.set_clim(clim)
        plt.colorbar(contour, ax=ax, label=name)
        ax.set_title(f"Contour: {name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-L / 2, L / 2)
        ax.set_ylim(-D / 2, D / 2)
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_dir}/{name.replace(' ', '_').lower()}.png", dpi=150)
        plt.show()
