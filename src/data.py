import numpy as np
import torch


def prepare_data(
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -0.5,
    y_max: float = 0.5,
    u_inlet: float = 1.0,
    N_b: int = 200,
    N_w: int = 800,
    N_c: int = 80000,
    device: str = "cpu",
):
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])

    def get_data():
        inlet_x = np.ones((N_b, 1)) * x_min
        inlet_y = np.random.uniform(y_min, y_max, (N_b, 1))
        inlet_u = np.ones((N_b, 1)) * u_inlet
        inlet_v = np.zeros((N_b, 1))
        inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
        inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

        xy_outlet = np.random.uniform([x_max, y_min], [x_max, y_max], (N_b, 2))
        outlet_v = np.zeros((N_b, 1))

        upwall_xy = np.random.uniform([x_min, y_max], [x_max, y_max], (N_w, 2))
        dnwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
        upwall_uv = np.zeros((N_w, 2))
        dnwall_uv = np.zeros((N_w, 2))

        xy_bnd = np.concatenate([inlet_xy, upwall_xy, dnwall_xy], axis=0)
        uv_bnd = np.concatenate([inlet_uv, upwall_uv, dnwall_uv], axis=0)

        xy_col = lb + (ub - lb) * np.random.rand(N_c, 2)
        xy_col = np.concatenate((xy_col, xy_bnd, xy_outlet), axis=0)

        xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
        uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
        xy_outlet = torch.tensor(xy_outlet, dtype=torch.float32).to(device)
        xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)

        return xy_col, xy_bnd, uv_bnd, xy_outlet

    return get_data
