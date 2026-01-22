"""
Utility functions for PINNs: Data generation, PDE operators, and visualization
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set style for better plots
plt.style.use("default")
sns.set_theme(style="whitegrid", palette="muted")


class PDEOperators:
    """Common PDE operators using torch.func for high-performance automatic differentiation"""

    @staticmethod
    def gradient(u_func: Callable, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of u with respect to x using torch.func.jacrev

        Args:
            u_func: Function that takes x and returns u
            x: Input tensor

        Returns:
            Gradient tensor of same shape as x
        """
        # Batch-compatible gradient using vmap and jacrev
        grad_func = torch.func.vmap(torch.func.jacrev(u_func))
        return grad_func(x).squeeze(1)

    @staticmethod
    def laplacian(u_func: Callable, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian (sum of second derivatives) using torch.func.hessian

        Args:
            u_func: Function that takes x and returns u
            x: Input tensor

        Returns:
            Laplacian of u
        """
        # Batch-compatible hessian
        hess_func = torch.func.vmap(torch.func.hessian(u_func))
        hessians = hess_func(x).squeeze(1).squeeze(2)
        
        # Laplacian is the trace of the Hessian (sum of diagonal elements)
        laplacian = torch.diagonal(hessians, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        return laplacian

    @staticmethod
    def divergence(v_func: Callable, x: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of vector field v using torch.func.jacrev

        Args:
            v_func: Function that takes x and returns vector v
            x: Spatial coordinates

        Returns:
            Divergence of v
        """
        # Batch-compatible jacobian
        jac_func = torch.func.vmap(torch.func.jacrev(v_func))
        jacobians = jac_func(x).squeeze(1) # shape: (batch, v_dim, x_dim)
        
        # Divergence is the trace of the Jacobian
        div = torch.diagonal(jacobians, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        return div


class DataGenerator:
    """Generate training data for PINNs"""

    @staticmethod
    def generate_collocation_points(
        bounds: Dict[str, Tuple[float, float]], n_points: int, method: str = "uniform"
    ) -> np.ndarray:
        """
        Generate collocation points for PDE residual

        Args:
            bounds: Dictionary of bounds for each dimension {'x': (x_min, x_max), 't': (t_min, t_max)}
            n_points: Number of collocation points
            method: Sampling method ('uniform', 'random', 'latin_hypercube')

        Returns:
            Array of collocation points
        """
        dim = len(bounds)

        if method == "uniform":
            # Create uniform grid
            n_per_dim = int(np.power(n_points, 1 / dim))
            grids = []
            for key, (low, high) in bounds.items():
                grids.append(np.linspace(low, high, n_per_dim))
            mesh = np.meshgrid(*grids)
            points = np.column_stack([m.ravel() for m in mesh])

        elif method == "random":
            # Random sampling
            points = np.zeros((n_points, dim))
            for i, (key, (low, high)) in enumerate(bounds.items()):
                points[:, i] = np.random.uniform(low, high, n_points)

        elif method == "latin_hypercube":
            # Latin hypercube sampling
            points = DataGenerator._latin_hypercube_sampling(n_points, bounds)

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return points.astype(np.float32)

    @staticmethod
    def _latin_hypercube_sampling(n_samples: int, bounds: Dict) -> np.ndarray:
        """Latin hypercube sampling for better space coverage"""
        dim = len(bounds)
        points = np.zeros((n_samples, dim))

        for i, (key, (low, high)) in enumerate(bounds.items()):
            # Create intervals
            intervals = np.linspace(low, high, n_samples + 1)

            # Random point in each interval
            for j in range(n_samples):
                points[j, i] = np.random.uniform(intervals[j], intervals[j + 1])

            # Shuffle this dimension
            np.random.shuffle(points[:, i])

        return points

    @staticmethod
    def generate_boundary_points(
        bounds: Dict[str, Tuple[float, float]],
        n_points_per_boundary: int,
        time_dependent: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate points on domain boundaries

        Args:
            bounds: Domain bounds
            n_points_per_boundary: Number of points per boundary
            time_dependent: If True, includes time dimension for boundaries

        Returns:
            Dictionary of boundary points
        """
        boundary_points = {}

        # Spatial boundaries
        if "x" in bounds:
            x_min, x_max = bounds["x"]

            if time_dependent and "t" in bounds:
                t_min, t_max = bounds["t"]
                t_points = np.random.uniform(t_min, t_max, n_points_per_boundary)

                # Left boundary
                boundary_points["left"] = np.column_stack(
                    [np.full(n_points_per_boundary, x_min), t_points]
                )

                # Right boundary
                boundary_points["right"] = np.column_stack(
                    [np.full(n_points_per_boundary, x_max), t_points]
                )
            else:
                # Static boundaries
                boundary_points["left"] = np.array([[x_min]])
                boundary_points["right"] = np.array([[x_max]])

        # Add y boundaries for 2D problems
        if "y" in bounds:
            y_min, y_max = bounds["y"]

            if "x" in bounds:
                x_min, x_max = bounds["x"]
                x_points = np.random.uniform(x_min, x_max, n_points_per_boundary)

                # Bottom boundary
                boundary_points["bottom"] = np.column_stack(
                    [x_points, np.full(n_points_per_boundary, y_min)]
                )

                # Top boundary
                boundary_points["top"] = np.column_stack(
                    [x_points, np.full(n_points_per_boundary, y_max)]
                )

        return {k: v.astype(np.float32) for k, v in boundary_points.items()}

    @staticmethod
    def generate_initial_points(
        bounds: Dict[str, Tuple[float, float]], n_points: int
    ) -> np.ndarray:
        """
        Generate points for initial conditions (at t=0)

        Args:
            bounds: Domain bounds
            n_points: Number of initial points

        Returns:
            Array of initial condition points
        """
        points = []

        # Spatial dimensions at t=0
        for key in ["x", "y", "z"]:
            if key in bounds:
                low, high = bounds[key]
                points.append(np.random.uniform(low, high, n_points))

        # Add t=0 if time-dependent
        if "t" in bounds:
            points.append(np.zeros(n_points))

        return np.column_stack(points).astype(np.float32)


class Visualizer:
    """Visualization utilities for PINNs"""

    @staticmethod
    def plot_solution_1d(
        x: np.ndarray,
        u_true: Optional[np.ndarray],
        u_pred: np.ndarray,
        title: str = "1D Solution",
        xlabel: str = "x",
        ylabel: str = "u(x)",
    ):
        """Plot 1D solution comparison"""
        plt.figure(figsize=(10, 6))

        if u_true is not None:
            plt.plot(x, u_true, "b-", label="True Solution", linewidth=2)
            plt.plot(x, u_pred, "r--", label="PINN Prediction", linewidth=2)
        else:
            plt.plot(x, u_pred, "b-", label="PINN Solution", linewidth=2)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_solution_2d(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        title: str = "2D Solution",
        xlabel: str = "x",
        ylabel: str = "y",
    ):
        """Plot 2D solution as contour/surface"""
        fig = plt.figure(figsize=(15, 5))

        # Contour plot
        ax1 = fig.add_subplot(131)
        contour = ax1.contourf(x, y, u, levels=20, cmap="viridis")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{title} - Contour")
        plt.colorbar(contour, ax=ax1)

        # 3D surface plot
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot_surface(x, y, u, cmap="viridis", alpha=0.9)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_zlabel("u")
        ax2.set_title(f"{title} - Surface")

        # Heatmap
        ax3 = fig.add_subplot(133)
        im = ax3.imshow(
            u,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.set_title(f"{title} - Heatmap")
        plt.colorbar(im, ax=ax3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_loss_history(loss_history: Dict[str, List[float]]):
        """Plot training loss history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Total loss
        axes[0, 0].semilogy(loss_history.get("total", []), label="Total Loss")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Physics loss
        if "physics" in loss_history and len(loss_history["physics"]) > 0:
            axes[0, 1].semilogy(
                loss_history["physics"], label="Physics Loss", color="orange"
            )
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].set_title("Physics Loss (PDE Residual)")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Boundary loss
        if "boundary" in loss_history and len(loss_history["boundary"]) > 0:
            axes[1, 0].semilogy(
                loss_history["boundary"], label="Boundary Loss", color="green"
            )
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("Boundary Condition Loss")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Data/Initial loss
        if "data" in loss_history and len(loss_history["data"]) > 0:
            axes[1, 1].semilogy(loss_history["data"], label="Data Loss", color="red")
        elif "initial" in loss_history and len(loss_history["initial"]) > 0:
            axes[1, 1].semilogy(
                loss_history["initial"], label="Initial Loss", color="red"
            )
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Data/Initial Condition Loss")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def animate_solution(
        x: np.ndarray,
        t: np.ndarray,
        u: np.ndarray,
        title: str = "Solution Evolution",
        save_path: Optional[str] = None,
    ):
        """
        Create animation of time-dependent solution

        Args:
            x: Spatial coordinates
            t: Time points
            u: Solution array (shape: [n_time, n_space])
            title: Animation title
            save_path: Path to save animation (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        (line,) = ax.plot([], [], "b-", linewidth=2)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(u.min(), u.max())
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text("")
            return line, time_text

        def animate(i):
            line.set_data(x, u[i])
            time_text.set_text(f"Time = {t[i]:.3f}")
            ax.set_title(f"{title} at t = {t[i]:.3f}")
            return line, time_text

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=len(t), interval=50, blit=True
        )

        if save_path:
            anim.save(save_path, writer="pillow")

        plt.show()
        return anim

    @staticmethod
    def plot_error_distribution(
        x: np.ndarray, error: np.ndarray, title: str = "Error Distribution"
    ):
        """Plot error distribution in space"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Spatial error distribution
        if x.shape[1] == 1:  # 1D
            axes[0].plot(x, np.abs(error), "r-", linewidth=2)
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("|Error|")
            axes[0].set_title(f"{title} - Spatial Distribution")
            axes[0].grid(True, alpha=0.3)
        else:  # 2D or higher
            scatter = axes[0].scatter(x[:, 0], x[:, 1], c=np.abs(error), cmap="hot")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].set_title(f"{title} - Spatial Distribution")
            plt.colorbar(scatter, ax=axes[0])

        # Error histogram
        axes[1].hist(error.ravel(), bins=50, density=True, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Error")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"{title} - Histogram")
        axes[1].grid(True, alpha=0.3)

        # Add statistics
        mean_error = np.mean(error)
        std_error = np.std(error)
        axes[1].axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.2e}"
        )
        axes[1].axvline(
            mean_error + std_error,
            color="orange",
            linestyle="--",
            label=f"Std: {std_error:.2e}",
        )
        axes[1].axvline(mean_error - std_error, color="orange", linestyle="--")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


class PDELibrary:
    """Library of common PDE definitions for testing"""

    def heat_equation_1d(
        x: torch.Tensor, u: Callable, model, alpha: float = 0.01
    ) -> torch.Tensor:
        """
        1D Heat equation: u_t = alpha * u_xx

        Args:
            x: Input tensor [x, t]
            u: Functional version of the network
            model: PINN model
            alpha: Thermal diffusivity

        Returns:
            PDE residual
        """
        # Compute derivatives using functional API
        du = PDEOperators.gradient(u, x)
        u_x = du[:, 0:1]
        u_t = du[:, 1:2]
        
        # Second derivative
        u_xx = PDEOperators.laplacian(u, x)
        
        # PDE residual
        residual = u_t - alpha * u_xx
        return residual

    @staticmethod
    def wave_equation_1d(
        x: torch.Tensor, u: Callable, model, c: float = 1.0
    ) -> torch.Tensor:
        """
        1D Wave equation: u_tt = c^2 * u_xx

        Args:
            x: Input tensor [x, t]
            u: Functional model
            model: PINN model
            c: Wave speed

        Returns:
            PDE residual
        """
        # For second order time derivative, we still need gradients
        def du_dt(xi):
            return torch.func.jacrev(u)(xi).squeeze(0)[1]
            
        u_tt = torch.func.vmap(torch.func.jacrev(du_dt))(x).squeeze(1)[:, 1:2]
        u_xx = PDEOperators.laplacian(u, x)

        # PDE residual
        residual = u_tt - c**2 * u_xx
        return residual

    @staticmethod
    def burgers_equation_1d(
        x: torch.Tensor, u: Callable, model, nu: float = 0.01
    ) -> torch.Tensor:
        """
        1D Burgers' equation: u_t + u * u_x = nu * u_xx
        """
        u_val = u(x) if not isinstance(u(x), torch.Tensor) else u(x) # Handle vmap output
        du = PDEOperators.gradient(u, x)
        u_x = du[:, 0:1]
        u_t = du[:, 1:2]
        u_xx = PDEOperators.laplacian(u, x)

        # PDE residual
        # Re-computing u at x for the non-linear term
        u_pred = torch.func.vmap(u)(x)
        residual = u_t + u_pred * u_x - nu * u_xx
        return residual

    @staticmethod
    def poisson_equation_2d(
        x: torch.Tensor, u: torch.Tensor, model, f: Callable
    ) -> torch.Tensor:
        """
        2D Poisson equation: -∇²u = f(x,y)

        Args:
            x: Input tensor [x, y]
            u: Output from network
            model: PINN model
            f: Source term function

        Returns:
            PDE residual
        """
        # Compute Laplacian
        laplacian_u = PDEOperators.laplacian(u, x)

        # Source term
        source = f(x[:, 0:1], x[:, 1:2])

        # PDE residual
        residual = -laplacian_u - source
        return residual

    @staticmethod
    def navier_stokes_2d(
        x: torch.Tensor, outputs: torch.Tensor, model, nu: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D Navier-Stokes equations (incompressible)

        Args:
            x: Input tensor [x, y, t]
            outputs: [u, v, p] velocities and pressure
            model: PINN model
            nu: Kinematic viscosity

        Returns:
            Tuple of (momentum_x_residual, momentum_y_residual, continuity_residual)
        """
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]

        # Compute derivatives
        u_x = PDEOperators.gradient(u, x)[:, 0:1]
        u_y = PDEOperators.gradient(u, x)[:, 1:2]
        u_t = PDEOperators.gradient(u, x)[:, 2:3]
        u_xx = PDEOperators.gradient(u_x, x)[:, 0:1]
        u_yy = PDEOperators.gradient(u_y, x)[:, 1:2]

        v_x = PDEOperators.gradient(v, x)[:, 0:1]
        v_y = PDEOperators.gradient(v, x)[:, 1:2]
        v_t = PDEOperators.gradient(v, x)[:, 2:3]
        v_xx = PDEOperators.gradient(v_x, x)[:, 0:1]
        v_yy = PDEOperators.gradient(v_y, x)[:, 1:2]

        p_x = PDEOperators.gradient(p, x)[:, 0:1]
        p_y = PDEOperators.gradient(p, x)[:, 1:2]

        # Momentum equations
        momentum_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        # Continuity equation
        continuity = u_x + v_y

        return momentum_x, momentum_y, continuity
