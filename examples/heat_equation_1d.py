"""
1D Heat Equation Example using Physics-Informed Neural Networks (PINNs)

This example demonstrates how to solve the 1D heat equation:
    ∂u/∂t = α * ∂²u/∂x²

with initial condition: u(x,0) = sin(πx)
and boundary conditions: u(0,t) = u(1,t) = 0

The analytical solution is: u(x,t) = sin(πx) * exp(-α*π²*t)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.pinn_model import PINN
from utils.pde_utils import DataGenerator, PDELibrary, PDEOperators, Visualizer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class HeatEquation1D:
    """
    Solver for the 1D Heat Equation using PINNs

    This class demonstrates the key concepts of PINNs:
    1. Encoding physics (PDE) as a loss function
    2. Enforcing boundary and initial conditions
    3. Training a neural network to satisfy all constraints
    """

    def __init__(
        self,
        alpha: float = 0.01,
        x_domain: Tuple[float, float] = (0.0, 1.0),
        t_domain: Tuple[float, float] = (0.0, 1.0),
        n_collocation: int = 10000,
        n_boundary: int = 100,
        n_initial: int = 100,
        n_test: int = 10000,
    ):
        """
        Initialize the Heat Equation solver

        Args:
            alpha: Thermal diffusivity coefficient
            x_domain: Spatial domain bounds
            t_domain: Time domain bounds
            n_collocation: Number of collocation points for PDE
            n_boundary: Number of boundary points
            n_initial: Number of initial condition points
            n_test: Number of test points for evaluation
        """
        self.alpha = alpha
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.n_test = n_test

        # Generate training data
        self.generate_training_data()

        # Initialize the PINN model
        self.model = self.initialize_model()

    def generate_training_data(self):
        """Generate all training data points"""

        # Define domain bounds
        bounds = {"x": self.x_domain, "t": self.t_domain}

        # 1. Collocation points (where PDE should be satisfied)
        # These are random points inside the domain where we enforce the PDE
        print(f"\nGenerating {self.n_collocation} collocation points...")
        self.x_collocation = DataGenerator.generate_collocation_points(
            bounds, self.n_collocation, method="random"
        )
        self.x_collocation_tensor = torch.tensor(
            self.x_collocation, dtype=torch.float32, requires_grad=True
        ).to(device)

        # 2. Boundary points (enforce boundary conditions)
        print(f"Generating {self.n_boundary} boundary points per boundary...")
        boundary_data = DataGenerator.generate_boundary_points(
            bounds, self.n_boundary, time_dependent=True
        )

        # Left boundary: u(0, t) = 0
        self.x_boundary_left = boundary_data["left"]
        self.u_boundary_left = np.zeros((self.n_boundary, 1))

        # Right boundary: u(1, t) = 0
        self.x_boundary_right = boundary_data["right"]
        self.u_boundary_right = np.zeros((self.n_boundary, 1))

        # Convert to tensors
        self.x_boundary_left_tensor = torch.tensor(
            self.x_boundary_left, dtype=torch.float32
        ).to(device)
        self.u_boundary_left_tensor = torch.tensor(
            self.u_boundary_left, dtype=torch.float32
        ).to(device)

        self.x_boundary_right_tensor = torch.tensor(
            self.x_boundary_right, dtype=torch.float32
        ).to(device)
        self.u_boundary_right_tensor = torch.tensor(
            self.u_boundary_right, dtype=torch.float32
        ).to(device)

        # 3. Initial condition points: u(x, 0) = sin(πx)
        print(f"Generating {self.n_initial} initial condition points...")
        x_init = np.random.uniform(self.x_domain[0], self.x_domain[1], self.n_initial)
        t_init = np.zeros(self.n_initial)
        self.x_initial = np.column_stack([x_init, t_init])

        # Initial condition values
        self.u_initial = np.sin(np.pi * x_init).reshape(-1, 1)

        # Convert to tensors
        self.x_initial_tensor = torch.tensor(self.x_initial, dtype=torch.float32).to(
            device
        )
        self.u_initial_tensor = torch.tensor(self.u_initial, dtype=torch.float32).to(
            device
        )

        # 4. Generate test data for evaluation
        print(f"Generating {self.n_test} test points...")
        self.generate_test_data()

    def generate_test_data(self):
        """Generate test data with analytical solution"""

        # Create a grid for testing
        x_test = np.linspace(self.x_domain[0], self.x_domain[1], 100)
        t_test = np.linspace(self.t_domain[0], self.t_domain[1], 100)
        X_test, T_test = np.meshgrid(x_test, t_test)

        self.x_test = np.column_stack([X_test.ravel(), T_test.ravel()])

        # Analytical solution: u(x,t) = sin(πx) * exp(-α*π²*t)
        self.u_test_true = np.sin(np.pi * X_test) * np.exp(
            -self.alpha * np.pi**2 * T_test
        )

        # Store grids for plotting
        self.X_test = X_test
        self.T_test = T_test

    def initialize_model(self) -> PINN:
        """
        Initialize the PINN model architecture

        Returns:
            Initialized PINN model
        """
        print("\nInitializing PINN model...")
        print(
            "Architecture: 2 inputs (x, t) -> [20, 20, 20] hidden layers -> 1 output (u)"
        )

        model = PINN(
            input_dim=2,  # (x, t)
            output_dim=1,  # u(x, t)
            hidden_layers=[20, 20, 20],  # Network architecture
            activation="tanh",  # Activation function
            use_batch_norm=False,  # No batch normalization
            dropout_rate=0.0,  # No dropout
            seed=42,  # For reproducibility
        ).to(device)

        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        return model

    def pde_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual for the heat equation

        The heat equation is: ∂u/∂t = α * ∂²u/∂x²
        The residual is: r = ∂u/∂t - α * ∂²u/∂x²

        Args:
            x: Input coordinates (x, t)
            u: Network output u(x, t)

        Returns:
            PDE residual
        """
        # Compute gradients
        u_grad = PDEOperators.gradient(u, x, order=1)
        u_x = u_grad[:, 0:1]  # ∂u/∂x
        u_t = u_grad[:, 1:2]  # ∂u/∂t

        # Second derivative
        u_xx = PDEOperators.gradient(u_x, x, order=1)[:, 0:1]  # ∂²u/∂x²

        # Heat equation residual
        residual = u_t - self.alpha * u_xx

        return residual

    def compute_loss(self, weights: Dict[str, float] = None) -> torch.Tensor:
        """
        Compute the total loss function

        The total loss consists of:
        1. PDE loss: How well the PDE is satisfied at collocation points
        2. Boundary loss: How well boundary conditions are satisfied
        3. Initial loss: How well initial conditions are satisfied

        Args:
            weights: Dictionary of loss weights

        Returns:
            Total weighted loss
        """
        if weights is None:
            weights = {"pde": 1.0, "boundary": 1.0, "initial": 1.0}

        # 1. Physics loss (PDE residual)
        u_collocation = self.model(self.x_collocation_tensor)
        pde_residual = self.pde_residual(self.x_collocation_tensor, u_collocation)
        loss_pde = torch.mean(pde_residual**2)

        # 2. Boundary condition losses
        u_left_pred = self.model(self.x_boundary_left_tensor)
        loss_boundary_left = torch.mean(
            (u_left_pred - self.u_boundary_left_tensor) ** 2
        )

        u_right_pred = self.model(self.x_boundary_right_tensor)
        loss_boundary_right = torch.mean(
            (u_right_pred - self.u_boundary_right_tensor) ** 2
        )

        loss_boundary = loss_boundary_left + loss_boundary_right

        # 3. Initial condition loss
        u_initial_pred = self.model(self.x_initial_tensor)
        loss_initial = torch.mean((u_initial_pred - self.u_initial_tensor) ** 2)

        # Total weighted loss
        total_loss = (
            weights["pde"] * loss_pde
            + weights["boundary"] * loss_boundary
            + weights["initial"] * loss_initial
        )

        # Store individual losses for monitoring
        self.current_losses = {
            "pde": loss_pde.item(),
            "boundary": loss_boundary.item(),
            "initial": loss_initial.item(),
            "total": total_loss.item(),
        }

        return total_loss

    def train(
        self,
        n_epochs: int = 10000,
        learning_rate: float = 0.001,
        print_every: int = 1000,
    ):
        """
        Train the PINN model

        Args:
            n_epochs: Number of training epochs
            learning_rate: Initial learning rate
            print_every: Print frequency
        """
        print(f"\n{'=' * 60}")
        print("Starting PINN Training")
        print(f"{'=' * 60}")
        print(f"Epochs: {n_epochs}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

        # Initialize optimizer (Adam with L-BFGS for fine-tuning)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler (reduce LR when loss plateaus)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1000, verbose=True
        )

        # Training history
        history = {
            "loss_total": [],
            "loss_pde": [],
            "loss_boundary": [],
            "loss_initial": [],
        }

        # Training loop
        self.model.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Compute loss
            loss = self.compute_loss()

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Update learning rate
            scheduler.step(loss)

            # Store history
            history["loss_total"].append(self.current_losses["total"])
            history["loss_pde"].append(self.current_losses["pde"])
            history["loss_boundary"].append(self.current_losses["boundary"])
            history["loss_initial"].append(self.current_losses["initial"])

            # Print progress
            if (epoch + 1) % print_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1}/{n_epochs}:")
                print(f"  Total Loss: {self.current_losses['total']:.6e}")
                print(f"  PDE Loss: {self.current_losses['pde']:.6e}")
                print(f"  Boundary Loss: {self.current_losses['boundary']:.6e}")
                print(f"  Initial Loss: {self.current_losses['initial']:.6e}")
                print(f"  Learning Rate: {current_lr:.6e}")
                print("-" * 40)

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"Final Total Loss: {self.current_losses['total']:.6e}")
        print(f"{'=' * 60}\n")

        self.history = history

    def evaluate(self):
        """Evaluate the trained model"""
        print("\nEvaluating model...")

        self.model.eval()

        with torch.no_grad():
            # Convert test data to tensor
            x_test_tensor = torch.tensor(self.x_test, dtype=torch.float32).to(device)

            # Predict
            u_pred = self.model(x_test_tensor).cpu().numpy()
            u_pred = u_pred.reshape(self.X_test.shape)

        # Compute error
        error = np.abs(u_pred - self.u_test_true)
        relative_error = error / (np.abs(self.u_test_true) + 1e-8)

        print(f"Mean Absolute Error: {np.mean(error):.6e}")
        print(f"Max Absolute Error: {np.max(error):.6e}")
        print(f"Mean Relative Error: {np.mean(relative_error):.6e}")

        self.u_pred = u_pred
        self.error = error

    def plot_results(self):
        """Visualize the results"""

        fig = plt.figure(figsize=(15, 10))

        # 1. True solution
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.contourf(
            self.X_test, self.T_test, self.u_test_true, levels=20, cmap="viridis"
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("t")
        ax1.set_title("True Solution")
        plt.colorbar(im1, ax=ax1)

        # 2. PINN prediction
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.contourf(
            self.X_test, self.T_test, self.u_pred, levels=20, cmap="viridis"
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        ax2.set_title("PINN Prediction")
        plt.colorbar(im2, ax=ax2)

        # 3. Absolute error
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.contourf(self.X_test, self.T_test, self.error, levels=20, cmap="hot")
        ax3.set_xlabel("x")
        ax3.set_ylabel("t")
        ax3.set_title("Absolute Error")
        plt.colorbar(im3, ax=ax3)

        # 4. Solution at different times
        ax4 = fig.add_subplot(2, 3, 4)
        time_indices = [0, 25, 50, 75, 99]
        for idx in time_indices:
            t_val = self.T_test[idx, 0]
            ax4.plot(
                self.X_test[idx, :],
                self.u_test_true[idx, :],
                label=f"t={t_val:.2f} (True)",
                linestyle="-",
                alpha=0.7,
            )
            ax4.plot(
                self.X_test[idx, :],
                self.u_pred[idx, :],
                label=f"t={t_val:.2f} (PINN)",
                linestyle="--",
                alpha=0.7,
            )
        ax4.set_xlabel("x")
        ax4.set_ylabel("u(x,t)")
        ax4.set_title("Solution at Different Times")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(True, alpha=0.3)

        # 5. Loss history
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.semilogy(self.history["loss_total"], label="Total", linewidth=2)
        ax5.semilogy(self.history["loss_pde"], label="PDE", alpha=0.7)
        ax5.semilogy(self.history["loss_boundary"], label="Boundary", alpha=0.7)
        ax5.semilogy(self.history["loss_initial"], label="Initial", alpha=0.7)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Loss")
        ax5.set_title("Training Loss History")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Error distribution
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(
            self.error.ravel(), bins=50, density=True, alpha=0.7, edgecolor="black"
        )
        ax6.set_xlabel("Absolute Error")
        ax6.set_ylabel("Density")
        ax6.set_title("Error Distribution")
        mean_error = np.mean(self.error)
        ax6.axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.2e}"
        )
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating PINN for heat equation"""

    print("\n" + "=" * 60)
    print("PHYSICS-INFORMED NEURAL NETWORKS (PINNs)")
    print("1D Heat Equation Example")
    print("=" * 60)

    print("\n📚 LEARNING OBJECTIVES:")
    print("1. Understand how PINNs encode physics into neural networks")
    print("2. Learn about the different loss components")
    print("3. See how PINNs can solve PDEs without labeled data")

    print("\n🔬 PROBLEM SETUP:")
    print("PDE: ∂u/∂t = α * ∂²u/∂x²  (Heat Equation)")
    print("Domain: x ∈ [0, 1], t ∈ [0, 1]")
    print("Initial: u(x,0) = sin(πx)")
    print("Boundary: u(0,t) = u(1,t) = 0")
    print("Thermal diffusivity: α = 0.01")

    # Create solver
    solver = HeatEquation1D(
        alpha=0.01,  # Thermal diffusivity
        x_domain=(0.0, 1.0),  # Spatial domain
        t_domain=(0.0, 1.0),  # Time domain
        n_collocation=10000,  # Physics points
        n_boundary=200,  # Boundary points
        n_initial=200,  # Initial points
        n_test=10000,  # Test points
    )

    # Train the model
    solver.train(n_epochs=5000, learning_rate=0.001, print_every=500)

    # Evaluate
    solver.evaluate()

    # Visualize results
    print("\n📊 Visualizing results...")
    solver.plot_results()

    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS:")
    print("=" * 60)
    print("\n1. NO LABELED DATA NEEDED:")
    print("   - PINNs only need the PDE and boundary/initial conditions")
    print("   - The physics acts as the supervision signal")

    print("\n2. LOSS FUNCTION COMPONENTS:")
    print("   - PDE Loss: Ensures the equation is satisfied inside the domain")
    print("   - Boundary Loss: Enforces boundary conditions")
    print("   - Initial Loss: Matches initial conditions")

    print("\n3. ADVANTAGES OF PINNs:")
    print("   - Can handle complex geometries")
    print("   - Mesh-free (no discretization needed)")
    print("   - Can solve inverse problems")
    print("   - Differentiable solution everywhere")

    print("\n4. TRAINING CONSIDERATIONS:")
    print("   - Balance between different loss terms is important")
    print("   - More collocation points → better physics enforcement")
    print("   - Network architecture affects solution quality")

    print("\n" + "=" * 60)
    print("Tutorial Complete! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
