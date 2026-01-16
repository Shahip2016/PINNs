"""
Inverse Problem Example using Physics-Informed Neural Networks (PINNs)

This example demonstrates how PINNs can solve inverse problems:
- Given: Some observations of a system
- Unknown: Parameters in the governing PDE
- Goal: Discover the unknown parameters while solving for the full solution

Example: Discovering thermal diffusivity in heat equation from partial observations
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.pinn_model import PINN
from utils.pde_utils import PDEOperators

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class InverseProblemSolver:
    """
    Solver for inverse problems using PINNs

    This demonstrates how PINNs can:
    1. Discover unknown parameters in PDEs
    2. Reconstruct full solutions from partial observations
    3. Handle noisy measurements
    """

    def __init__(
        self,
        true_alpha: float = 0.5,  # True parameter (unknown to the model)
        x_domain: Tuple[float, float] = (0.0, 1.0),
        t_domain: Tuple[float, float] = (0.0, 1.0),
        n_observations: int = 100,
        n_collocation: int = 5000,
        noise_level: float = 0.01,
    ):
        """
        Initialize inverse problem solver

        Args:
            true_alpha: True diffusivity (to generate synthetic data)
            x_domain: Spatial domain
            t_domain: Time domain
            n_observations: Number of observation points
            n_collocation: Number of collocation points
            noise_level: Noise level for observations
        """
        self.true_alpha = true_alpha
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.n_observations = n_observations
        self.n_collocation = n_collocation
        self.noise_level = noise_level

        # Initialize learnable parameter (our guess)
        # Start with a wrong initial guess to show learning
        self.alpha_init = 0.1  # Initial guess (different from true value)
        self.alpha = nn.Parameter(
            torch.tensor([self.alpha_init], dtype=torch.float32).to(device)
        )

        print(f"\n{'=' * 60}")
        print("INVERSE PROBLEM SETUP")
        print(f"{'=' * 60}")
        print(f"True parameter (unknown): α = {self.true_alpha}")
        print(f"Initial guess: α = {self.alpha_init}")
        print(f"Noise level: {noise_level * 100:.1f}%")
        print(f"Number of observations: {n_observations}")

        # Generate data
        self.generate_synthetic_data()
        self.generate_training_points()

        # Initialize PINN model
        self.model = self.initialize_model()

    def generate_synthetic_data(self):
        """Generate synthetic observations from the true solution"""

        print("\n📊 Generating synthetic observations...")

        # Random observation points in space and time
        x_obs = np.random.uniform(
            self.x_domain[0], self.x_domain[1], self.n_observations
        )
        t_obs = np.random.uniform(
            self.t_domain[0], self.t_domain[1], self.n_observations
        )

        # True solution: u(x,t) = sin(πx) * exp(-α*π²*t)
        u_true = np.sin(np.pi * x_obs) * np.exp(-self.true_alpha * np.pi**2 * t_obs)

        # Add noise to simulate real measurements
        noise = np.random.normal(0, self.noise_level, u_true.shape)
        u_obs = u_true + noise * np.abs(u_true)  # Proportional noise

        # Store observation data
        self.x_obs = np.column_stack([x_obs, t_obs])
        self.u_obs = u_obs.reshape(-1, 1)

        # Convert to tensors
        self.x_obs_tensor = torch.tensor(self.x_obs, dtype=torch.float32).to(device)
        self.u_obs_tensor = torch.tensor(self.u_obs, dtype=torch.float32).to(device)

        print(f"Generated {self.n_observations} noisy observations")
        print(f"Signal-to-Noise Ratio: {np.std(u_true) / np.std(noise):.2f}")

    def generate_training_points(self):
        """Generate collocation points for physics loss"""

        # Collocation points (where PDE should be satisfied)
        x_col = np.random.uniform(
            self.x_domain[0], self.x_domain[1], self.n_collocation
        )
        t_col = np.random.uniform(
            self.t_domain[0], self.t_domain[1], self.n_collocation
        )
        self.x_collocation = np.column_stack([x_col, t_col])

        self.x_collocation_tensor = torch.tensor(
            self.x_collocation, dtype=torch.float32, requires_grad=True
        ).to(device)

        # Initial condition points
        n_initial = 100
        x_init = np.linspace(self.x_domain[0], self.x_domain[1], n_initial)
        t_init = np.zeros(n_initial)
        self.x_initial = np.column_stack([x_init, t_init])
        self.u_initial = np.sin(np.pi * x_init).reshape(-1, 1)

        self.x_initial_tensor = torch.tensor(self.x_initial, dtype=torch.float32).to(
            device
        )
        self.u_initial_tensor = torch.tensor(self.u_initial, dtype=torch.float32).to(
            device
        )

        # Boundary points
        n_boundary = 100
        t_boundary = np.random.uniform(self.t_domain[0], self.t_domain[1], n_boundary)

        # Left boundary (x=0)
        self.x_boundary_left = np.column_stack([np.zeros(n_boundary), t_boundary])
        self.u_boundary_left = np.zeros((n_boundary, 1))

        # Right boundary (x=1)
        self.x_boundary_right = np.column_stack([np.ones(n_boundary), t_boundary])
        self.u_boundary_right = np.zeros((n_boundary, 1))

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

    def initialize_model(self) -> PINN:
        """Initialize the PINN model"""

        model = PINN(
            input_dim=2,  # (x, t)
            output_dim=1,  # u(x, t)
            hidden_layers=[32, 32, 32, 32],  # Deeper network for inverse problem
            activation="tanh",
            use_batch_norm=False,
            dropout_rate=0.0,
            seed=42,
        ).to(device)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Unknown PDE parameters: 1 (thermal diffusivity α)")

        return model

    def compute_loss(
        self, weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for inverse problem

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if weights is None:
            weights = {
                "data": 1.0,  # Weight for observation fitting
                "physics": 0.1,  # Weight for PDE residual
                "initial": 1.0,  # Weight for initial condition
                "boundary": 1.0,  # Weight for boundary conditions
            }

        losses = {}

        # 1. Data loss (fitting observations)
        u_obs_pred = self.model(self.x_obs_tensor)
        loss_data = torch.mean((u_obs_pred - self.u_obs_tensor) ** 2)
        losses["data"] = loss_data

        # 2. Physics loss (PDE residual with unknown parameter)
        u_col = self.model(self.x_collocation_tensor)

        # Compute derivatives
        u_grad = PDEOperators.gradient(u_col, self.x_collocation_tensor)
        u_x = u_grad[:, 0:1]
        u_t = u_grad[:, 1:2]
        u_xx = PDEOperators.gradient(u_x, self.x_collocation_tensor)[:, 0:1]

        # Heat equation residual with learnable parameter
        pde_residual = u_t - self.alpha * u_xx
        loss_physics = torch.mean(pde_residual**2)
        losses["physics"] = loss_physics

        # 3. Initial condition loss
        u_init_pred = self.model(self.x_initial_tensor)
        loss_initial = torch.mean((u_init_pred - self.u_initial_tensor) ** 2)
        losses["initial"] = loss_initial

        # 4. Boundary condition losses
        u_left_pred = self.model(self.x_boundary_left_tensor)
        loss_boundary_left = torch.mean(
            (u_left_pred - self.u_boundary_left_tensor) ** 2
        )

        u_right_pred = self.model(self.x_boundary_right_tensor)
        loss_boundary_right = torch.mean(
            (u_right_pred - self.u_boundary_right_tensor) ** 2
        )

        loss_boundary = loss_boundary_left + loss_boundary_right
        losses["boundary"] = loss_boundary

        # Total weighted loss
        total_loss = sum(weights.get(key, 1.0) * loss for key, loss in losses.items())

        # Store for monitoring
        self.current_losses = {key: loss.item() for key, loss in losses.items()}
        self.current_losses["total"] = total_loss.item()

        return total_loss, losses

    def train(
        self,
        n_epochs: int = 10000,
        learning_rate: float = 0.001,
        param_lr: float = 0.01,
        print_every: int = 500,
    ):
        """
        Train the model to discover unknown parameters

        Args:
            n_epochs: Number of training epochs
            learning_rate: Learning rate for neural network
            param_lr: Learning rate for unknown parameter
            print_every: Print frequency
        """
        print(f"\n{'=' * 60}")
        print("TRAINING INVERSE PROBLEM")
        print(f"{'=' * 60}")

        # Separate optimizers for network and parameter
        network_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        param_optimizer = optim.Adam([self.alpha], lr=param_lr)

        # Learning rate schedulers
        network_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            network_optimizer, mode="min", factor=0.5, patience=1000, verbose=False
        )
        param_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            param_optimizer, mode="min", factor=0.5, patience=1000, verbose=False
        )

        # Training history
        self.history = {
            "loss_total": [],
            "loss_data": [],
            "loss_physics": [],
            "loss_initial": [],
            "loss_boundary": [],
            "alpha": [],
            "alpha_error": [],
        }

        # Training loop
        self.model.train()

        for epoch in range(n_epochs):
            # Zero gradients
            network_optimizer.zero_grad()
            param_optimizer.zero_grad()

            # Compute loss
            loss, _ = self.compute_loss()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([self.alpha], max_norm=1.0)

            # Update weights
            network_optimizer.step()
            param_optimizer.step()

            # Ensure parameter stays positive
            with torch.no_grad():
                self.alpha.data = torch.clamp(self.alpha.data, min=0.001)

            # Update schedulers
            network_scheduler.step(loss)
            param_scheduler.step(loss)

            # Store history
            current_alpha = self.alpha.item()
            alpha_error = abs(current_alpha - self.true_alpha)

            self.history["loss_total"].append(self.current_losses["total"])
            self.history["loss_data"].append(self.current_losses["data"])
            self.history["loss_physics"].append(self.current_losses["physics"])
            self.history["loss_initial"].append(self.current_losses["initial"])
            self.history["loss_boundary"].append(self.current_losses["boundary"])
            self.history["alpha"].append(current_alpha)
            self.history["alpha_error"].append(alpha_error)

            # Print progress
            if (epoch + 1) % print_every == 0:
                relative_error = alpha_error / self.true_alpha * 100
                print(f"\nEpoch {epoch + 1}/{n_epochs}:")
                print(f"  Total Loss: {self.current_losses['total']:.6e}")
                print(f"  Data Loss: {self.current_losses['data']:.6e}")
                print(f"  Physics Loss: {self.current_losses['physics']:.6e}")
                print(
                    f"  Discovered α: {current_alpha:.4f} (True: {self.true_alpha:.4f})"
                )
                print(f"  Parameter Error: {relative_error:.2f}%")

        self.discovered_alpha = self.alpha.item()

        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"True α: {self.true_alpha:.4f}")
        print(f"Discovered α: {self.discovered_alpha:.4f}")
        print(
            f"Relative Error: {abs(self.discovered_alpha - self.true_alpha) / self.true_alpha * 100:.2f}%"
        )

    def evaluate(self):
        """Evaluate the solution with discovered parameter"""

        print("\n📈 Evaluating solution with discovered parameter...")

        # Create test grid
        x_test = np.linspace(self.x_domain[0], self.x_domain[1], 100)
        t_test = np.linspace(self.t_domain[0], self.t_domain[1], 100)
        X_test, T_test = np.meshgrid(x_test, t_test)

        x_test_points = np.column_stack([X_test.ravel(), T_test.ravel()])

        # True solution with true parameter
        u_true = np.sin(np.pi * X_test) * np.exp(-self.true_alpha * np.pi**2 * T_test)

        # PINN prediction
        self.model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test_points, dtype=torch.float32).to(device)
            u_pred = self.model(x_test_tensor).cpu().numpy()
            u_pred = u_pred.reshape(X_test.shape)

        # Solution with discovered parameter (analytical formula)
        u_discovered = np.sin(np.pi * X_test) * np.exp(
            -self.discovered_alpha * np.pi**2 * T_test
        )

        # Compute errors
        error_pinn = np.abs(u_pred - u_true)
        error_param = np.abs(u_discovered - u_true)

        print(f"Mean Absolute Error (PINN): {np.mean(error_pinn):.6e}")
        print(f"Mean Absolute Error (Discovered Param): {np.mean(error_param):.6e}")

        self.X_test = X_test
        self.T_test = T_test
        self.u_true = u_true
        self.u_pred = u_pred
        self.u_discovered = u_discovered

    def plot_results(self):
        """Visualize inverse problem results"""

        fig = plt.figure(figsize=(16, 12))

        # 1. Parameter evolution
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.plot(self.history["alpha"], label="Discovered α", linewidth=2)
        ax1.axhline(
            y=self.true_alpha, color="r", linestyle="--", label="True α", linewidth=2
        )
        ax1.axhline(
            y=self.alpha_init,
            color="g",
            linestyle=":",
            label="Initial guess",
            alpha=0.5,
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("α value")
        ax1.set_title("Parameter Discovery Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parameter error
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.semilogy(self.history["alpha_error"], linewidth=2, color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("|α - α_true|")
        ax2.set_title("Parameter Error Evolution")
        ax2.grid(True, alpha=0.3)

        # 3. Loss history
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.semilogy(self.history["loss_total"], label="Total", linewidth=2)
        ax3.semilogy(self.history["loss_data"], label="Data", alpha=0.7)
        ax3.semilogy(self.history["loss_physics"], label="Physics", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss History")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Observation points
        ax4 = fig.add_subplot(3, 3, 4)
        scatter = ax4.scatter(
            self.x_obs[:, 0],
            self.x_obs[:, 1],
            c=self.u_obs.ravel(),
            cmap="viridis",
            s=50,
        )
        ax4.set_xlabel("x")
        ax4.set_ylabel("t")
        ax4.set_title(f"Observation Points (n={self.n_observations})")
        plt.colorbar(scatter, ax=ax4)

        # 5. True solution
        ax5 = fig.add_subplot(3, 3, 5)
        im5 = ax5.contourf(
            self.X_test, self.T_test, self.u_true, levels=20, cmap="viridis"
        )
        ax5.set_xlabel("x")
        ax5.set_ylabel("t")
        ax5.set_title(f"True Solution (α={self.true_alpha:.3f})")
        plt.colorbar(im5, ax=ax5)

        # 6. PINN reconstruction
        ax6 = fig.add_subplot(3, 3, 6)
        im6 = ax6.contourf(
            self.X_test, self.T_test, self.u_pred, levels=20, cmap="viridis"
        )
        ax6.set_xlabel("x")
        ax6.set_ylabel("t")
        ax6.set_title(f"PINN Reconstruction (α={self.discovered_alpha:.3f})")
        plt.colorbar(im6, ax=ax6)

        # 7. Solution at specific times
        ax7 = fig.add_subplot(3, 3, 7)
        time_indices = [0, 30, 60, 90]
        for idx in time_indices:
            t_val = self.T_test[idx, 0]
            ax7.plot(
                self.X_test[idx, :],
                self.u_true[idx, :],
                label=f"t={t_val:.2f} (True)",
                linestyle="-",
                alpha=0.7,
            )
            ax7.plot(
                self.X_test[idx, :],
                self.u_pred[idx, :],
                label=f"t={t_val:.2f} (PINN)",
                linestyle="--",
                alpha=0.7,
            )

        # Add observation points at t≈0
        obs_at_t0 = self.x_obs[self.x_obs[:, 1] < 0.1]
        if len(obs_at_t0) > 0:
            u_obs_t0 = self.u_obs[self.x_obs[:, 1] < 0.1]
            ax7.scatter(
                obs_at_t0[:, 0],
                u_obs_t0,
                color="red",
                s=20,
                alpha=0.5,
                label="Observations",
            )

        ax7.set_xlabel("x")
        ax7.set_ylabel("u(x,t)")
        ax7.set_title("Solution Profiles")
        ax7.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax7.grid(True, alpha=0.3)

        # 8. Absolute error
        ax8 = fig.add_subplot(3, 3, 8)
        error = np.abs(self.u_pred - self.u_true)
        im8 = ax8.contourf(self.X_test, self.T_test, error, levels=20, cmap="hot")
        ax8.set_xlabel("x")
        ax8.set_ylabel("t")
        ax8.set_title("Absolute Error")
        plt.colorbar(im8, ax=ax8)

        # 9. Relative error in parameter space
        ax9 = fig.add_subplot(3, 3, 9)
        alpha_range = np.linspace(0.1, 1.0, 50)
        data_losses = []

        # Compute data loss for different alpha values
        for alpha_test in alpha_range:
            u_test = np.sin(np.pi * self.x_obs[:, 0]) * np.exp(
                -alpha_test * np.pi**2 * self.x_obs[:, 1]
            )
            data_loss = np.mean((u_test - self.u_obs.ravel()) ** 2)
            data_losses.append(data_loss)

        ax9.plot(alpha_range, data_losses, "b-", linewidth=2)
        ax9.axvline(x=self.true_alpha, color="r", linestyle="--", label="True α")
        ax9.axvline(
            x=self.discovered_alpha, color="g", linestyle="-", label="Discovered α"
        )
        ax9.set_xlabel("α")
        ax9.set_ylabel("Data Loss")
        ax9.set_title("Loss Landscape")
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Main function for inverse problem demonstration"""

    print("\n" + "=" * 70)
    print("🔬 PHYSICS-INFORMED NEURAL NETWORKS FOR INVERSE PROBLEMS")
    print("=" * 70)

    print("\n📚 WHAT ARE INVERSE PROBLEMS?")
    print("-" * 40)
    print("• Forward Problem: Known parameters → Find solution")
    print("• Inverse Problem: Known (partial) solution → Find parameters")
    print("\nExamples in real world:")
    print("• Medical imaging: measurements → internal structure")
    print("• Seismology: surface waves → underground properties")
    print("• Material science: observed behavior → material properties")

    print("\n🎯 THIS EXAMPLE:")
    print("-" * 40)
    print("We have a heat diffusion process with UNKNOWN thermal diffusivity")
    print("Given: Sparse, noisy temperature measurements")
    print(
        "Goal: Discover the thermal diffusivity AND reconstruct full temperature field"
    )

    # Create and solve inverse problem
    solver = InverseProblemSolver(
        true_alpha=0.5,  # True parameter (unknown to model)
        x_domain=(0.0, 1.0),
        t_domain=(0.0, 1.0),
        n_observations=100,  # Only 100 sparse measurements
        n_collocation=5000,
        noise_level=0.02,  # 2% noise in measurements
    )

    # Train to discover parameter
    solver.train(
        n_epochs=8000,
        learning_rate=0.001,  # For neural network
        param_lr=0.01,  # For parameter discovery
        print_every=1000,
    )

    # Evaluate reconstruction
    solver.evaluate()

    # Visualize results
    solver.plot_results()

    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS ABOUT INVERSE PROBLEMS WITH PINNs")
    print("=" * 70)

    print("\n1. DATA-PHYSICS TRADE-OFF:")
    print("   • Data loss: Fits observations")
    print("   • Physics loss: Ensures physical consistency")
    print("   • Balance is crucial for accurate parameter discovery")

    print("\n2. ADVANTAGES OVER TRADITIONAL METHODS:")
    print("   • No need for repeated forward simulations")
    print("   • Can handle sparse and noisy data")
    print("   • Provides uncertainty quantification naturally")
    print("   • Simultaneous parameter discovery and solution reconstruction")

    print("\n3. CHALLENGES:")
    print("   • Non-convex optimization landscape")
    print("   • Multiple parameters can be harder to identify")
    print("   • Requires careful initialization and tuning")

    print("\n4. APPLICATIONS:")
    print("   • System identification")
    print("   • Parameter estimation from experiments")
    print("   • Data assimilation")
    print("   • Model discovery")

    print("\n" + "=" * 70)
    print("🎉 Inverse Problem Tutorial Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
