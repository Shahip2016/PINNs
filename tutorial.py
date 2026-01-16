"""
Physics-Informed Neural Networks (PINNs) - Complete Tutorial
=============================================================

This tutorial will teach you everything about PINNs from scratch.
We'll cover theory, implementation, and practical examples.
"""

import os
import sys
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     PHYSICS-INFORMED NEURAL NETWORKS (PINNs) - COMPLETE TUTORIAL    ║
║                                                                      ║
║     Learn how to solve PDEs with Deep Learning                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PART 1: UNDERSTANDING PINNs
# ============================================================================

print("""
================================================================================
PART 1: WHAT ARE PHYSICS-INFORMED NEURAL NETWORKS?
================================================================================

Traditional PDE Solvers vs PINNs
---------------------------------

TRADITIONAL NUMERICAL METHODS (FEM, FDM, FVM):
• Discretize domain into mesh/grid
• Approximate derivatives with finite differences
• Solve large linear/nonlinear systems
• Accuracy depends on mesh resolution
• Can be computationally expensive for complex geometries

PHYSICS-INFORMED NEURAL NETWORKS:
• Use neural networks as universal function approximators
• Encode physics directly in the loss function
• Mesh-free approach
• Leverage automatic differentiation
• Can handle complex geometries naturally

The Key Insight
---------------
Neural networks can approximate any continuous function (Universal Approximation Theorem).
By training them to satisfy PDEs, boundary conditions, and initial conditions,
we get solutions that respect the underlying physics.

Mathematical Foundation
-----------------------
Consider a general PDE:
    F(u, ∂u/∂x, ∂²u/∂x², ..., x, t) = 0  in domain Ω
    B(u, x, t) = 0  on boundary ∂Ω
    I(u, x, 0) = u₀(x)  initial condition

A PINN approximates u(x,t) with a neural network u_θ(x,t) where θ are the network parameters.

The loss function combines:
    L = L_physics + L_boundary + L_initial + L_data

Where:
    L_physics = mean(|F(u_θ, ∂u_θ/∂x, ...)|²)  - PDE residual
    L_boundary = mean(|B(u_θ)|²)               - Boundary conditions
    L_initial = mean(|I(u_θ) - u₀|²)           - Initial conditions
    L_data = mean(|u_θ - u_observed|²)         - Data fitting (if available)
""")

input("\nPress Enter to continue to Part 2...")

# ============================================================================
# PART 2: A SIMPLE IMPLEMENTATION
# ============================================================================

print("""
================================================================================
PART 2: IMPLEMENTING A SIMPLE PINN
================================================================================

Let's solve a simple ODE to understand the core concepts:
    du/dx = -2x,  u(0) = 1

Analytical solution: u(x) = 1 - x²
""")


class SimplePINN(nn.Module):
    """A minimal PINN for solving a simple ODE"""

    def __init__(self, layers=[1, 20, 20, 1]):
        super(SimplePINN, self).__init__()

        # Build network
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Initialize weights
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def train_simple_ode():
    """Train PINN to solve du/dx = -2x, u(0) = 1"""

    print("\nTraining Simple ODE Example...")
    print("-" * 40)

    # Create model
    model = SimplePINN([1, 20, 20, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training data
    # Collocation points (where ODE should be satisfied)
    x_collocation = torch.linspace(0, 2, 100).view(-1, 1).to(device)
    x_collocation.requires_grad = True

    # Boundary point
    x_boundary = torch.tensor([[0.0]], requires_grad=True).to(device)
    u_boundary = torch.tensor([[1.0]]).to(device)

    # Training loop
    losses = []
    for epoch in range(5000):
        optimizer.zero_grad()

        # Compute network output
        u = model(x_collocation)

        # Compute derivative du/dx using autograd
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]

        # Physics loss: du/dx + 2x = 0
        physics_loss = torch.mean((du_dx + 2 * x_collocation) ** 2)

        # Boundary loss: u(0) = 1
        u_boundary_pred = model(x_boundary)
        boundary_loss = torch.mean((u_boundary_pred - u_boundary) ** 2)

        # Total loss
        loss = physics_loss + boundary_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")

    # Plot results
    model.eval()
    x_test = torch.linspace(0, 2, 200).view(-1, 1).to(device)
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy()

    x_test_np = x_test.cpu().numpy()
    u_true = 1 - x_test_np**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Solution plot
    ax1.plot(x_test_np, u_true, "b-", label="Analytical: u(x) = 1 - x²", linewidth=2)
    ax1.plot(x_test_np, u_pred, "r--", label="PINN Prediction", linewidth=2)
    ax1.scatter([0], [1], color="green", s=100, zorder=5, label="Boundary Condition")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax1.set_title("ODE Solution: du/dx = -2x, u(0) = 1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss history
    ax2.semilogy(losses)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss History")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    error = np.mean(np.abs(u_pred - u_true))
    print(f"\nMean Absolute Error: {error:.6f}")
    print("✓ Simple ODE solved successfully!")


# Run simple example
train_simple_ode()

input("\nPress Enter to continue to Part 3...")

# ============================================================================
# PART 3: AUTOMATIC DIFFERENTIATION - THE SECRET SAUCE
# ============================================================================

print("""
================================================================================
PART 3: AUTOMATIC DIFFERENTIATION - THE SECRET SAUCE
================================================================================

The Power of Automatic Differentiation
---------------------------------------
Traditional numerical methods approximate derivatives using finite differences:
    ∂u/∂x ≈ (u(x+h) - u(x))/h

This introduces discretization errors and requires careful choice of h.

PINNs use automatic differentiation (autograd) to compute EXACT derivatives
of the neural network output with respect to inputs.

How It Works
------------
1. Neural network: u = f_θ(x)
2. PyTorch tracks all operations
3. Derivatives computed via chain rule
4. No discretization error!

Example: Computing Higher-Order Derivatives
""")


def demonstrate_autograd():
    """Demonstrate automatic differentiation"""

    print("\nDemonstrating Automatic Differentiation")
    print("-" * 40)

    # Create a simple function: u(x) = x³
    x = torch.tensor([[2.0]], requires_grad=True)
    u = x**3

    print(f"Function: u(x) = x³")
    print(f"At x = {x.item():.1f}: u = {u.item():.1f}")

    # First derivative: du/dx = 3x²
    du_dx = torch.autograd.grad(u, x, create_graph=True)[0]
    print(f"First derivative du/dx = 3x² = {du_dx.item():.1f}")

    # Second derivative: d²u/dx² = 6x
    d2u_dx2 = torch.autograd.grad(du_dx, x, create_graph=True)[0]
    print(f"Second derivative d²u/dx² = 6x = {d2u_dx2.item():.1f}")

    # Third derivative: d³u/dx³ = 6
    d3u_dx3 = torch.autograd.grad(d2u_dx2, x)[0]
    print(f"Third derivative d³u/dx³ = 6 = {d3u_dx3.item():.1f}")

    print("\n✓ All derivatives computed exactly using automatic differentiation!")


demonstrate_autograd()

input("\nPress Enter to continue to Part 4...")

# ============================================================================
# PART 4: SOLVING THE HEAT EQUATION
# ============================================================================

print("""
================================================================================
PART 4: SOLVING A PDE - THE HEAT EQUATION
================================================================================

The 1D Heat Equation
--------------------
∂u/∂t = α * ∂²u/∂x²

Where:
• u(x,t) is temperature
• α is thermal diffusivity
• Domain: x ∈ [0, 1], t ∈ [0, 0.1]

Conditions:
• Initial: u(x,0) = sin(πx)
• Boundary: u(0,t) = u(1,t) = 0

Analytical solution: u(x,t) = sin(πx) * exp(-απ²t)
""")


class HeatPINN(nn.Module):
    """PINN for solving the heat equation"""

    def __init__(self):
        super(HeatPINN, self).__init__()

        # Network architecture
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)

        # Initialize weights
        for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """Forward pass"""
        inputs = torch.cat([x, t], dim=1)
        h = torch.tanh(self.layer1(inputs))
        h = torch.tanh(self.layer2(h))
        h = torch.tanh(self.layer3(h))
        return self.layer4(h)


def solve_heat_equation():
    """Solve heat equation using PINN"""

    print("\nSolving Heat Equation with PINN...")
    print("-" * 40)

    # Parameters
    alpha = 0.1

    # Create model
    model = HeatPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate training points
    n_collocation = 2000
    n_boundary = 100
    n_initial = 100

    # Collocation points (inside domain)
    x_col = torch.rand(n_collocation, 1) * 1.0
    t_col = torch.rand(n_collocation, 1) * 0.1
    x_col = x_col.to(device).requires_grad_(True)
    t_col = t_col.to(device).requires_grad_(True)

    # Initial condition points
    x_init = torch.linspace(0, 1, n_initial).view(-1, 1).to(device)
    t_init = torch.zeros(n_initial, 1).to(device)
    u_init = torch.sin(np.pi * x_init).to(device)

    # Boundary points
    t_bound = torch.linspace(0, 0.1, n_boundary).view(-1, 1).to(device)
    x_left = torch.zeros(n_boundary, 1).to(device)
    x_right = torch.ones(n_boundary, 1).to(device)

    # Training
    print("Training...")
    losses = []

    for epoch in range(3000):
        optimizer.zero_grad()

        # Physics loss (PDE residual)
        u = model(x_col, t_col)

        # Compute derivatives
        u_x = torch.autograd.grad(
            u, x_col, torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_col, torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u, t_col, torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        # Heat equation residual
        residual = u_t - alpha * u_xx
        physics_loss = torch.mean(residual**2)

        # Initial condition loss
        u_init_pred = model(x_init, t_init)
        initial_loss = torch.mean((u_init_pred - u_init) ** 2)

        # Boundary condition loss
        u_left = model(x_left, t_bound)
        u_right = model(x_right, t_bound)
        boundary_loss = torch.mean(u_left**2) + torch.mean(u_right**2)

        # Total loss
        loss = physics_loss + initial_loss + boundary_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")

    # Evaluate and plot
    print("\nEvaluating solution...")
    model.eval()

    # Create test grid
    x_test = torch.linspace(0, 1, 50).to(device)
    t_test = torch.linspace(0, 0.1, 50).to(device)
    X, T = torch.meshgrid(x_test, t_test, indexing="ij")

    with torch.no_grad():
        U_pred = model(X.reshape(-1, 1), T.reshape(-1, 1))
        U_pred = U_pred.reshape(50, 50).cpu().numpy()

    # Analytical solution
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()
    U_true = np.sin(np.pi * X_np) * np.exp(-alpha * np.pi**2 * T_np)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True solution
    im1 = axes[0, 0].contourf(X_np, T_np, U_true, levels=20, cmap="viridis")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("t")
    axes[0, 0].set_title("True Solution")
    plt.colorbar(im1, ax=axes[0, 0])

    # PINN solution
    im2 = axes[0, 1].contourf(X_np, T_np, U_pred, levels=20, cmap="viridis")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    axes[0, 1].set_title("PINN Solution")
    plt.colorbar(im2, ax=axes[0, 1])

    # Error
    error = np.abs(U_pred - U_true)
    im3 = axes[1, 0].contourf(X_np, T_np, error, levels=20, cmap="hot")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    axes[1, 0].set_title("Absolute Error")
    plt.colorbar(im3, ax=axes[1, 0])

    # Loss history
    axes[1, 1].semilogy(losses)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nMean Absolute Error: {np.mean(error):.6f}")
    print("✓ Heat equation solved successfully!")


solve_heat_equation()

input("\nPress Enter to continue to Part 5...")

# ============================================================================
# PART 5: ADVANCED CONCEPTS
# ============================================================================

print("""
================================================================================
PART 5: ADVANCED CONCEPTS AND BEST PRACTICES
================================================================================

1. NETWORK ARCHITECTURE
------------------------
• Depth: 3-6 hidden layers typically sufficient
• Width: 20-50 neurons per layer for simple problems
• Activation: tanh often works better than ReLU for PINNs
• Initialization: Xavier/Glorot initialization recommended

2. SAMPLING STRATEGIES
----------------------
• Uniform sampling: Good for simple domains
• Latin hypercube: Better space-filling properties
• Adaptive sampling: Add points where error is high
• Importance sampling: Focus on difficult regions

3. LOSS BALANCING
-----------------
• Different loss terms may have different scales
• Use adaptive weights or gradient normalization
• Monitor individual losses during training

4. OPTIMIZATION STRATEGIES
--------------------------
• Adam for initial training
• L-BFGS for fine-tuning
• Learning rate scheduling
• Gradient clipping for stability

5. HANDLING COMPLEX PHYSICS
---------------------------
• Domain decomposition for large problems
• Transfer learning from simpler problems
• Physics-informed neural operators for parametric PDEs
• Variational formulations for conservation laws

6. INVERSE PROBLEMS
-------------------
PINNs can discover unknown parameters in PDEs:
• Make parameters learnable (nn.Parameter)
• Use sparse observations as additional loss
• Simultaneously learn parameters and solution

7. ADVANTAGES OF PINNs
----------------------
✓ Mesh-free method
✓ Handle complex geometries easily
✓ Natural handling of inverse problems
✓ Continuous, differentiable solutions
✓ Can incorporate sparse data
✓ Transfer learning capabilities

8. LIMITATIONS TO CONSIDER
--------------------------
✗ Can struggle with sharp gradients
✗ Training can be slow for large problems
✗ Hyperparameter tuning required
✗ May need many collocation points
✗ Optimization landscape can be complex
""")

input("\nPress Enter to see the final summary...")

# ============================================================================
# SUMMARY
# ============================================================================

print("""
================================================================================
SUMMARY: PHYSICS-INFORMED NEURAL NETWORKS
================================================================================

What We've Learned
------------------
1. PINNs encode physics directly into neural network training
2. Automatic differentiation enables exact derivative computation
3. Loss function combines physics, boundaries, and data
4. Can solve forward and inverse problems
5. Mesh-free approach with continuous solutions

Key Takeaways
-------------
• PINNs bridge deep learning and scientific computing
• They offer a new paradigm for solving PDEs
• Particularly useful for:
  - Inverse problems
  - Complex geometries
  - Data assimilation
  - Uncertainty quantification

Next Steps
----------
1. Try the detailed examples in the examples/ folder:
   - heat_equation_1d.py: Full heat equation solver
   - inverse_problem.py: Parameter discovery example

2. Experiment with different:
   - Network architectures
   - Sampling strategies
   - Loss weights
   - PDEs

3. Explore advanced topics:
   - Adaptive PINNs
   - Conservative PINNs
   - Variational PINNs
   - Physics-informed DeepONets

Resources for Further Learning
-------------------------------
• Original paper: Raissi et al. (2019) - "Physics-informed neural networks"
• DeepXDE library for production use
• SciANN for TensorFlow implementation
• NeuralPDE.jl for Julia

================================================================================
🎉 TUTORIAL COMPLETE! You now understand the fundamentals of PINNs!
================================================================================

To run more examples:
  python examples/heat_equation_1d.py
  python examples/inverse_problem.py
""")

print("\n" + "=" * 70)
print("Thank you for learning about Physics-Informed Neural Networks!")
print("=" * 70)
