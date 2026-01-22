"""
1D Heat Equation Example using Physics-Informed Neural Networks (PINNs)
Optimized with torch.func and Device Awareness
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pinn_model import PINN
from utils.pde_utils import DataGenerator, PDELibrary, PDEOperators, Visualizer

# Setup
torch.manual_seed(42)
np.random.seed(42)

def main():
    # Parameters
    alpha = 0.01
    n_collocation = 5000
    n_epochs = 2000
    
    # Initialize model
    model = PINN(
        input_dim=2, 
        output_dim=1, 
        hidden_layers=[32, 32, 32], 
        activation="tanh"
    )
    
    # Optional: model.compile() 
    
    # Generate Collocation Points
    bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    x_coll = DataGenerator.generate_collocation_points(bounds, n_collocation)
    x_coll_tensor = torch.tensor(x_coll).to(model.device)
    
    print(f"Training on device: {model.device}")
    
    # Define PDE Loss function
    def heat_loss_wrapper(model_in, **kwargs):
        return model_in.physics_loss(
            kwargs['x'], 
            PDELibrary.heat_equation_1d, 
            alpha=alpha
        )

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = model.train_model(
        optimizer=optimizer,
        n_epochs=n_epochs,
        loss_fn=heat_loss_wrapper,
        x=x_coll_tensor,
        print_every=500
    )
    
    # Visualize
    Visualizer.plot_loss_history(history)
    print("Training complete.")

if __name__ == "__main__":
    main()
