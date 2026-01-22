import torch
import numpy as np
import pytest
from src.pinn_model import PINN
from utils.pde_utils import PDEOperators

def test_pinn_initialization():
    model = PINN(input_dim=2, output_dim=1, hidden_layers=[10, 10])
    assert model.input_dim == 2
    assert model.output_dim == 1
    # Check if weights are initialized (not all zeros)
    for param in model.parameters():
        assert torch.sum(torch.abs(param)) > 0

def test_pinn_forward():
    model = PINN(input_dim=2, output_dim=1, hidden_layers=[10, 10])
    x = torch.randn(5, 2).to(model.device)
    y = model(x)
    assert y.shape == (5, 1)

def test_pde_operators_gradient():
    # u(x, y) = x^2 + y^2
    # grad u = [2x, 2y]
    def u_func(x):
        return (x**2).sum(dim=-1, keepdim=True)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    grad = PDEOperators.gradient(u_func, x)
    
    expected_grad = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    assert torch.allclose(grad, expected_grad)

def test_pde_operators_laplacian():
    # u(x, y) = x^2 + y^2
    # laplacian u = 2 + 2 = 4
    def u_func(x):
        return (x**2).sum(dim=-1, keepdim=True)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lap = PDEOperators.laplacian(u_func, x)
    
    expected_lap = torch.tensor([[4.0], [4.0]])
    assert torch.allclose(lap, expected_lap)

def test_device_placement():
    model = PINN(input_dim=2, output_dim=1, hidden_layers=[10])
    # Ensure parameters are on the expected device
    for param in model.parameters():
        assert param.device.type in ["cuda", "mps", "cpu"]
