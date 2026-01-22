"""
Physics-Informed Neural Network (PINN) Model Implementation
Based on Raissi et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

# Use torch.func for advanced AD if available (PyTorch 2.0+)
try:
    import torch.func
    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDEs

    This implementation provides a flexible framework for:
    - Forward problems: Given PDE and boundary/initial conditions, find the solution
    - Inverse problems: Given some observations, infer unknown parameters in the PDE
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        init_type: str = "xavier",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the PINN model

        Args:
            input_dim: Dimension of input (e.g., 2 for (x, t) in 1D+time problems)
            output_dim: Dimension of output (e.g., 1 for scalar field u(x,t))
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'sin')
            init_type: Weight initialization ('xavier', 'kaiming', 'orthogonal')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            seed: Random seed for reproducibility
        """
        super(PINN, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.init_type = init_type

        # Select activation function
        self.activation_name = activation
        self.activation = self._get_activation(activation)

        # Build the network
        self.network = self._build_network()

        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)

        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()

        # Store training history
        self.loss_history = {
            "total": [],
            "data": [],
            "physics": [],
            "boundary": [],
            "initial": [],
        }

    def compile(self, **kwargs):
        """Compile the underlying network using torch.compile"""
        if hasattr(torch, "compile"):
            self.network = torch.compile(self.network, **kwargs)
            return self
        else:
            print("torch.compile is not available in this version of PyTorch")
            return self

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "sin": lambda x: torch.sin(x),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }

        if name.lower() not in activations:
            raise ValueError(
                f"Activation {name} not supported. Choose from {list(activations.keys())}"
            )

        return activations[name.lower()]

    def _build_network(self) -> nn.Sequential:
        """Build the neural network architecture"""
        layers = []

        # All layer dimensions
        all_layers = [self.input_dim] + self.hidden_layers + [self.output_dim]

        # Build hidden layers
        for i in range(len(all_layers) - 1):
            # Linear layer
            layers.append(
                ("linear_" + str(i), nn.Linear(all_layers[i], all_layers[i + 1]))
            )

            # Add batch normalization if specified (except for output layer)
            if self.use_batch_norm and i < len(all_layers) - 2:
                layers.append(("bn_" + str(i), nn.BatchNorm1d(all_layers[i + 1])))

            # Add activation (except for output layer)
            if i < len(all_layers) - 2:
                if callable(self.activation):
                    layers.append(
                        ("activation_" + str(i), LambdaLayer(self.activation))
                    )
                else:
                    layers.append(("activation_" + str(i), self.activation))

                # Add dropout if specified
                if self.dropout_rate > 0:
                    layers.append(("dropout_" + str(i), nn.Dropout(self.dropout_rate)))

        return nn.Sequential(OrderedDict(layers))

    def _initialize_weights(self):
        """Initialize network weights using selected strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif self.init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu" if self.activation_name == "relu" else "tanh")
                elif self.init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def compute_gradients(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        order: int = 1,
        retain_graph: bool = True,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """
        Compute gradients using automatic differentiation

        Args:
            outputs: Output tensor from the network
            inputs: Input tensor (must have requires_grad=True)
            order: Order of derivative (1 for first derivative, 2 for second, etc.)
            retain_graph: Whether to retain computation graph
            create_graph: Whether to create graph for higher-order derivatives

        Returns:
            Gradient tensor
        """
        gradient = outputs
        for _ in range(order):
            gradient = grad(
                outputs=gradient,
                inputs=inputs,
                grad_outputs=torch.ones_like(gradient),
                retain_graph=retain_graph,
                create_graph=create_graph,
            )[0]
        return gradient

    def physics_loss(
        self, x: torch.Tensor, pde_func: Callable, **pde_params
    ) -> torch.Tensor:
        """
        Compute physics loss (PDE residual)

        Args:
            x: Collocation points where PDE should be satisfied
            pde_func: Function that computes PDE residual
            **pde_params: Additional parameters for the PDE

        Returns:
            Physics loss (mean squared PDE residual)
        """
        # Ensure points are on the correct device
        x = x.to(self.device)
        
        if HAS_TORCH_FUNC:
            # When using torch.func, we pass a functional version of the model
            def functional_u(x_in):
                return self.forward(x_in)
            
            residual = pde_func(x, functional_u, self, **pde_params)
        else:
            # Fallback to standard autograd
            x.requires_grad = True
            u = self.forward(x)
            residual = pde_func(x, u, self, **pde_params)

        # Mean squared error of residual
        loss = torch.mean(residual**2)

        return loss

    def boundary_loss(
        self, x_boundary: torch.Tensor, boundary_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss

        Args:
            x_boundary: Points on the boundary
            boundary_values: Target values at boundary points

        Returns:
            Boundary loss
        """
        u_pred = self.forward(x_boundary)
        loss = torch.mean((u_pred - boundary_values) ** 2)
        return loss

    def initial_loss(
        self, x_initial: torch.Tensor, initial_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial condition loss

        Args:
            x_initial: Points at initial time
            initial_values: Target values at initial time

        Returns:
            Initial condition loss
        """
        u_pred = self.forward(x_initial)
        loss = torch.mean((u_pred - initial_values) ** 2)
        return loss

    def data_loss(self, x_data: torch.Tensor, u_data: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss

        Args:
            x_data: Input points with known values
            u_data: Known values at x_data

        Returns:
            Data loss
        """
        u_pred = self.forward(x_data)
        loss = torch.mean((u_pred - u_data) ** 2)
        return loss

    def total_loss(
        self, loss_dict: Dict[str, Tuple[torch.Tensor, float]]
    ) -> torch.Tensor:
        """
        Compute weighted total loss

        Args:
            loss_dict: Dictionary of losses with their weights
                      Format: {'loss_name': (loss_value, weight), ...}

        Returns:
            Total weighted loss
        """
        total = torch.tensor(0.0, requires_grad=True)

        for name, (loss_value, weight) in loss_dict.items():
            if loss_value is not None:
                total = total + weight * loss_value
                self.loss_history[name].append(loss_value.item())

        self.loss_history["total"].append(total.item())
        return total

    def train_model(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        loss_fn: Callable,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        verbose: bool = True,
        print_every: int = 100,
        **loss_kwargs,
    ) -> Dict:
        """
        Train the PINN model

        Args:
            optimizer: PyTorch optimizer
            n_epochs: Number of training epochs
            loss_fn: Function that computes the total loss
            scheduler: Learning rate scheduler (optional)
            verbose: Whether to print training progress
            print_every: Print frequency
            **loss_kwargs: Additional arguments for loss function

        Returns:
            Dictionary containing training history
        """
        self.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(self, **loss_kwargs)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional, helps with stability)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Update learning rate if scheduler is provided
            if scheduler is not None:
                scheduler.step()

            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.6e}, LR: {current_lr:.6e}"
                )

        return self.loss_history

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Make predictions

        Args:
            x: Input points
            return_numpy: Whether to return numpy array (True) or torch tensor (False)

        Returns:
            Predictions
        """
        self.eval()

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.forward(x)

        if return_numpy:
            return predictions.numpy()
        else:
            return predictions


class LambdaLayer(nn.Module):
    """Custom layer for lambda functions"""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AdaptiveWeights(nn.Module):
    """
    Adaptive weight calculation for multi-task learning in PINNs
    Based on "Self-Adaptive Physics-Informed Neural Networks"
    """

    def __init__(self, n_losses: int, method: str = "gradnorm"):
        """
        Initialize adaptive weights

        Args:
            n_losses: Number of loss components
            method: Weight adaptation method ('gradnorm', 'uncertainty', 'fixed')
        """
        super(AdaptiveWeights, self).__init__()
        self.n_losses = n_losses
        self.method = method

        if method == "uncertainty":
            # Learnable log variance parameters
            self.log_vars = nn.Parameter(torch.zeros(n_losses))
        else:
            # Fixed or computed weights
            self.weights = torch.ones(n_losses)

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted loss with adaptive weights

        Args:
            losses: List of loss components

        Returns:
            Tuple of (weighted total loss, weights)
        """
        if self.method == "uncertainty":
            # Uncertainty weighting
            precisions = torch.exp(-self.log_vars)
            weighted_losses = precisions * torch.stack(losses)
            total_loss = torch.sum(weighted_losses) + torch.sum(self.log_vars)
            weights = precisions
        elif self.method == "gradnorm":
            # GradNorm weighting (simplified version)
            weights = self.weights / torch.sum(self.weights)
            weighted_losses = weights * torch.stack(losses)
            total_loss = torch.sum(weighted_losses)
        else:
            # Fixed weights
            weights = self.weights
            weighted_losses = weights * torch.stack(losses)
            total_loss = torch.sum(weighted_losses)

        return total_loss, weights


class MultiScalePINN(PINN):
    """
    Multi-scale PINN using Fourier features for better handling of high-frequency solutions
    Based on "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        fourier_features: int = 256,
        fourier_scale: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Multi-scale PINN

        Args:
            fourier_features: Number of Fourier features
            fourier_scale: Scale factor for Fourier features
            **kwargs: Additional arguments for parent PINN class
        """
        self.fourier_features = fourier_features
        self.fourier_scale = fourier_scale

        # Random Fourier feature matrix
        self.B = nn.Parameter(
            torch.randn(input_dim, fourier_features) * fourier_scale,
            requires_grad=False,
        )

        # Adjust input dimension for Fourier features (sin and cos)
        super().__init__(
            input_dim=2 * fourier_features,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            **kwargs,
        )

    def fourier_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding"""
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Fourier encoding"""
        x_encoded = self.fourier_encoding(x)
        return self.network(x_encoded)
