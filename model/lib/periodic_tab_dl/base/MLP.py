import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 32,
            output_size: int = 1,
            num_layers: int = 1,
            dropout_prob: float = 0.2,
            batch_norm: bool = True,
            activation_fn: nn.Module = nn.ReLU()
    ):
        """
        MLP: A flexible Multi-Layer Perceptron base module.

        :param input_size: Size of the input features.
        :param hidden_size: Size of hidden layers.
        :param output_size: Size of the output (1 for regression tasks).
        :param num_layers: Number of fully connected hidden layers in the MLP.
        :param dropout_prob: Dropout probability for regularization.
        :param batch_norm: Whether to use batch normalization.
        :param activation_fn: Activation function to apply after each layer (default is ReLU).
        """
        super(MLP, self).__init__()

        # Build the network layers
        layers = []
        current_input_size = input_size

        if num_layers > 0:
            for _ in range(num_layers):
                layers.append(nn.Linear(current_input_size, hidden_size))

                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))

                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout_prob))

                current_input_size = hidden_size

        # Output layer (no activation function for regression)
        layers.append(nn.Linear(current_input_size, output_size))

        # Create the sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor of shape [batch_size, output_size] (regression predictions).
        """
        return self.network(x)
