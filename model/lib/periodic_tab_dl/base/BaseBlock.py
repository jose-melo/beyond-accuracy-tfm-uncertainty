from typing import Dict, Callable

import torch
from torch import nn


class BaseBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            encoder: Callable,
            encoder_params: Dict,
            num_layers: int,
            compression_dim: int,
            dropout_prob: float
    ):
        """
        BaseBlock stacks multiple encoder layers (e.g., ChebyshevEncoder) with optional compression layers, batch normalization,
        and dropout between each layer.

        :param input_size: Size of the input features.
        :param encoder: Encoder module class (e.g., ChebyshevEncoder) used for feature transformation.
        :param encoder_params: Dictionary of parameters for initializing the encoder.
        :param num_layers: Number of encoder layers to stack.
        :param compression_dim: Dimension to compress features between layers; if None, retains input size.
        :param dropout_prob: Dropout probability after normalization layers (0 to 1).
        """
        super(BaseBlock, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_input_size = input_size

        for i in range(num_layers):
            # Initialize the encoder layer
            encoder_layer = encoder(current_input_size, **encoder_params)
            self.layers.append(encoder_layer)

            # Compute the output feature dimension after the encoder layer
            total_features = encoder_layer.output_dim * current_input_size

            if i < num_layers - 1:
                # Add optional compression, batch normalization, and dropout layers
                compress_dim = compression_dim if compression_dim else current_input_size
                self.compress_layers.append(nn.Linear(total_features, compress_dim))
                self.norm_layers.append(nn.BatchNorm1d(compress_dim))
                self.dropout_layers.append(nn.Dropout(p=dropout_prob))
                current_input_size = compress_dim
            else:
                # Set the final output dimension of the block
                self.output_dim = total_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BaseBlock.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Output tensor after stacked transformations and optional normalization/dropout.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply transformation
            if i < self.num_layers - 1:
                x = self.compress_layers[i](x)
                x = self.norm_layers[i](x)
                x = self.dropout_layers[i](x)
        return x
