import torch
from torch import nn

from ..base.MLP import MLP
from ..categorical.CategoricalMLP import CategoricalMLP


class BaseTabNet(nn.Module):
    def __init__(
            self,
            processing_layer: nn.Module,
            categorical_input_size: int,
            hidden_size: int,
            output_size: int,
            use_residual: bool
    ):
        """
        BaseTabNet integrates continuous and categorical feature processing with an optional residual connection for
        continuous features. Processes continuous features via a Fourier or custom transformation layer and categorical
        features via a categorical MLP.

        :param processing_layer: Transformation layer for continuous features (e.g., Fourier layer).
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param hidden_size: Size of hidden layers for categorical feature processing.
        :param output_size: Output size; if > 1, indicates classification; otherwise, regression.
        :param use_residual: Whether to apply a residual connection to continuous features.
        """
        super(BaseTabNet, self).__init__()

        self.processing_layer = processing_layer
        self.use_residual = use_residual

        # Initialize categorical feature processing
        self.categorical_layer = CategoricalMLP(input_size=categorical_input_size, hidden_size=hidden_size)

        # Determine total features by combining processed continuous and categorical features
        self.total_features = self.processing_layer.output_dim + hidden_size
        self.mlp = MLP(input_size=self.total_features, output_size=output_size)

    def forward(self, x_continuous: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BaseTabNet model.

        :param x_continuous: Tensor of continuous features, shape [batch_size, continuous_input_size].
        :param x_categorical: Tensor of one-hot encoded categorical features, shape [batch_size, categorical_input_size].
        :return: Output tensor after passing through the network.
        """

        # Transform continuous features and add residual if enabled
        x_continuous_processed = self.processing_layer(x_continuous)
        if self.use_residual and x_continuous.shape == x_continuous_processed.shape:
            x_continuous_processed += x_continuous

        # Process one-hot encoded categorical features
        x_categorical_processed = self.categorical_layer(x_categorical)

        # Concatenate processed continuous and categorical features
        x_combined = torch.cat([x_continuous_processed, x_categorical_processed], dim=1)

        # Pass through final MLP layer for output
        return self.mlp(x_combined)
