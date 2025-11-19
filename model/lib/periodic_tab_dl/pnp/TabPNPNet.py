import torch
from torch import Tensor

from ..base.BaseTabNet import BaseTabNet
from ..pnp.PNPEncoder import PNPEncoder


class TabPNPNet(BaseTabNet):
    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            hidden_size: int,
            output_size: int,
            use_residual: bool = False
    ):
        """
        TabPNPNet processes continuous (periodic and non-periodic) and categorical features. It applies Fourier
        transformations to periodic inputs and Chebyshev transformations to non-periodic inputs, with optional
        residual connections.

        :param periodic_input_size: Number of periodic continuous input features.
        :param non_periodic_input_size: Number of non-periodic continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features per periodic feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms for non-periodic features.
        :param hidden_size: Number of neurons in hidden layers for categorical feature processing.
        :param output_size: Size of the model's output; >1 indicates multi-output, 1 for single-output tasks.
        :param use_residual: If True, applies residual connections in the processing layers.
        """
        # Initialize PNPEncoder for periodic and non-periodic feature transformations
        pnp_layer = PNPEncoder(
            periodic_input_size=periodic_input_size,
            non_periodic_input_size=non_periodic_input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms
        )

        # Initialize BaseTabNet with the PNP layer for continuous features and an MLP for categorical features
        super(TabPNPNet, self).__init__(
            processing_layer=pnp_layer,
            categorical_input_size=categorical_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_residual=use_residual
        )

    def forward(
            self,
            x_continuous_periodic: Tensor,
            x_continuous_non_periodic: Tensor,
            x_categorical: Tensor
    ) -> Tensor:
        """
        Forward pass through the TabPNPNet, applying Fourier and Chebyshev transformations to periodic and non-periodic
        inputs, followed by MLP layers for categorical processing and final output.

        :param x_continuous_periodic: Tensor of periodic continuous features.
        :param x_continuous_non_periodic: Tensor of non-periodic continuous features.
        :param x_categorical: Tensor of one-hot encoded categorical features.
        :return: Model output after processing all inputs.
        """
        # Transform continuous features using Fourier and Chebyshev layers
        x_continuous_processed = self.processing_layer(x_continuous_periodic, x_continuous_non_periodic)

        # Process categorical features using the categorical MLP
        x_categorical_processed = self.categorical_layer(x_categorical)

        # Concatenate processed continuous and categorical features
        x_combined = torch.cat([x_continuous_processed, x_categorical_processed], dim=1)

        # Pass concatenated features through the final MLP layer for output
        return self.mlp(x_combined)
