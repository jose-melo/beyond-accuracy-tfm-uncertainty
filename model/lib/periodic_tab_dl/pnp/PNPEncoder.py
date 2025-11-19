import torch
from torch import nn

from ..fourier.FourierBlock import FourierBlock
from ...orthogonal_poly.OrthogonalPolynomialBlock import \
    OrthogonalPolynomialBlock
from src.config import POLY_TYPE


class PNPEncoder(nn.Module):
    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            num_layers: int = 1,
            compression_dim=None,
            dropout_prob: float = 0.2
    ):
        """
        PNPEncoder combines Fourier and Chebyshev transformations for periodic and non-periodic inputs, respectively,
        with residual connections for enhanced feature learning.

        :param periodic_input_size: Number of periodic input features.
        :param non_periodic_input_size: Number of non-periodic input features.
        :param num_fourier_features: Number of Fourier features to generate per periodic feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms per non-periodic feature.
        """
        super(PNPEncoder, self).__init__()

        # Initialize FourierBlock for periodic features, if applicable
        self.fourier_layer = FourierBlock(
            input_size=periodic_input_size,
            num_features_per_input=num_fourier_features,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        ) if periodic_input_size > 0 else None

        # Initialize OrthogonalPolynomialBlock for non-periodic features, if applicable
        self.orthogonal_poly_layer = OrthogonalPolynomialBlock(
            input_size=non_periodic_input_size,
            max_poly_terms=max_poly_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob,
            polynomial_type=POLY_TYPE
        ) if non_periodic_input_size > 0 else None

        # Calculate total output dimensions based on Fourier and Chebyshev layers
        total_fourier_features = self.fourier_layer.output_dim if self.fourier_layer else 0
        total_chebyshev_features = self.orthogonal_poly_layer.output_dim if self.orthogonal_poly_layer else 0
        self.output_dim = total_fourier_features + total_chebyshev_features

    def forward(self, x_periodic: torch.Tensor = None, x_non_periodic: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through Fourier and Chebyshev transformations.

        :param x_periodic: Tensor of periodic features with shape [batch_size, periodic_input_size].
        :param x_non_periodic: Tensor of non-periodic features with shape [batch_size, non_periodic_input_size].
        :return: Concatenated output tensor of transformed features.
        """
        transformed_features = []

        # Apply Fourier transformation if periodic features are provided
        if self.fourier_layer and x_periodic is not None:
            transformed_features.append(self.fourier_layer(x_periodic))

        # Apply Chebyshev transformation if non-periodic features are provided
        if self.orthogonal_poly_layer and x_non_periodic is not None:
            transformed_features.append(self.orthogonal_poly_layer(x_non_periodic))

        # Concatenate transformed features or return an empty tensor if no features are available
        if transformed_features:
            return torch.cat(transformed_features, dim=-1)
        else:
            device = x_periodic.device if x_periodic is not None else x_non_periodic.device
            return torch.empty(0, device=device)
