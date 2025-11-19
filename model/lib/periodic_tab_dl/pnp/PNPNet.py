from torch import Tensor

from ..base.BaseNet import BaseNet
from ..pnp.PNPEncoder import PNPEncoder


class PNPNet(BaseNet):
    def __init__(
            self,
            periodic_input_size: int,
            non_periodic_input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            output_size: int,
            use_residual: bool = False
    ):
        """
        PNPNet integrates Fourier and Chebyshev transformations for periodic and non-periodic input features
        within an MLP framework. Residual connections can be applied to enhance stability in feature processing.

        :param periodic_input_size: Number of periodic input features.
        :param non_periodic_input_size: Number of non-periodic input features.
        :param num_fourier_features: Number of Fourier features generated per periodic feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms for non-periodic features.
        :param output_size: Desired size of the model's output; >1 indicates multi-output, 1 for single-output tasks.
        :param use_residual: If True, applies residual connections in the processing layers.
        """
        # Initialize PNPEncoder for feature transformation
        pnp_layer = PNPEncoder(
            periodic_input_size=periodic_input_size,
            non_periodic_input_size=non_periodic_input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms
        )

        # Initialize BaseNet with the configured PNP layer and residual connections if specified
        super(PNPNet, self).__init__(
            processing_layer=pnp_layer,
            output_size=output_size,
            use_residual=use_residual
        )

    def forward(self, x_periodic: Tensor, x_non_periodic: Tensor) -> Tensor:
        """
        Forward pass through the PNPNet, applying Fourier and Chebyshev transformations to periodic and non-periodic
        inputs, followed by MLP layers for final output.

        :param x_periodic: Tensor containing periodic input features.
        :param x_non_periodic: Tensor containing non-periodic input features.
        :return: Model output after processing.
        """
        # Apply the processing layer (PNPEncoder) to both periodic and non-periodic features
        x_processed = self.processing_layer(x_periodic, x_non_periodic)

        # Apply the MLP layer for final output transformation
        return self.mlp(x_processed)
