from ..autopnp.AutoPNPBlock import AutoPNPBlock
from ..base.BaseNet import BaseNet


class AutoPNPNet(BaseNet):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            output_size: int = 1,
            use_residual: bool = True,
            poly_type: str = "chebyshev"
    ):
        """
        AutoPNPNet integrates the AutoPNPBlock with Fourier and Chebyshev transformations in a multi-layer structure.
        Includes an optional residual connection to enhance stability in feature learning.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features to generate per layer.
        :param max_poly_terms: Number of Chebyshev polynomial terms per layer.
        :param num_layers: Number of AutoPNPBlock layers to stack.
        :param compression_dim: Dimension to compress features between layers; if None, retains input size.
        :param output_size: Size of the network output; >1 indicates multi-output, 1 for single-output tasks.
        :param use_residual: Whether to apply a residual connection for continuous feature enhancement.
        """
        # Initialize the AutoPNPBlock with specified parameters
        autopnp_block = AutoPNPBlock(
            input_size=input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            poly_type=poly_type
        )

        # Initialize BaseNet with AutoPNPBlock and optional residual connection
        super(AutoPNPNet, self).__init__(
            processing_layer=autopnp_block,
            output_size=output_size,
            use_residual=use_residual
        )
