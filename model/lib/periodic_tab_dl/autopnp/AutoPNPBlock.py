from ..autopnp.AutoPNPEncoder import AutoPNPEncoder
from ..base.BaseBlock import BaseBlock


class AutoPNPBlock(BaseBlock):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.1,
            poly_type:str = "chebyshev",
    ):
        """
        AutoPNPBlock stacks multiple AutoPNPEncoder layers, applying optional compression, batch normalization,
        and dropout between each layer for flexible feature transformation.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features to generate per layer.
        :param max_poly_terms: Number of Chebyshev polynomial terms in each layer.
        :param num_layers: Number of stacked AutoPNPEncoder layers.
        :param compression_dim: Target dimensionality for feature compression between layers; if None, defaults to input size.
        :param dropout_prob: Probability of dropout applied after each batch normalization layer (0 to 1).
        """
        # Initialize the BaseBlock with AutoPNPEncoder and specified parameters
        super(AutoPNPBlock, self).__init__(
            input_size=input_size,
            encoder=AutoPNPEncoder,
            encoder_params={
                "num_fourier_features": num_fourier_features,
                "max_poly_terms": max_poly_terms,
                "poly_type": poly_type
            },
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )
