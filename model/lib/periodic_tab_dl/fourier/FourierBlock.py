from ..base.BaseBlock import BaseBlock
from ..fourier.FourierEncoder import FourierEncoder


class FourierBlock(BaseBlock):
    def __init__(
            self,
            input_size: int,
            num_features_per_input: int,
            num_layers: int,
            compression_dim: int,
            dropout_prob: float
    ):
        """
        FourierBlock that stacks multiple FourierEncoder layers with optional compression layers,
        batch normalization, layer normalization, and dropout.

        :param input_size: Size of the input features.
        :param num_features_per_input: Number of Fourier features to generate in each layer.
        :param num_layers: Number of FourierEncoder layers to stack.
        :param compression_dim: Dimension to compress features to between layers.
        :param dropout_prob: Probability of dropout after normalization layers (0 to 1).
        """
        super(FourierBlock, self).__init__(
            input_size=input_size,
            encoder=FourierEncoder,
            encoder_params={"num_features_per_input": num_features_per_input},
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )
