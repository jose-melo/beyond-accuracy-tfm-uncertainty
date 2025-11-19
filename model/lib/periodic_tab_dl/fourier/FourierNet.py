from ..base.BaseNet import BaseNet
from ..fourier.FourierBlock import FourierBlock


class FourierNet(BaseNet):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.2,
            output_size: int = 1,
            use_residual: bool = False
    ):
        """
        FourierNet applies Fourier transformations with an MLP base network, integrating an optional residual connection
        for continuous feature enhancement.

        :param input_size: Number of input features.
        :param num_fourier_features: Number of Fourier features to learn in each layer.
        :param num_layers: Number of stacked FourierBlock layers.
        :param compression_dim: Size to compress features between layers; retains input size if not specified.
        :param dropout_prob: Dropout probability applied after normalization layers (0 to 1).
        :param output_size: Size of the network output; >1 indicates classification, 1 indicates regression.
        :param use_residual: If True, applies a residual connection for continuous features.
        """
        # Initialize the Fourier transformation block for processing
        fourier_block = FourierBlock(
            input_size=input_size,
            num_features_per_input=num_fourier_features,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        # Initialize BaseNet with FourierBlock as the processing layer
        super(FourierNet, self).__init__(
            processing_layer=fourier_block,
            output_size=output_size,
            use_residual=use_residual
        )
