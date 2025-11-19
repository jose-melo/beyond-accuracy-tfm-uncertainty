from ..base.BaseTabNet import BaseTabNet
from ..fourier.FourierBlock import FourierBlock


class TabFourierNet(BaseTabNet):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            compression_dim: int = None,
            dropout_prob: float = 0.2,
            use_residual: bool = True
    ):
        """
        TabFourierNet integrates Fourier transformations for continuous features with an MLP for categorical features,
        supporting an optional residual connection on continuous features.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features to learn for each continuous input feature.
        :param hidden_size: Size of hidden layers for processing categorical features.
        :param output_size: Size of the network's output; >1 for classification, 1 for regression.
        :param use_residual: If True, applies a residual connection to continuous features.
        """
        # Initialize Fourier transformation layer for continuous features
        fourier_layer = FourierBlock(
            input_size=continuous_input_size,
            num_features_per_input=num_fourier_features,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob
        )

        # Initialize BaseTabNet with the Fourier layer and categorical MLP
        super(TabFourierNet, self).__init__(
            processing_layer=fourier_layer,
            categorical_input_size=categorical_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_residual=use_residual
        )
