from torch import nn, Tensor

from ..base.MLP import MLP


class BaseNet(nn.Module):
    def __init__(
            self,
            processing_layer: nn.Module,
            output_size: int,
            use_residual: bool
    ):
        """
        BaseNet combines a processing layer (e.g., a Chebyshev transformation) with an MLP layer for flexible output.
        Optionally includes a residual connection for enhanced learning stability.

        :param processing_layer: The layer for initial feature transformation.
        :param output_size: The size of the output; if greater than 1, this indicates classification; otherwise, regression.
        :param use_residual: Whether to apply a residual connection between the input and processed output.
        """
        super(BaseNet, self).__init__()

        self.processing_layer = processing_layer
        self.use_residual = use_residual
        self.mlp = MLP(input_size=self.processing_layer.output_dim, output_size=output_size)

    def forward(self, x: Tensor) -> Tensor:
        # Apply the processing layer (e.g., Chebyshev transformation)
        x_processed = self.processing_layer(x)

        # Apply the MLP layer for final output
        out = self.mlp(x_processed)

        # Add residual connection if enabled and shapes match
        if self.use_residual and x.shape == out.shape:
            out += x

        return out
