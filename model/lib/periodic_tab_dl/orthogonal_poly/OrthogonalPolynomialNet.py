from typing import Optional

from ..base.BaseNet import BaseNet
from ...orthogonal_poly.OrthogonalPolynomialBlock import \
    OrthogonalPolynomialBlock
from src.config import POLY_TYPE


class OrthogonalPolynomialNet(BaseNet):
    def __init__(
            self,
            input_size: int,
            max_poly_terms: int,
            num_layers: int = 1,
            compression_dim: Optional[int] = None,
            dropout_prob: float = 0.2,
            output_size: int = 1,
            use_residual: bool = True,
            polynomial_type: str = POLY_TYPE
    ):
        """
        OrthogonalPolynomialNet constructs a neural network that applies a series of orthogonal polynomial transformations
        followed by an MLP-based network. Includes an optional residual connection for enhanced stability.

        :param input_size: Number of input features.
        :param max_poly_terms: Number of polynomial terms to use in each transformation layer.
        :param num_layers: Number of transformation layers to stack.
        :param compression_dim: Dimension to compress features to between layers; if None, defaults to input size.
        :param dropout_prob: Probability of dropout applied after normalization layers (0 to 1).
        :param output_size: Size of the final network output; use 1 for regression, >1 for classification.
        :param use_residual: Whether to include a residual connection between input and output.
        :param polynomial_type: Type of orthogonal polynomial to use (e.g., 'chebyshev', 'legendre').
        """
        # Initialize the orthogonal polynomial transformation block
        orthogonal_poly_block = OrthogonalPolynomialBlock(
            input_size=input_size,
            max_poly_terms=max_poly_terms,
            num_layers=num_layers,
            compression_dim=compression_dim,
            dropout_prob=dropout_prob,
            polynomial_type=polynomial_type
        )

        # Initialize the BaseNet with the processing layer and other parameters
        super().__init__(
            processing_layer=orthogonal_poly_block,
            output_size=output_size,
            use_residual=use_residual
        )
