from typing import Optional

from ..base.BaseTabNet import BaseTabNet
from ...orthogonal_poly.OrthogonalPolynomialBlock import \
    OrthogonalPolynomialBlock
from src.config import POLY_TYPE


class TabOrthogonalPolynomialNet(BaseTabNet):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            max_poly_terms: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            compression_dim: Optional[int] = 1,
            dropout_prob: float = 0.2,
            use_residual: bool = True,
            polynomial_type: str = POLY_TYPE
    ):
        """
        TabOrthogonalPolynomialNet integrates orthogonal polynomial transformations for continuous features with an MLP
        for categorical features, with an optional residual connection for enhanced stability.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param max_poly_terms: Number of polynomial terms to use in OrthogonalPolynomialBlock.
        :param hidden_size: Size of hidden layers for processing categorical features.
        :param output_size: Size of the network output; use 1 for regression and >1 for classification.
        :param num_layers: Number of transformation layers to stack in OrthogonalPolynomialBlock.
        :param compression_dim: Target dimension for feature compression between layers; defaults to input size if None.
        :param dropout_prob: Probability of dropout after each normalization layer (range: 0 to 1).
        :param use_residual: Whether to apply a residual connection to the continuous feature transformation.
        :param polynomial_type: Type of orthogonal polynomial to use (e.g., 'chebyshev', 'legendre').
        """
        # Initialize the OrthogonalPolynomialBlock for processing continuous features
        orthogonal_poly_block = OrthogonalPolynomialBlock(
            input_size=continuous_input_size,
            max_poly_terms=max_poly_terms,
            num_layers=num_layers,
            compression_dim=compression_dim or continuous_input_size,
            dropout_prob=dropout_prob,
            polynomial_type=polynomial_type
        )

        # Initialize the BaseTabNet with OrthogonalPolynomialBlock for continuous features and MLP for categorical features
        super().__init__(
            processing_layer=orthogonal_poly_block,
            categorical_input_size=categorical_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_residual=use_residual
        )
