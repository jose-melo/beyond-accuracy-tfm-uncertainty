from typing import Optional, Dict

from ..base.BaseBlock import BaseBlock
from ...orthogonal_poly.OrthogonalPolynomialFactory import \
    OrthogonalPolynomialFactory


class OrthogonalPolynomialBlock(BaseBlock):
    def __init__(
            self,
            input_size: int,
            max_poly_terms: int,
            num_layers: int,
            compression_dim: Optional[int],
            dropout_prob: float,
            polynomial_type: str
    ):
        """
        OrthogonalPolynomialBlock applies a series of orthogonal polynomial encoders to transform input features.
        Supports various polynomial types (e.g., Chebyshev, Legendre) with optional compression, batch normalization,
        and dropout between layers.

        :param input_size: Number of input features.
        :param max_poly_terms: Number of terms in the polynomial for each layer.
        :param num_layers: Number of stacked encoder layers.
        :param compression_dim: Target size for feature compression between layers; if None, uses the input size.
        :param dropout_prob: Dropout probability after each normalization layer.
        :param polynomial_type: Type of orthogonal polynomial to use (e.g., 'chebyshev', 'legendre').
        """
        encoder_class = OrthogonalPolynomialFactory.get_polynomial(polynomial_type)
        encoder_params: Dict[str, int] = {
            "max_terms": max_poly_terms
        }

        super().__init__(
            input_size=input_size,
            encoder=encoder_class,
            encoder_params=encoder_params,
            num_layers=num_layers,
            compression_dim=compression_dim or input_size,
            dropout_prob=dropout_prob,
        )
