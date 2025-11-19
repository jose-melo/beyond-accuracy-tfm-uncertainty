from ..orthogonal_poly.encoders.ChebyshevEncoder import ChebyshevEncoder
from ..orthogonal_poly.encoders.HermiteEncoder import HermiteEncoder
from ..orthogonal_poly.encoders.LaguerreEncoder import LaguerreEncoder
from ..orthogonal_poly.encoders.LegendreEncoder import LegendreEncoder


class OrthogonalPolynomialFactory:
    @staticmethod
    def get_polynomial(polynomial_type: str):
        """
        Factory function to create an instance of OrthogonalPolynomialEncoder of the specified polynomial type.

        :param polynomial_type: Type of orthogonal polynomial ('chebyshev', 'legendre', 'hermite', 'laguerre')
        """
        polynomial_type = polynomial_type.lower()
        if polynomial_type == 'chebyshev':
            return ChebyshevEncoder
        elif polynomial_type == 'legendre':
            return LegendreEncoder
        elif polynomial_type == 'hermite':
            return HermiteEncoder
        elif polynomial_type == 'laguerre':
            return LaguerreEncoder
        else:
            raise ValueError(f"Unsupported polynomial type: {polynomial_type}")
