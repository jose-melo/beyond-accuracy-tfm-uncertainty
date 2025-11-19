import torch

from ...orthogonal_poly.OrthogonalPolynomialEncoder import OrthogonalPolynomialEncoder


class ChebyshevEncoder(OrthogonalPolynomialEncoder):
    def generate_polynomials(self, x):
        """
        Generate Chebyshev polynomials up to max_terms.

        :param x: Input tensor of shape [batch_size, input_size, 1].
        :return: Chebyshev polynomials tensor of shape [batch_size, input_size, max_terms].
        """
        # Ensure inputs are within [-1, 1] for numerical stability
        x = torch.clamp(x, -1.0, 1.0)

        T = [torch.ones_like(x), x]  # T0 and T1

        # Recurrence for Chebyshev polynomials
        for n in range(2, self.max_terms):
            T_next = 2 * x * T[-1] - T[-2]
            T.append(T_next)

        # Stack polynomials along the last dimension
        x_cheb = torch.cat(T[:self.max_terms], dim=2)  # Shape: [batch_size, input_size, max_terms]
        return x_cheb
