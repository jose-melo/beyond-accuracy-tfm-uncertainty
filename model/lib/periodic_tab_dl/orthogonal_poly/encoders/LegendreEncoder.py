import torch

from ...orthogonal_poly.OrthogonalPolynomialEncoder import OrthogonalPolynomialEncoder


class LegendreEncoder(OrthogonalPolynomialEncoder):
    def generate_polynomials(self, x):
        """
        Generate Legendre polynomials up to max_terms.

        :param x: Input tensor of shape [batch_size, input_size, 1].
        :return: Legendre polynomials tensor of shape [batch_size, input_size, max_terms].
        """
        # Ensure inputs are within [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        P = [torch.ones_like(x), x]  # P0 and P1

        # Recurrence for Legendre polynomials
        for n in range(2, self.max_terms):
            n_float = torch.tensor(n, dtype=x.dtype, device=x.device)
            term1 = ((2.0 * n_float - 1.0) / n_float) * x * P[-1]
            term2 = ((n_float - 1.0) / n_float) * P[-2]
            P_next = term1 - term2
            P.append(P_next)

        # Stack polynomials along the last dimension
        x_legendre = torch.cat(P[:self.max_terms], dim=2)  # Shape: [batch_size, input_size, max_terms]
        return x_legendre
