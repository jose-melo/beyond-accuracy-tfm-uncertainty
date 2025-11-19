import torch

from ...orthogonal_poly.OrthogonalPolynomialEncoder import OrthogonalPolynomialEncoder


class LaguerreEncoder(OrthogonalPolynomialEncoder):
    def generate_polynomials(self, x):
        """
        Generate Laguerre polynomials up to max_terms.

        :param x: Input tensor of shape [batch_size, input_size, 1].
        :return: Laguerre polynomials tensor of shape [batch_size, input_size, max_terms].
        """
        # Ensure inputs are non-negative
        x = torch.clamp(x, min=0.0)

        L = [torch.ones_like(x), 1 - x]  # L0 and L1

        # Recurrence for Laguerre polynomials
        for n in range(2, self.max_terms):
            n_float = torch.tensor(n, dtype=x.dtype, device=x.device)
            L_next = ((2 * n_float - 1 - x) * L[-1] - (n_float - 1) * L[-2]) / n_float
            L.append(L_next)

        # Stack polynomials along the last dimension
        x_laguerre = torch.cat(L[:self.max_terms], dim=2)  # Shape: [batch_size, input_size, max_terms]
        return x_laguerre
