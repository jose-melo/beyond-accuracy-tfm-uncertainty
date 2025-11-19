import torch

from ...orthogonal_poly.OrthogonalPolynomialEncoder import OrthogonalPolynomialEncoder


class HermiteEncoder(OrthogonalPolynomialEncoder):
    def generate_polynomials(self, x):
        """
        Generate Hermite polynomials up to max_terms.

        :param x: Input tensor of shape [batch_size, input_size, 1].
        :return: Hermite polynomials tensor of shape [batch_size, input_size, max_terms].
        """
        H = [torch.ones_like(x), 2 * x]  # H0 and H1

        # Recurrence for Hermite polynomials
        for n in range(2, self.max_terms):
            H_next = 2 * x * H[-1] - 2 * (n - 1) * H[-2]
            H.append(H_next)

        # Stack polynomials along the last dimension
        x_hermite = torch.cat(H[:self.max_terms], dim=2)  # Shape: [batch_size, input_size, max_terms]
        return x_hermite
