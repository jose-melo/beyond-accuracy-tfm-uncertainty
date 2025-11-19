import torch
from torch import nn

from ..fourier.FourierEncoder import FourierEncoder
from ..orthogonal_poly.OrthogonalPolynomialFactory import OrthogonalPolynomialFactory

class AutoPNPEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            use_cross_attention: bool = False,
            num_heads: int = 4,
            poly_type: str = "chebyshev",
    ):
        """
        AutoPNPNet that integrates Fourier and Orthogonal Polynomial layers with optional cross attention.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param max_poly_terms: Number of polynomial terms.
        :param use_cross_attention: Whether to apply cross attention between features.
        :param num_heads: Number of attention heads (used if cross attention is enabled).
        """
        super(AutoPNPEncoder, self).__init__()

        self.use_cross_attention = use_cross_attention

        # Fourier and Orthogonal Polynomial layers
        self.fourier_layer = FourierEncoder(input_size, num_fourier_features)
        self.orthogonal_poly_layer = OrthogonalPolynomialFactory.get_polynomial(poly_type)(
            input_size, max_poly_terms
        )

        # Define feature dimensions
        self.fourier_dim = self.fourier_layer.output_dim
        self.poly_dim = self.orthogonal_poly_layer.output_dim

        # Embedding dimension for attention or final projection
        self.embed_dim = 128  # You can adjust this as needed

        if self.use_cross_attention:
            # Linear projections to common embedding space
            self.fourier_proj = nn.Linear(self.fourier_dim * input_size, self.embed_dim)
            self.poly_proj = nn.Linear(self.poly_dim * input_size, self.embed_dim)

            # Cross attention layer
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
            )

            self.linear = nn.Linear(self.embed_dim, self.embed_dim * input_size)

            # Optional activation and normalization
            self.activation = nn.ReLU()
            self.norm_layer = nn.LayerNorm(self.embed_dim * input_size)

            self.output_dim = self.embed_dim
        else:
            self.output_dim = self.fourier_dim + self.poly_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional cross attention between Fourier and polynomial features.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Tensor of shape [batch_size, output_dim].
        """
        # Apply Fourier and polynomial transformations
        x_fourier = self.fourier_layer(x)  # Shape: [batch_size, fourier_dim]
        x_poly = self.orthogonal_poly_layer(x)  # Shape: [batch_size, poly_dim]

        if self.use_cross_attention:
            # Project to common embedding dimension
            x_fourier_proj = self.fourier_proj(x_fourier)  # Shape: [batch_size, embed_dim]
            x_poly_proj = self.poly_proj(x_poly)  # Shape: [batch_size, embed_dim]

            # Reshape for attention: [batch_size, seq_len, embed_dim]
            x_fourier_proj = x_fourier_proj.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
            x_poly_proj = x_poly_proj.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]

            # Concatenate along the sequence dimension
            combined = torch.cat((x_fourier_proj, x_poly_proj), dim=1)  # Shape: [batch_size, 2, embed_dim]

            # Apply cross attention
            attn_output, _ = self.cross_attention(
                query=combined, key=combined, value=combined
            )  # Shape: [batch_size, 2, embed_dim]

            attn_output = self.linear(attn_output) # Shape: [batch_size, 2, embed_dim * input_size]

            # Optionally, apply activation and normalization
            attn_output = self.activation(attn_output)
            attn_output = self.norm_layer(attn_output)

            # Pool the attended features (e.g., by averaging)
            attn_output = attn_output.mean(dim=1)  # Shape: [batch_size, embed_dim]

            return attn_output  # Shape: [batch_size, embed_dim]
        else:
            # Simply concatenate the features without attention
            return torch.cat([x_fourier, x_poly], dim=1)  # Shape: [batch_size, fourier_dim + poly_dim]
