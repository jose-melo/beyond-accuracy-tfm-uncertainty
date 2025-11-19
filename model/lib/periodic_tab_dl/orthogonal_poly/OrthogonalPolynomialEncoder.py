import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class OrthogonalPolynomialEncoder(nn.Module, ABC):
    def __init__(
            self,
            input_size,
            max_terms,
            num_heads=1,
            kernel_size=5,
            scale=True,
            normalize=True,
            residual=False,
            activation=nn.SiLU(),
    ):
        """
        Base class for Multi-Headed Orthogonal Polynomial Encoder.

        :param input_size: Number of input features.
        :param max_terms: Maximum number of polynomial terms.
        :param num_heads: Number of heads for multi-headed encoding.
        :param kernel_size: Size of the kernel for interaction modulation.
        :param scale: Whether to apply learnable scaling to each input.
        :param normalize: Whether to normalize the final output.
        :param residual: Whether to add residual connections.
        :param activation: Activation function after polynomial transformations.
        """
        super(OrthogonalPolynomialEncoder, self).__init__()
        self.input_size = input_size
        self.max_terms = max_terms
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = scale
        self.normalize = normalize
        self.residual = residual
        self.activation = activation
        self.d_k = 32  # Dimension of the queries and keys
        self.d_v = 32  # Dimension of the values

        # Optional scaling parameter for each input feature
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Separate learnable weights for each head
        self.poly_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(input_size, max_terms))
                for _ in range(num_heads)
            ]
        )

        # Separate learnable kernels for each head
        self.kernels = nn.ParameterList(
            [
                nn.Parameter(torch.randn(input_size, max_terms, kernel_size))
                for _ in range(num_heads)
            ]
        )

        # Output dimension per head
        self.output_dim_per_head = input_size * max_terms * kernel_size  # Adjust as needed

        # Update the output dimension

        # Optional normalization layer
        if self.normalize:
            self.norm_layer = nn.LayerNorm(self.output_dim_per_head * self.num_heads)  # Updated

        # Projection layers for cross-head attention
        self.query_proj = nn.Linear(self.output_dim_per_head, self.d_k)
        self.key_proj = nn.Linear(self.output_dim_per_head, self.d_k)
        self.value_proj = nn.Linear(self.output_dim_per_head, self.d_v)
        self.output_proj = nn.Linear(self.num_heads * self.d_v, self.output_dim_per_head * self.num_heads)

        self.output_dim = max_terms * kernel_size * num_heads

    @abstractmethod
    def generate_polynomials(self, x):
        """
        Generate orthogonal polynomials up to max_terms.

        :param x: Input tensor of shape [batch_size, input_size, 1].
        :return: Tensor of polynomials of shape [batch_size, input_size, max_terms].
        """
        pass

    def forward(self, x):
        """
        Forward pass of the Multi-Headed Orthogonal Polynomial Encoder.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Encoded tensor of shape [batch_size, output_dim].
        """
        batch_size, input_size = x.shape

        # Apply scaling if enabled
        if self.scale:
            x = x * self.scale_param

        # Expand dimensions for polynomial generation
        x_expanded = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]

        # Generate orthogonal polynomials
        x_poly = self.generate_polynomials(x_expanded)  # Shape: [batch_size, input_size, max_terms]

        # Multi-head encoding
        head_outputs = []
        for i in range(self.num_heads):
            # Apply the learnable kernel for this head
            x_head = torch.einsum(
                'bim,imk->bik', x_poly, self.kernels[i]
            )  # Shape: [batch_size, input_size, kernel_size]

            # Apply learnable weights for this head
            weights = self.poly_weights[i].unsqueeze(0)  # Shape: [1, input_size, max_terms]
            weights = weights.unsqueeze(-1)  # Shape: [1, input_size, max_terms, 1]
            x_head = x_head.unsqueeze(2)  # Shape: [batch_size, input_size, 1, kernel_size]
            x_head_weighted = x_head * weights  # Broadcasting over batch_size

            # Apply residual connections if enabled
            if self.residual:
                x_residual = x.unsqueeze(-1).unsqueeze(-1)  # [batch_size, input_size, 1, 1]
                x_head_weighted = x_head_weighted + x_residual

            # Apply activation function
            if self.activation:
                x_head_weighted = self.activation(x_head_weighted)

            # Flatten for this head and collect
            x_head_flat = x_head_weighted.reshape(batch_size, -1)  # Use reshape instead of view
            head_outputs.append(x_head_flat.unsqueeze(1))  # Add head dimension

        # Stack head outputs: Shape [batch_size, num_heads, output_dim_per_head]
        x_heads = torch.cat(head_outputs, dim=1)

        # Apply cross-head attention
        x_multi_head = self.cross_head_attention(x_heads)  # Output: [batch_size, output_dim]

        # Normalize if enabled
        if self.normalize:
            x_multi_head = self.norm_layer(x_multi_head)

        return x_multi_head

    def cross_head_attention(self, x_heads):
        """
        Apply self-attention across heads.

        :param x_heads: Tensor of shape [batch_size, num_heads, output_dim_per_head]
        :return: Tensor of shape [batch_size, output_dim]
        """
        batch_size, num_heads, output_dim_per_head = x_heads.shape

        # Compute query, key, and value projections
        # Reshape to merge batch and num_heads dimensions for projection
        x_heads_reshaped = x_heads.reshape(batch_size * num_heads, output_dim_per_head)  # Use reshape

        query = self.query_proj(x_heads_reshaped)  # Shape: [batch_size * num_heads, d_k]
        key = self.key_proj(x_heads_reshaped)  # Shape: [batch_size * num_heads, d_k]
        value = self.value_proj(x_heads_reshaped)  # Shape: [batch_size * num_heads, d_v]

        # Reshape back to [batch_size, num_heads, d_k or d_v]
        query = query.reshape(batch_size, num_heads, self.d_k)
        key = key.reshape(batch_size, num_heads, self.d_k)
        value = value.reshape(batch_size, num_heads, self.d_v)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_k)
        # scores shape: [batch_size, num_heads, num_heads]

        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights shape: [batch_size, num_heads, num_heads]

        # Compute attended values
        attended = torch.matmul(attn_weights, value)  # Shape: [batch_size, num_heads, d_v]

        # Concatenate heads and project
        attended = attended.reshape(batch_size, -1)  # Use reshape instead of view
        output = self.output_proj(attended)  # Final linear projection to [batch_size, output_dim]

        return output
