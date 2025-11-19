import torch
from torch import nn
import math

class FourierEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_features_per_input,
            kernel_size=1,
            scale_input=True,
            init_frequency_range=(0.0, 5.0),
            use_feature_scaling=True,
            use_convolution=True,
            activation='sin_cos',
            frequency_init='log',
            frequency_scale=None,
            use_phase_shift=True,
            use_learnable_amplitude=True,
            use_rff=True,  # New parameter to toggle RFF
            rff_sigma=1.0   # Bandwidth parameter for RFF
    ):
        """
        Fourier Feature Encoder Layer with optional Random Fourier Features (RFF).

        :param input_size: Number of input features.
        :param num_features_per_input: Number of Fourier features per input feature.
        :param kernel_size: Size of the kernel for local feature extraction.
        :param scale_input: Whether to scale the input with a learnable per-feature scaling parameter.
        :param init_frequency_range: Range for the initialization of the frequency matrix B.
        :param use_feature_scaling: Whether to add learnable scaling for the Fourier features.
        :param use_convolution: Whether to apply a convolution over the input features.
        :param activation: Activation function to use ('sin_cos', 'sin', 'cos', 'tanh').
        :param frequency_init: Method for initializing frequencies ('uniform', 'normal', 'log').
        :param frequency_scale: Scaling factor for frequencies (if None, uses init_frequency_range).
        :param use_phase_shift: Whether to use learnable phase shifts in the Fourier features.
        :param use_learnable_amplitude: Whether to use learnable amplitude scaling in the Fourier features.
        :param use_rff: Whether to use Random Fourier Features.
        :param rff_sigma: Bandwidth parameter for RFF (standard deviation of the normal distribution).
        """
        super(FourierEncoder, self).__init__()
        self.input_size = input_size
        self.num_features_per_input = num_features_per_input
        self.scale_input = scale_input
        self.use_feature_scaling = use_feature_scaling
        self.use_convolution = use_convolution
        self.activation = activation
        self.frequency_init = frequency_init
        self.use_phase_shift = use_phase_shift
        self.use_learnable_amplitude = use_learnable_amplitude
        self.use_rff = use_rff
        self.rff_sigma = rff_sigma

        if self.use_convolution:
            # Learnable kernel for localized feature extraction (1D convolution)
            self.kernel = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True  # Enable bias for additional learnable parameters
            )

        if self.use_rff:
            # Random Fourier Features: Initialize frequencies from a normal distribution
            self.register_buffer('B', torch.empty(input_size, num_features_per_input))
            self._initialize_rff_frequencies()
        else:
            # Learnable frequency matrix B
            self.B = nn.Parameter(torch.empty(input_size, num_features_per_input))
            self._initialize_frequencies(init_frequency_range, frequency_scale)

        # Optional per-feature scaling of input
        if self.scale_input:
            self.scale_param = nn.Parameter(torch.ones(input_size))

        # Optional learnable phase shifts
        if self.use_phase_shift:
            self.phi = nn.Parameter(torch.zeros(input_size, num_features_per_input))

        # Optional learnable amplitude scaling
        if self.use_learnable_amplitude:
            self.amplitude = nn.Parameter(torch.ones(input_size, num_features_per_input))

        # Optional scaling of Fourier features after transformation
        if self.use_feature_scaling:
            feature_dim = num_features_per_input * (2 if self.activation == 'sin_cos' else 1)
            self.feature_scaling = nn.Parameter(torch.ones(input_size, feature_dim))

        # Compute the Fourier feature dimension
        self.output_dim = num_features_per_input * (2 if self.activation == 'sin_cos' else 1)

    def _initialize_rff_frequencies(self):
        """
        Initializes the frequency matrix B for Random Fourier Features.
        Frequencies are drawn from a normal distribution with mean 0 and standard deviation 1/sigma.
        """
        with torch.no_grad():
            self.B.normal_(mean=0.0, std=1.0 / self.rff_sigma)

    def _initialize_frequencies(self, init_range, frequency_scale):
        if self.frequency_init == 'uniform':
            nn.init.uniform_(self.B, *init_range)
        elif self.frequency_init == 'normal':
            nn.init.normal_(self.B, mean=0.0, std=init_range[1])
        elif self.frequency_init == 'log':
            log_min_freq = torch.log(torch.tensor(init_range[0] + 1e-8))
            log_max_freq = torch.log(torch.tensor(init_range[1] + 1e-8))
            nn.init.uniform_(self.B, log_min_freq.item(), log_max_freq.item())
            self.B.data = torch.exp(self.B.data)
        else:
            raise ValueError(f"Invalid frequency_init: {self.frequency_init}")

        if frequency_scale is not None:
            self.B.data *= frequency_scale

    def forward(self, x):
        """
        Forward pass of the FourierEncoder.

        :param x: Input tensor of shape [batch_size, input_size].
        :return: Encoded tensor of shape [batch_size, output_dim].
        """
        # Input scaling (if enabled)
        if self.scale_input:
            x = x * self.scale_param  # Shape: [batch_size, input_size]

        # Optional convolutional layer
        if self.use_convolution:
            x = x.unsqueeze(1)  # Shape: [batch_size, 1, input_size]
            x = self.kernel(x)  # Shape: [batch_size, 1, input_size]
            x = x.squeeze(1)    # Shape: [batch_size, input_size]

        # Prepare for Fourier transformation
        x_expanded = x.unsqueeze(2)  # Shape: [batch_size, input_size, 1]
        if self.use_rff:
            B_expanded = self.B.unsqueeze(0).detach()  # [1, input_size, num_features_per_input]
        else:
            B_expanded = self.B.unsqueeze(0)  # [1, input_size, num_features_per_input]

        # Apply learnable amplitude (if enabled)
        if self.use_learnable_amplitude:
            amplitude = self.amplitude.unsqueeze(0)  # [1, input_size, num_features_per_input]
        else:
            amplitude = 1.0

        # Apply learnable phase shift (if enabled)
        if self.use_phase_shift:
            phi = self.phi.unsqueeze(0)  # [1, input_size, num_features_per_input]
        else:
            phi = 0.0

        # Project the input through frequencies with amplitude and phase shift
        x_proj = 2 * math.pi * x_expanded * B_expanded + phi  # [batch_size, input_size, num_features_per_input]

        # Apply activation function
        if self.activation == 'sin_cos':
            x_sin = amplitude * torch.sin(x_proj)
            x_cos = amplitude * torch.cos(x_proj)
            x_fourier = torch.cat([x_sin, x_cos], dim=2)
        elif self.activation == 'sin':
            x_fourier = amplitude * torch.sin(x_proj)
        elif self.activation == 'cos':
            x_fourier = amplitude * torch.cos(x_proj)
        elif self.activation == 'tanh':
            x_fourier = amplitude * torch.tanh(x_proj)
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        # Optional scaling of Fourier features
        if self.use_feature_scaling:
            x_fourier = x_fourier * self.feature_scaling.unsqueeze(0)

        # Flatten the Fourier features
        x_fourier = x_fourier.view(x.size(0), -1)  # [batch_size, fourier_feature_dim]

        return x_fourier
