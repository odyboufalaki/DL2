import torch
import torch.nn as nn
from abc import ABC
from models import ScaleGMN

class Decoder(nn.Module, ABC):
    """
    Abstract base class for decoders in ScaleGMN.
    This class defines the interface for all decoders.
    """

    def __init__(self, model_args, **kwargs):
        self.net = None
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Tensor of shape [B, latent_dim]
        returns: Tensor of shape [B, output_dim]
        """
        return self.net(z)


class MLPDecoder(Decoder):
    """
    Generic MLP-based decoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.input_dim = model_args['d_input']
        self.hidden_dims = model_args['d_hidden']
        self.num_layers = len(self.hidden_dims)
        self.data_layer_layout = model_args['data_layer_layout']
        self.output_dim = sum([
            (self.data_layer_layout[i_layer] + 1) * self.data_layer_layout[i_layer + 1]
            for i_layer in range(len(self.data_layer_layout) - 1)
        ])

        layers = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Tensor of shape [B, latent_dim]
        returns: Tensor of shape [B, output_dim]
        """
        return self.net(z)
    
    def reconstruct_mlp(parameters, mlp_dimensions):
        weights = []
        biases = []
        start_idx = 0

        for i in range(len(mlp_dimensions) - 1):
            weight_size = mlp_dimensions[i] * mlp_dimensions[i + 1]
            bias_size = mlp_dimensions[i + 1]

            weight = parameters[start_idx:start_idx + weight_size].view(
                mlp_dimensions[i + 1], mlp_dimensions[i]
            )
            start_idx += weight_size

            bias = parameters[start_idx:start_idx + bias_size]
            start_idx += bias_size

            weights.append(weight)
            biases.append(bias)

        sequential_layers = []
        for i in range(len(mlp_dimensions) - 1):
            layer = nn.Linear(mlp_dimensions[i], mlp_dimensions[i + 1])
            layer.weight = nn.Parameter(weights[i])
            layer.bias = nn.Parameter(biases[i])
            sequential_layers.append(layer)
            if i < len(mlp_dimensions) - 2:  # Add activation for all but the last layer
            sequential_layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*sequential_layers)
        return weights, biases


class Autoencoder(nn.Module, ABC):
    """
    Abstract base class for autoencoders in ScaleGMN.
    This class defines the interface for all autoencoders.
    """

    def __init__(self, model_args, **kwargs):
        self.encoder = None
        self.decoder = None
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


class MLPAutoencoder(Autoencoder):
    """
    Generic MLP-based autoencoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.encoder = ScaleGMN(model_args["scalegmn_args"], **kwargs)
        self.decoder = MLPDecoder(model_args["decoder_args"], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == "__main__":
    # Example usage
    decoder = MLPDecoder(latent_dim=128, output_dim=784)
    z = torch.randn(32, 128)  # Batch of 32 samples
    output = decoder(z)
    print(output.shape)  # Should be [32, 784]