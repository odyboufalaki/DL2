import torch
import torch.nn as nn
from abc import ABC


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
        self.num_layers = model_args['num_layers']
        self.hidden_dims = model_args['d_hidden']
        self.data_layer_layout = model_args['data_layer_layout']
        self.output_dim = sum(self.data_layer_layout)

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


if __name__ == "__main__":
    # Example usage
    decoder = MLPDecoder(latent_dim=128, hidden_dims=[256, 512], output_dim=784)
    z = torch.randn(32, 128)  # Batch of 32 samples
    output = decoder(z)
    print(output.shape)  # Should be [32, 784]