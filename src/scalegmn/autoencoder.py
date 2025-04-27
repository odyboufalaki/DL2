import torch
import torch.nn as nn
from abc import ABC
from ..data.base_datasets import Batch
from .models import ScaleGMN
from .inr import *

class Decoder(nn.Module, ABC):
    """
    Abstract base class for decoders in ScaleGMN.
    This class defines the interface for all decoders.
    """

    def __init__(self):
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

    def reconstruct_inr_state_dict(
        self,
        params_flatten: torch.Tensor,
        *,
        in_features: int = 2,
        n_layers: int = 3,
        hidden_features: int = 32,
        out_features: int = 1,
        ) -> nn.Module:
        """
        Reconstruct the state dict of an INR model from a flat parameter vector.

        Args:
            flat_params:     1D tensor containing all weights & biases for an INR with this arch.
            in_features:     INR.__init__ in_features
            n_layers:        INR.__init__ n_layers
            hidden_features: INR.__init__ hidden_features
            out_features:    INR.__init__ out_features
            fix_pe:          INR.__init__ fix_pe

        Returns:
            new_sd:         state_dict of the INR model with the same architecture as the original
                            but with the weights and biases replaced by those in flat_params.
        """
        # Instantiate fresh INR
        model = INR(
            in_features=in_features,
            n_layers=n_layers,
            hidden_features=hidden_features,
            out_features=out_features
        )

        # Break out its native params → shapes
        _, init_params = make_functional(model)
        _, shapes = params_to_tensor(init_params)

        new_params_tuple = tensor_to_params(params_flatten, shapes)

        # Build a new state_dict mapping names → tensors
        sd = model.state_dict()
        new_sd = {}
        for (name, orig_tensor), new_tensor in zip(sd.items(), new_params_tuple):
            # ensure the shapes line up
            assert orig_tensor.shape == new_tensor.shape, (
                f"shape mismatch for {name}: "
                f"{orig_tensor.shape} vs {new_tensor.shape}"
            )
            new_sd[name] = new_tensor

        # Load into model
        #model.load_state_dict(new_sd)

        return new_sd
    

class Autoencoder(nn.Module, ABC):
    """
    Abstract base class for autoencoders in ScaleGMN.
    This class defines the interface for all autoencoders.
    """

    def __init__(self):
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