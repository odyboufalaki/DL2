import torch
import torch.nn as nn
from abc import ABC
from models import ScaleGMN
from inr import *

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
    
    def reconstruct_inr_model(
        flat_params: torch.Tensor,
        *,
        in_features: int = 2,
        n_layers: int = 3,
        hidden_features: int = 32,
        out_features: int = 1,
        pe_features: Optional[int] = None,
        fix_pe: bool = True,
        ) -> nn.Module:
        """
        Reconstruct an INR model from a flat parameter vector.

        Args:
            flat_params:     1D tensor containing all weights & biases for an INR with this arch.
            in_features:     INR.__init__ in_features
            n_layers:        INR.__init__ n_layers
            hidden_features: INR.__init__ hidden_features
            out_features:    INR.__init__ out_features
            pe_features:     INR.__init__ pe_features
            fix_pe:          INR.__init__ fix_pe

        Returns:
            A new INR instance whose parameters are set to flat_params.
        """
        # 1. Instantiate fresh INR
        model = INR(
            in_features=in_features,
            n_layers=n_layers,
            hidden_features=hidden_features,
            out_features=out_features
        )

        # 2. Break out its native params → shapes
        fmodel, init_params = make_functional(model)
        _, shapes = params_to_tensor(init_params)

        # 3. Slice flat_params into a tuple of parameter tensors
        new_params_tuple = tensor_to_params(flat_params, shapes)

        # 4. Build a new state_dict mapping names → tensors
        sd = model.state_dict()
        new_sd = {}
        for (name, orig_tensor), new_tensor in zip(sd.items(), new_params_tuple):
            # ensure the shapes line up
            assert orig_tensor.shape == new_tensor.shape, (
                f"shape mismatch for {name}: "
                f"{orig_tensor.shape} vs {new_tensor.shape}"
            )
            new_sd[name] = new_tensor

        # 5. Load into model
        model.load_state_dict(new_sd)

        return model
    

    def inr_mse(
    flat_params: torch.Tensor,
    target_img: torch.Tensor,
    *,
    reconstruct_fn,        # your `reconstruct_inr_model` or similar factory
    in_features: int = 2,
    n_layers: int = 3,
    hidden_features: int = 32,
    out_features: int = 1,
    pe_features=None,
    fix_pe=True,
    ) -> torch.Tensor:
        """
        Compute MSE between INR reconstruction and a target 28×28 image.

        Args:
            flat_params:   (P,) tensor of all INR weights & biases.
            target_img:    (28,28) or (1,28,28) tensor with pixel values ∈ [0,1].
            reconstruct_fn:callable, e.g. your `reconstruct_inr_model`.
            in_features, n_layers, hidden_features, out_features, pe_features, fix_pe:
                        INR hyper-parameters (must match how flat_params was produced).

        Returns:
            A 0-dim tensor = mean squared error over the 28×28 grid.
        """
        # 1. rebuild a model with those params
        model = reconstruct_fn(
            flat_params,
            in_features=in_features,
            n_layers=n_layers,
            hidden_features=hidden_features,
            out_features=out_features,
        )

        # 2. make a (1, 28*28, 2) coord tensor and push through
        coords = make_coordinates((28, 28), bs=1).to(flat_params.device)  # → (1, 784, 2)
        pred = model(coords)                                             # → (1, 784, 1)
        pred = pred.view(1, 28, 28)                                      # → (1, 28, 28)

        # 3. if target is (28,28) make it (1,28,28)
        if target_img.ndim == 2:
            target_img = target_img.unsqueeze(0)

        # 4. compute MSE
        return F.mse_loss(pred, target_img)
    

    def batch_inr_mse(
    flat_params_batch: torch.Tensor,   # (B, P)
    target_imgs: torch.Tensor,         # (B, 28, 28) or (B,1,28,28)
    reconstruct_fn,                    # same as above
    **inr_kwargs
    ) -> torch.Tensor:
        losses = []
        for fp, tgt in zip(flat_params_batch, target_imgs):
            losses.append(inr_mse(fp, tgt, reconstruct_fn=reconstruct_fn, **inr_kwargs))
        return torch.stack(losses).mean()  # overall mean across batch



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