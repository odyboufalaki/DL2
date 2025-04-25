import torch
import torch.nn as nn
from nfn.common import WeightSpaceFeatures
import math
from src.data.base_datasets import Batch
from ..scalegmn.inr import INR


def select_criterion(criterion: str, criterion_args: dict) -> nn.Module:
    _map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(**criterion_args),
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'ReconstructionLoss': nn.MSELoss(),  # TODO fix
    }
    if criterion not in _map.keys():
        raise NotImplementedError
    else:
        return _map[criterion]


def L2_distance(x, x_hat, batch_size=1):
    """
    Compute L2 Loss between the inputs.
    """

    if isinstance(x, torch.Tensor) and isinstance(x_hat, torch.Tensor):
        loss = torch.square(x - x_hat).sum() / batch_size

    elif isinstance(x, dict) and isinstance(x_hat, dict):
        loss = 0
        for key in x:
            loss += torch.square(x[key] - x_hat[key]).sum()

        loss = loss / batch_size

    elif isinstance(x, WeightSpaceFeatures):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, Batch):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, tuple):
        # problem here w1 and w2 are 3D not 4D
        #diff_weights = sum([torch.sum(torch.square(w1 - w2), (1,2,3)) for w1, w2 in zip(x_hat[0], x[0])])
        #diff_biases = sum([torch.sum(torch.square(b1 - b2), (1,2)) for b1, b2 in zip(x_hat[1], x[1])])
        diff_weights = sum([torch.sum(torch.square(w1 - w2), (1,2)) for w1, w2 in zip(x_hat[0], x[0])])
        diff_biases = sum([torch.sum(torch.square(b1 - b2), (1)) for b1, b2 in zip(x_hat[1], x[1])])
        loss = (diff_weights + diff_biases) / batch_size
    else:
        raise NotImplemented

    return loss


def reconstruction_inr_loss(
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
    Compute MSE between INR reconstruction and a target 28x28 image.

    Args:
        flat_params:   (P,) tensor of all INR weights & biases.
        target_img:    (28,28) or (1,28,28) tensor with pixel values ∈ [0,1].
        reconstruct_fn:callable, e.g. your `reconstruct_inr_model`.
        in_features, n_layers, hidden_features, out_features, pe_features, fix_pe:
                    INR hyper-parameters (must match how flat_params was produced).

    Returns:
        A 0-dim tensor = mean squared error over the 28x28 grid.
    """
    # rebuild a model with those params
    model = reconstruct_fn(
        flat_params,
        in_features=in_features,
        n_layers=n_layers,
        hidden_features=hidden_features,
        out_features=out_features,
    )

    # make a (1, 28*28, 2) coord tensor and push through
    coords = make_coordinates((28, 28), bs=1).to(flat_params.device)  # → (1, 784, 2)
    pred = model(coords)                                             # → (1, 784, 1)
    pred = pred.view(1, 28, 28)                                      # → (1, 28, 28)

    # if target is (28,28) make it (1,28,28)
    if target_img.ndim == 2:
        target_img = target_img.unsqueeze(0)

    # compute MSE
    return F.mse_loss(pred, target_img)