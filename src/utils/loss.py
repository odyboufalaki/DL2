import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from nfn.common import WeightSpaceFeatures
import math
from src.data.base_datasets import Batch
from ..scalegmn.inr import INR, make_coordinates


def select_criterion(criterion: str, criterion_args: dict) -> nn.Module:
    _map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(**criterion_args),
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'ReconstructionLoss': weighted_mse_loss,  # Allows for weighted loss,
        'VaeLoss': variational_loss,
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

def variational_loss(input: torch.Tensor, target: torch.Tensor,
                       mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # reconstruction loss
        recon_loss = F.mse_loss(input, target)
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
        # average over batch
        kld = torch.mean(kld)
        # Total loss is the one to optimize
        loss = recon_loss + kld
        return loss


def weighted_mse_loss(input, target, weight=None):
    """
    Compute a weighted mean squared error loss.
    """
    if weight is not None:
        return torch.mean(weight * torch.square(input - target))
    else:
        return mse_loss(input, target)