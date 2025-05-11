import torch

import argparse
import os
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import Callable, Dict, Tuple

Tensor = torch.Tensor
StateDict = Dict[str, Tensor]
Transform = Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]

from src.utils.helpers import set_seed
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.inr import INR
from src.data.base_datasets import Batch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--outdir", type=str, default="latent/resources/manifold_ablation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------
# Group transformation functions
def _row_sign_flip(
    w: Tensor, b: Tensor, w_next: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Default ⇄ sign-flip transformation that keeps the MLP function unchanged.
    Works on (out, in) → (out) → (next_out, out) tensors.
    """
    flips = torch.randint(0, 2, (w.size(0),), device=w.device, dtype=w.dtype) * 2 - 1
    w = flips.view(-1, 1) * w
    b = flips * b
    w_next = w_next * flips.view(1, -1)
    return w, b, w_next


def _transform_layer(
    sd: StateDict,
    layer_idx: int,
    transform: Transform = _row_sign_flip,
    prefix: str = "seq.",
) -> StateDict:
    """
    Apply `transform` to layer `layer_idx` (and its successor) inside a
    *Sequential*-style state-dict and return the mutated copy.

    Parameters
    ----------
    sd          : the state-dict (is shallow-copied for safety)
    layer_idx   : 0-based index of the layer whose (W, b) you want to modify
    transform   : fn( weight, bias, next_weight ) -> (W', b', next_W')
    prefix      : key prefix used in the state-dict (default “seq.”)

    Notes
    -----
    • The last layer cannot be transformed with this flip because it has
      no outgoing weight matrix; trying to do so raises ValueError.
    • Works equally for CPU or CUDA tensors.
    """
    w_key = f"{prefix}{layer_idx}.weight"
    b_key = f"{prefix}{layer_idx}.bias"
    wn_key = f"{prefix}{layer_idx+1}.weight"

    if wn_key not in sd:
        raise ValueError(f"layer {layer_idx} is last layer — no next_weight to adjust")

    # copy to avoid in-place edits leaking outside
    new_sd = sd.copy()

    w, b, w_next = (new_sd[w_key], new_sd[b_key], new_sd[wn_key])
    w, b, w_next = transform(w, b, w_next)

    new_sd[w_key] = w
    new_sd[b_key] = b
    new_sd[wn_key] = w_next
    return new_sd


def get_augmented_dataset(
    device: torch.device,
    inr_path: str,
):
    """
    Augment the dataset by applying random group transformations on the
    weights and biases of the input INR. The transformations are applied to

    Parameters
    ----------
    device : torch.device
        The device (CPU or GPU) on which the dataset will be processed.
    Returns
    Batch
        The augmented dataset as a Batch object, containing the transformed
        weights and biases of the model.

    """

    torch.manual_seed(0)
    print("Buenos días")
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ensure deterministic behavior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load arbitrary INR

    sd = torch.load(inr_path, map_location=device)

    original_inr = INR()
    original_inr.load_state_dict(sd)

    """
    data = {
        "seq.0.weight": (32, 2),
        "seq.0.bias": (32),
        "seq.1.weight": (32, 32),
        "seq.1.bias": (32),
        "seq.2.weight": (1, 32),
        "seq.2.bias": (1),
    """
    NUMBER_OF_AUGMENTATIONS = 2
    L = 2  # Total number of hidden layers

    # weights = [(
    #   WEIGHTS LAYER 0 --> BATH_SIZE x 32 x 2
    # )
    weights, biases = [[], [], []], [[], [], []]

    # Perform augmentation
    for i in range(NUMBER_OF_AUGMENTATIONS):
        layer_to_flip = torch.randint(0, L, (1,)).item()  # pick a random layer
        sd = _transform_layer(
            sd=sd,
            layer_idx=layer_to_flip,
        )  # apply the transformation

        for j in [0, 1, 2]:
            weights[j].append(sd[f"seq.{j}.weight"].unsqueeze(0))
            biases[j].append(sd[f"seq.{j}.bias"].unsqueeze(0))

    for j in [0, 1, 2]:
        weights[j] = torch.cat(weights[j], dim=0)
        biases[j] = torch.cat(biases[j], dim=0)

    print(weights[0].shape)
    Batch(weights=weights, biases=biases, label=NUMBER_OF_AUGMENTATIONS * [2])
    batch = batch.to(device)


# Load the model
path = "/Users/administrador/Desktop/amsterdam/1.2/DL2/DL2/data/mnist-inrs/mnist_png_training_2_23089/checkpoints/model_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the augmented dataset
batch = get_augmented_dataset(device, path)

# Load model

# Pass batch through encoder -> Obtain latent representations

#
