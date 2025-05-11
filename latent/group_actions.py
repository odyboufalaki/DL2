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


# --------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--outdir", type=str, default="latent/resources/manifold_ablation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------


# ---------------------------------------------------------------------
# 1.  A generic transformation ------------------------------------------------
def row_sign_flip(
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


# ---------------------------------------------------------------------
# 2.  “Transform any layer” utility -----------------------------------
def transform_layer(
    sd: StateDict,
    layer_idx: int,
    transform: Transform = row_sign_flip,
    prefix: str = "seq.",
) -> StateDict:
    """
    Apply `transform` to layer `layer_idx` (and its successor) inside a
    *Sequential*-style state-dict and return the mutated copy.

    Parameters
    ----------
    sd          : the state-dict (is shallow-copied for safety)
    layer_idx   : 0-based index of the layer whose (W, b) you want to modify
    transform   : fn( weight, bias, next_weight ) -> (W′, b′, next_W′)
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


def transform_module_layer(
    model: torch.nn.Sequential, layer_idx: int, transform: Transform = row_sign_flip
):
    if layer_idx + 1 >= len(model):
        raise ValueError("Cannot transform the final layer")
    with torch.no_grad():
        w, b = model[layer_idx].weight.data, model[layer_idx].bias.data
        w_next = model[layer_idx + 1].weight.data
        w, b, w_next = transform(w, b, w_next)
        model[layer_idx].weight.data.copy_(w)
        model[layer_idx].bias.data.copy_(b)
        model[layer_idx + 1].weight.data.copy_(w_next)
    return model


# --------------------------------------------------
def main():
    print("Hello")
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ensure deterministic behavior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load arbitrary INR

    path = "/Users/administrador/Desktop/amsterdam/1.2/DL2/DL2/data/mnist-inrs/mnist_png_training_2_23089/checkpoints/model_final.pth"
    data = torch.load(path, map_location="cpu")

    """
    data = {
        "seq.0.weight": (32, 2),
        "seq.0.bias": (32),
        "seq.1.weight": (32, 32),
        "seq.1.bias": (32),
        "seq.2.weight": (1, 32),
        "seq.2.bias": (1),
    """

    NUMBER_OF_AUGMENTATIONS = 1
    L = 2  # Total number of hidden layers

    print("Saving original INR")
    test_inr(
        [data[f"seq.{j}.weight"].unsqueeze(0) for j in range(3)],
        [data[f"seq.{j}.bias"].unsqueeze(0) for j in range(3)],
        permuted_weights=False,
        save=True,
        img_name="original",
    )

    # Perform augmentation
    for i in range(NUMBER_OF_AUGMENTATIONS):
        layer_to_flip = torch.randint(0, L, (1,)).item()  # pick a random layer
        data = transform_layer(data, layer_to_flip)  # apply the transformation

        weights = [data[f"seq.{j}.weight"].unsqueeze(0) for j in range(3)]
        biases = [data[f"seq.{j}.bias"].unsqueeze(0) for j in range(3)]

        test_inr(
            weights,
            biases,
            permuted_weights=False,
            save=True,
            img_name=f"augmented_{i}",
        )


if __name__ == "__main__":
    main()
