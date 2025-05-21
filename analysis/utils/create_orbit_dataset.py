import json
import torch

import argparse
import os
import matplotlib.pyplot as plt
import torch_geometric

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import yaml

Tensor = torch.Tensor
StateDict = Dict[str, Tensor]
Transform = Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]

from src.utils.helpers import overwrite_conf, set_seed
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset
from src.scalegmn.inr import INR
from src.data.base_datasets import Batch
from tqdm import tqdm
import shutil


# Variables
INR_LAYER_ARCHITECTURE = [2, 32, 32, 1]
INR_NUMBER_LAYERS = len(INR_LAYER_ARCHITECTURE) - 1


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inr_path",
        type=str,
        default="data/mnist-inrs/mnist_png_training_2_23089/checkpoints/model_final.pth",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/mnist-inrs-orbit",
    )
    p.add_argument(
        "--dataset_size",
        type=int,
        default=2**12,
        help="Number of augmented INRs to generate",
    )
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


# ------------------------------
# Group transformation functions
def _row_sign_flip(
    w: torch.Tensor, b: torch.Tensor, w_next: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly flips the sign of the rows of the weight matrix `w` and the
    corresponding biases in `b`. The next layer's weight matrix `w_next` is
    also adjusted accordingly.

    Parameters:
        w (Tensor): The weight matrix of the current layer.
        b (Tensor): The bias vector of the current layer.
        w_next (Tensor): The weight matrix of the next layer.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: The transformed weight matrix,
        bias vector, and next layer's weight matrix, along with the flip signs.
    """
    flips = torch.randint(0, 2, (w.size(0),), device=w.device, dtype=w.dtype) * 2 - 1
    w = torch.diag(flips) @ w
    b = flips * b
    w_next = w_next @ torch.diag(flips)
    return w, b, w_next, flips


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
    w, b, w_next, flips = transform(w, b, w_next)

    new_sd[w_key] = w
    new_sd[b_key] = b
    new_sd[wn_key] = w_next
    return new_sd, flips


def transform_inr(
    sd: StateDict,
    layers_to_flip: List[int],
    transform: Transform = _row_sign_flip,
    prefix: str = "seq.",
) -> Tuple[StateDict, List[torch.Tensor]]:
    """
    Apply `transform` to the specified layers inside a *Sequential*-style
    state-dict and return the mutated copy.

    Parameters
    ----------
    sd : StateDict
        The state-dict (is shallow-copied for safety).
    layers_to_flip : List[int]
        List of 0-based indices of the layers to be transformed.
    transform : Transform
        Function to apply to the weights and biases of the layers.
    prefix : str
        Key prefix used in the state-dict (default "seq.").

    Returns
    -------
    Tuple[StateDict, List[Tensor]]
        The transformed state-dict and a list of flip tensors for each layer.
    """
    flips = []
    for layer_idx in layers_to_flip:
        sd, flip = _transform_layer(sd, layer_idx, transform, prefix)
        flips.append(flip)
    return sd, flips


def test_orbit_dataset(dataset, device, tolerance=1e-5):
    """
    Evaluate the INRs in a 28x28 grid and assert that they are all within a
    specified distance tolerance.

    Parameters
    ----------
    dataset : List[Batch]
        The dataset containing the augmented INRs to evaluate.
    tolerance : float
        The maximum allowable distance between the outputs of the INRs.

    Raises
    ------
    AssertionError
        If any pair of INRs produces outputs that differ by more than the
        specified tolerance.
    """
    grid_size = 100  # Fine-grained grid
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device)

    # Precompute outputs for all INRs
    outputs = []
    for inr in dataset:
        outputs.append(inr(grid))
        # Subtract the outputs of each INR from every other INR

    # Compute statistics: mean and standard deviation
    all_outputs = torch.stack(outputs)
    std_output = torch.max(torch.std(all_outputs, dim=0))

    print(f"Sanity check for obit dataset. std = {std_output}")

    if torch.max(std_output) > tolerance:
        raise AssertionError(
            f"Outputs differ by more than {tolerance} at some points in the grid."
        )


def generate_orbit_dataset(
    output_dir: str,
    inr_path: str,
    device: torch.device,
    dataset_size: int = 2**12,
) -> None:
    """
    Compute the loss matrix for the given dataset using the INR model.

    Args:
        output_dir (str): Directory to save the augmented dataset.
        inr_path (str): Path to the original INR model.
        device (torch.device): Device to use for computation (CPU or GPU).
        dataset_size (int): Number of augmented INRs to generate.

    Returns:
        None
    """
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ensure deterministic behavior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load arbitrary INR

    original_sd = torch.load(inr_path, map_location=device)

    inr_label = inr_path.split("/")[-3].split("_")[-2]

    flip_history = []
    dataset = []
    for inr_id in tqdm(range(dataset_size), desc="Generating augmented dataset"):
        layers_to_flip = list(range(0, INR_NUMBER_LAYERS - 1))
        sd, flips = transform_inr(
            sd=original_sd,
            layers_to_flip=layers_to_flip,
        )  # apply the transformation
        iteration_flips = flips  # flips is a tuple of flip tensors, one for each layer
        # Do not repeat the same transformation
        # if any(
        #     torch.equal(history_layer_flip, iteration_layer_flip)
        #     for history_flips in flip_history
        #     for history_layer_flip, iteration_layer_flip in zip(history_flips, iteration_flips)
        # ):
        #     print(
        #         f"Skipping INR {inr_id} because it has the same transformation as a previous one."
        #     )
        #     continue

        flip_history.append(iteration_flips)

        inr = INR()
        inr = inr.to(device)
        inr.load_state_dict(sd)
        dataset.append(inr)

    # Test the augmented dataset
    # If it fails the test, it will raise an AssertionError and stop the execution
    test_orbit_dataset(dataset, device, tolerance=1e-5)

    splits_json = dict()
    splits_json["test"] = {"path": [], "label": dataset_size * [inr_label]}
    for inr_id, inr in enumerate(dataset):
        # Save the transformed INR to the output directory
        output_path_dir = os.path.join(
            output_dir,
            inr_path.split("/")[2] + "_augmented_" + str(inr_id),
            "checkpoints",
        )
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)

        output_path = os.path.join(
            output_path_dir,
            "model_final.pth",
        )
        # {"test": {"path": ["data/mnist-inrs/mnist
        splits_json["test"]["path"].append(output_path)
        torch.save(inr.state_dict(), output_path)

    # Save the splits JSON file
    with open(os.path.join(output_dir, "mnist_orbit_splits.json"), "w") as f:
        json.dump(splits_json, f, indent=4)

    print(f"Saved {dataset_size} orbit samples to {output_dir}.")


def delete_orbit_dataset(output_dir: str) -> None:
    """
    Delete the orbit dataset directory.

    Args:
        output_dir (str): Directory to delete.

    Returns:
        None
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted the directory: {output_dir}")
    else:
        print(f"The directory {output_dir} does not exist.")

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.inr_path:
        args.inr_path = "data/mnist-inrs/mnist_png_training_2_23089/checkpoints/model_final.pth"

    if not args.output_dir:
        args.output_dir = "data/mnist-inrs-orbit"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    batches = generate_orbit_dataset(
        output_dir=args.output_dir,
        inr_path=args.inr_path,
        device=device,
    )
