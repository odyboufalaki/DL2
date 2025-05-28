import argparse
import gc
import json
import os
import random
from functools import partial
from typing import Callable

import torch
from torch.nn.functional import mse_loss
import torch_geometric
from tqdm import tqdm
import yaml

from analysis.utils.create_orbit_dataset import (
    generate_orbit_dataset,
    delete_orbit_dataset,
)
from analysis.utils.utils import (
    create_tmp_torch_geometric_loader,
    instantiate_inr_all_batches,
    interpolate_batch,
    load_ground_truth_image,
    load_orbit_dataset_and_model,
    perturb_inr_all_batches,
    plot_interpolation_curves,
    remove_tmp_torch_geometric_loader,
)
from analysis.linear_assignment import match_weights_biases_batch
from src.data.base_datasets import Batch
from src.scalegmn.autoencoder import create_batch_wb
from src.scalegmn.inr import INR
from src.scalegmn.models import ScaleGMN
from src.phase_canonicalization.test_inr import test_inr
from src.utils.helpers import overwrite_conf, set_seed
import pathlib

NUM_INTERPOLATION_SAMPLES = 40
BATCH_SIZE = 64

def convert_and_prepare_weights(rebased_weights, rebased_biases, device=None):
    """
    Convert rebased weights and biases to tensors and prepare them for _wb_to_tuple.
    
    Args:
        rebased_weights: List of lists of numpy arrays (weights for each layer of each INR)
        rebased_biases: List of lists of numpy arrays (biases for each layer of each INR)
        device: Optional torch.device to move tensors to
        
    Returns:
        Tuple of (weights, biases) in the format expected by _wb_to_tuple
    """
    # Convert to tensors
    weights_tensors = [
        [torch.from_numpy(w).to(device).float() if device else torch.from_numpy(w).float() 
         for w in inr_weights]
        for inr_weights in rebased_weights
    ]
    
    biases_tensors = [
        [torch.from_numpy(b).to(device).float() if device else torch.from_numpy(b).float() 
         for b in inr_biases]
        for inr_biases in rebased_biases
    ]
    
    # Reshape for _wb_to_tuple
    # We need to stack the weights and biases for each layer across the batch
    weights = [
        torch.stack([w[i] for w in weights_tensors]).unsqueeze(-1).permute(0,2,1,3)  # Add channel dimension
        for i in range(len(weights_tensors[0]))  # For each layer
    ]
    
    biases = [
        torch.stack([b[i] for b in biases_tensors]).unsqueeze(-1)  # Add channel dimension
        for i in range(len(biases_tensors[0]))  # For each layer
    ]
    
    return weights, biases


# ------------------------------
# Experiment funtionality
def inr_loss(
    ground_truth_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
) -> torch.Tensor:
    """MSE loss function for INR."""
    return mse_loss(
        reconstructed_image,
        ground_truth_image,
        reduction="none",
    ).mean(dim=list(range(1, reconstructed_image.dim())))


def inr_loss_batches(
    batch: Batch | torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    reconstructed: bool = False,
) -> torch.Tensor:
    """
    Compute the loss for a given dataset using the INR model.
    """
    
    weights_dev = [w.to(device) for w in batch.weights]
    biases_dev = [b.to(device) for b in batch.biases]
    imgs = test_inr(
        weights_dev,
        biases_dev,
        permuted_weights=reconstructed,
    )

    return loss_fn(imgs)


def compute_loss_matrix(
    interpolated_batches: list[Batch],
    mnist_ground_truth_img: torch.Tensor,
    device: torch.device,
    reconstructed: bool = False,
) -> torch.Tensor:
    """
    Compute the loss matrix for the given dataset using the INR model.

    Args:
        interpolated_batches (list[Batch]): List of batches to compute the loss for.
        mnist_ground_truth_img (torch.Tensor): Ground truth image for the dataset.
        device (torch.device): Device to perform the computation on.
        reconstructed (bool): Whether the weights are reconstructed or not.

    Returns:
        torch.Tensor: Loss matrix of shape (BATCH_SIZE, NUM_INTERPOLATION_SAMPLES).
    """
    loss_matrix = []
    for interpolated_batch in interpolated_batches:
        loss = inr_loss_batches(
            batch=interpolated_batch,
            loss_fn=partial(inr_loss, mnist_ground_truth_img),
            device=device,
            reconstructed=reconstructed,
        )
        loss_matrix.append(loss)
    loss_matrix = torch.stack(loss_matrix).permute(1, 0)
    return loss_matrix


@torch.no_grad()
def interpolation_experiment(
    args: argparse.Namespace,
    conf: dict, 
    device: torch.device,
):
    net, loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    perturbed_dataset_batches: list[Batch]
    perturbed_dataset_batches = perturb_inr_all_batches(
        loader=loader,
        perturbation=args.perturbation,
    )

    perturbed_dataset_inrs: list[INR]
    perturbed_dataset_inrs = instantiate_inr_all_batches(
        all_batches=perturbed_dataset_batches,
        device=device,
    )
    perturbed_dataset_inrs = perturbed_dataset_inrs[1:] + [perturbed_dataset_inrs[0]]

    # Create torch gometric loader
    loader_perturbed = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset_inrs,
        tmp_dir=args.tmp_dir,
        conf=conf,
        device=device,
    )

    # Load ground truth image
    mnist_ground_truth_img = load_ground_truth_image(
        args.mnist_ground_truth_img,
        device=device,
    )

    loss_matrix_original, loss_matrix_reconstruction = [], []
    for (batch_original, wb_original), (batch_perturbed, wb_perturbed) in tqdm(
        zip(loader, loader_perturbed), desc="Processing batches", total=len(loader)
    ):
        batch_original = batch_original.to(device)
        batch_perturbed = batch_perturbed.to(device)
        wb_original = wb_original.to(device)
        wb_perturbed = wb_perturbed.to(device)
        
        # Interpolation in original weight space
        interpolated_batches: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
        interpolated_batches = interpolate_batch(
            wb_original, wb_perturbed, NUM_INTERPOLATION_SAMPLES, 
        )

        # loss_matrix [BATCH_SIZE, NUM_INTERPOLATION_SAMPLES]
        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
            reconstructed=True,
        )
        loss_matrix_original.append(loss_matrix)

        # Interpolation in reconstructed weight space
        w_reconstructed_original, b_reconstructed_original = create_batch_wb(
            net(batch_original),
        )
        
        w_reconstructed_perturbed, b_reconstructed_perturbed = create_batch_wb(
            net(batch_perturbed),
        )
        wb_reconstructed_original = Batch(
            weights=w_reconstructed_original,
            biases=b_reconstructed_original,
            label=wb_original.label,
        )
        wb_reconstructed_perturbed = Batch(
            weights=w_reconstructed_perturbed,
            biases=b_reconstructed_perturbed,
            label=wb_perturbed.label,
        )
        
        interpolated_batches_reconstruction: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
        interpolated_batches_reconstruction = interpolate_batch(
            wb_reconstructed_original,
            wb_reconstructed_perturbed,
            num_samples=NUM_INTERPOLATION_SAMPLES,
        )

        # loss_matrix_reconstruction [BATCH_SIZE, NUM_INTERPOLATION_SAMPLES]
        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches_reconstruction,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
        )
        loss_matrix_reconstruction.append(loss_matrix)

    loss_matrix_original = torch.cat(loss_matrix_original, dim=0)
    loss_matrix_reconstruction = torch.cat(loss_matrix_reconstruction, dim=0)

    # Delete tmp dir
    remove_tmp_torch_geometric_loader(
        tmp_dir=args.tmp_dir,
    )

    return loss_matrix_original, loss_matrix_reconstruction

    # # Plot interpolation curve
    # plot_interpolation_curves(
    #     loss_matrices=[
    #         (loss_matrix_original, "Original"),
    #         (loss_matrix_reconstruction, "Reconstructed")
    #     ],
    #     save_path=args.image_save_path,
    # )


def test_interpolation_experiment():
    """
    Run the interpolation experiment with the given arguments and configuration.
    """
    args = get_args()
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
    conf["batch_size"] = BATCH_SIZE

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    args.mnist_ground_truth_img = "data/mnist/train/2/23089.png"
    args.save_path="analysis/resources/interpolation/interpolation.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interpolation_experiment(
        args=args,
        conf=conf,
        device=device,
        experiment_name="interpolation_experiment",
    )


def linear_assignment_experiment(
    args: argparse.Namespace,
    conf: dict, 
    device: torch.device,
    matching_type: str,
):
    """
    Run the linear assignment experiment with the given arguments and configuration.
    """
    net, loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    perturbed_dataset_batches: list[Batch]
    perturbed_dataset_batches = perturb_inr_all_batches(
        loader=loader,
        perturbation=args.perturbation,
    )

    perturbed_dataset_inrs: list[INR]
    perturbed_dataset_inrs = instantiate_inr_all_batches(
        all_batches=perturbed_dataset_batches,
        device=device,
    )
    perturbed_dataset_inrs = perturbed_dataset_inrs[1:] + [perturbed_dataset_inrs[0]]

    # Create torch gometric loader
    loader_perturbed = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset_inrs,
        tmp_dir=args.tmp_dir,
        conf=conf,
        device=device,
    )

    # Load ground truth image
    mnist_ground_truth_img = load_ground_truth_image(
        args.mnist_ground_truth_img,
        device=device,
    )

    loss_matrix_original, loss_matrix_reconstruction = [], []
    for (batch_original, wb_original), (batch_perturbed, wb_perturbed) in tqdm(
        zip(loader, loader_perturbed), desc="Processing batches", total=len(loader)
    ):
        batch_original = batch_original.to(device)
        batch_perturbed = batch_perturbed.to(device)
        wb_original = wb_original.to(device)
        wb_perturbed = wb_perturbed.to(device)
    
        interpolated_batches: list[Batch] 
        interpolated_batches = interpolate_batch(
            wb_original,
            wb_perturbed,
            num_samples=NUM_INTERPOLATION_SAMPLES,
        )

        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
            reconstructed=True,
        )
        loss_matrix_original.append(loss_matrix)
        
     
        rebased_weights, rebased_biases = match_weights_biases_batch(
            weights_A_batch=wb_original.weights,
            weights_B_batch=wb_perturbed.weights,
            biases_A_batch=wb_original.biases,
            biases_B_batch=wb_perturbed.biases,
            matching_type=matching_type,
        )
    
        rebased_weights, rebased_biases = convert_and_prepare_weights(rebased_weights, rebased_biases, device=device)
        
        # rebased weights and biases
        wb_rebased = Batch(
            weights=rebased_weights,
            biases=rebased_biases,
            label=wb_perturbed.label,
        )
        
        interpolated_batches_transformation: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
        interpolated_batches_transformation = interpolate_batch(
            wb_original,
            wb_rebased,
            num_samples=NUM_INTERPOLATION_SAMPLES,
        )

        # loss_matrix_reconstruction [BATCH_SIZE, NUM_INTERPOLATION_SAMPLES]
        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches_transformation,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
            reconstructed=True,
        )
        loss_matrix_reconstruction.append(loss_matrix)

    loss_matrix_original = torch.cat(loss_matrix_original, dim=0)
    loss_matrix_reconstruction = torch.cat(loss_matrix_reconstruction, dim=0)

    # Delete tmp dir
    remove_tmp_torch_geometric_loader(
        tmp_dir=args.tmp_dir,
    )

    return loss_matrix_original, loss_matrix_reconstruction

# ------------------------------
# Main function
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conf",
        type=str,
        default="configs/mnist_rec/scalegmn_autoencoder.yml",
        help="YAML config used during training",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default="analysis/tmp_dir/orbit",
    )
    p.add_argument(
        "--split_path",
        type=str,
        default="analysis/tmp_dir/orbit/mnist_orbit_splits.json",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="analysis/tmp_dir",
    )
    p.add_argument(
        "--dataset_size",
        type=int,
        default=512,
        help="Number of augmented INRs to generate",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs to perform",
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0,
        help="Perturbation to apply to the INR weights",
    )
    p.add_argument(
        "--linear_assignment",
        type=str,
        default=None,
        choices=["PD", "DP", "P", "D", None],
        help="Type of linear assignment to use for matching weights and biases (PD, DP, P, D)",
    )
    p.add_argument(
        "--save_matrices",
        action="store_true",
        help="Save the loss matrices for later analysis",
    )
    p.add_argument(
        "--orbit_transformation",
        type=str,
        default="PD",
        choices=["PD", "P", "D"],
        help="Type of transformation to apply to create the orbit dataset",
    )
    return p.parse_args()


def main():
    """
    Run the main function with different orbits from different INRs.
    This function creates a new dataset with different orbits and runs the interpolation experiment.
    """
    args = get_args()
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
    conf["batch_size"] = BATCH_SIZE

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    num_runs = args.num_runs

    # Sample random INRs from the test set
    splits = json.load(open(conf["data"]["split_path"]))
    sampled_inr_paths = random.sample(splits["test"]["path"], num_runs)
   
    orbit_dataset_path = args.tmp_dir + "/orbit"
    args.tmp_dir = args.tmp_dir + "/perturbed_orbit"  # Passed as argument to interpolation_experiment
    
    loss_matrix_original_list = []
    loss_matrix_reconstruction_list = []

    for experiment_id, inr_path in enumerate(sampled_inr_paths):
        conf = yaml.safe_load(open(args.conf))
        conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
        conf["batch_size"] = BATCH_SIZE
        
        print("-" * 50)
        print(f"Running experiment {experiment_id + 1}/{num_runs}...")
        ## Create the orbit dataset of the INR
        generate_orbit_dataset(
            output_dir=orbit_dataset_path,
            inr_path=inr_path,
            device=device,
            dataset_size=args.dataset_size,
            transform_type=args.orbit_transformation,
        )

        ## Run orbit interpolation experiment
        inr_label = inr_path.split("/")[-3].split("_")[-2]
        inr_id = inr_path.split("/")[-3].split("_")[-1]
        possible_pahts = [f"data/mnist/test/{inr_label}/{inr_id}.png", f"data/mnist/train/{inr_label}/{inr_id}.png"]

        # The image is either in the train or test set
        for path in possible_pahts:
            if os.path.exists(path):
                args.mnist_ground_truth_img = path
            break
        else:
            raise FileNotFoundError(f"None of the paths exist: {possible_pahts}")

      
        if args.linear_assignment:
            loss_matrix_original, loss_matrix_reconstruction = linear_assignment_experiment(
                args=args,
                conf=conf, 
                device=device,
                matching_type=args.linear_assignment,
            )
        else:
            loss_matrix_original, loss_matrix_reconstruction = interpolation_experiment(
                args=args,
                conf=conf, 
                device=device,
            )

        # Clear unused variables and free memory
        delete_orbit_dataset(orbit_dataset_path)

        loss_matrix_original_list.append(loss_matrix_original)
        loss_matrix_reconstruction_list.append(loss_matrix_reconstruction)

        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate the matrices after the loop
    loss_matrix_original_list = torch.cat(loss_matrix_original_list, dim=0)
    loss_matrix_reconstruction_list = torch.cat(loss_matrix_reconstruction_list, dim=0)

    # Save loss matrices for later analysis
    if args.save_matrices:
        output_dir = pathlib.Path("analysis/resources/interpolation/matrices")
        output_dir.mkdir(parents=True, exist_ok=True)

        method = "linear_assignment" if args.linear_assignment else "scalegmn"
        filename_original = f"loss_matrix-naive-{method}-{args.orbit_transformation}-numruns={num_runs}-perturbation={args.perturbation}.pt"
        filename_reconstruction = f"loss_matrix-reconstruction-{method}-{args.orbit_transformation}-numruns={num_runs}-perturbation={args.perturbation}.pt"

        torch.save(loss_matrix_original_list, output_dir / filename_original)
        torch.save(loss_matrix_reconstruction_list, output_dir / filename_reconstruction)

    plot_interpolation_curves(
        loss_matrices=[
            (loss_matrix_original_list, "Naive"),
            (loss_matrix_reconstruction_list, "Linear Assignment")
        ],
        save_path=f"analysis/resources/interpolation/interpolation_numruns={num_runs}_perturbation={args.perturbation}.png",
    )


if __name__ == "__main__":
    main()