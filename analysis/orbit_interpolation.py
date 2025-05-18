from functools import partial
from typing import Callable

import yaml
from src.scalegmn.models import ScaleGMN
from src.data.base_datasets import Batch
import torch
from torch.nn.functional import mse_loss
import argparse
import os
import torch_geometric
import gc

from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.inr import INR
from analysis.utils import (
    collect_latents,
    instantiate_inr_all_batches,
    load_orbit_dataset_and_model,
    create_tmp_torch_geometric_loader,
    remove_tmp_torch_geometric_loader,
    perturb_inr_all_batches,
    load_ground_truth_image,
    interpolate_batch,
    plot_interpolation_curves,
)
from src.scalegmn.autoencoder import create_batch_wb
from src.phase_canonicalization.test_inr import test_inr
from tqdm import tqdm

NUM_INTERPOLATION_SAMPLES = 10
BATCH_SIZE = 35

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
        default="data/mnist-inrs-orbit",
    )
    p.add_argument(
        "--split_path",
        type=str,
        default="data/mnist-inrs-orbit/mnist_orbit_splits.json",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--outdir", type=str, default="latent/resources/orbit_analysis")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    return p.parse_args()


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
def main():
    args = get_args()
    torch.set_float32_matmul_precision("high")
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
    conf["batch_size"] = BATCH_SIZE

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        perturbation=5 * 1e-3,
    )

    perturbed_dataset_inrs: list[INR]
    perturbed_dataset_inrs = instantiate_inr_all_batches(
        all_batches=perturbed_dataset_batches,
        device=device,
    )
    perturbed_dataset_inrs = perturbed_dataset_inrs[1:] + [perturbed_dataset_inrs[0]]

    if args.debug:
        perturbed_dataset_inrs = perturbed_dataset_inrs[:10]

    # Create torch gometric loader
    loader_perturbed = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset_inrs,
        tmp_dir="analysis/tmp_dir",
        conf=conf,
        device=device,
    )

    # Load ground truth image
    mnist_ground_truth_img = "data/mnist/train/2/23089.png"  # TODO: generalize
    mnist_ground_truth_img = load_ground_truth_image(
        mnist_ground_truth_img,
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
        tmp_dir="analysis/tmp_dir",
    )

    # Plot interpolation curve
    plot_interpolation_curves(
        loss_matrices=[
            (loss_matrix_original, "Original"),
            (loss_matrix_reconstruction, "Reconstructed")
        ],
        save_path="analysis/resources/interpolation/interpolation.png",
    )



if __name__ == "__main__":
    main()