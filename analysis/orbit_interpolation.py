from functools import partial
from typing import Callable
from src.scalegmn.models import ScaleGMN
from src.data.base_datasets import Batch
import torch
from torch.nn.functional import mse_loss
import argparse
import os
import torch_geometric

from src.utils.helpers import set_seed
from src.scalegmn.inr import INR
from analysis.utils import (
    collect_latents,
    load_orbit_dataset_and_model,
    create_tmp_torch_geometric_loader,
    remove_tmp_torch_geometric_loader,
    perturb_inr_all_batches,
    load_ground_truth_image,
)
from src.scalegmn.autoencoder import create_batch_wb
from src.phase_canonicalization.test_inr import test_inr

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


def INR_loss(
    ground_truth_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
) -> torch.Tensor:
    """MSE loss function for INR."""
    return mse_loss(
        reconstructed_image,
        ground_truth_image,
        reduction="none",
    ).mean(dim=list(range(1, reconstructed_image.dim())))


def test_inr_losses_batch(
    batch: Batch | torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    reconstructed: bool = False,
) -> float:
    """
    Compute the loss for a given dataset using the INR model.
    """
    
    if not reconstructed:
        weights_dev = [w.to(device) for w in batch.weights]
        biases_dev = [b.to(device) for b in batch.biases]
        imgs = test_inr(
            weights_dev,
            biases_dev,
            permuted_weights=True,
        )
    
    else:
        w_recon, b_recon = create_batch_wb(batch)
        w_recon = [w.to(device) for w in w_recon]
        b_recon = [b.to(device) for b in b_recon]
        imgs = test_inr(w_recon, b_recon)

    return loss_fn(imgs)


def compare_losses(
    net: ScaleGMN,
    loader: torch_geometric.data.DataLoader,
    device: torch.device,
):
    mnist_ground_truth_img = "data/mnist/train/2/23089.png"
    mnist_ground_truth_img = load_ground_truth_image(
        mnist_ground_truth_img,
        device=device,
    )

    for _, wb in loader:
        loss_vector = test_inr_losses_batch(
            batch=wb,
            loss_fn=partial(INR_loss, mnist_ground_truth_img.unsqueeze(0)),
            device=device,
            reconstructed=False,
        )
        print("Original INRs loss", loss_vector.mean().item())
        break

    for batch, wb in loader:
        batch = batch.to(device)
        loss_vector = test_inr_losses_batch(
            batch=net(batch),
            loss_fn=partial(INR_loss, mnist_ground_truth_img.unsqueeze(0)),
            device=device,
            reconstructed=True,
        )
        print("Reconstructed INRs loss", loss_vector.mean().item())
        break


def main():
    args = get_args()
    torch.set_float32_matmul_precision("high")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, loader = load_orbit_dataset_and_model(
        conf=args.conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    compare_losses(
        net=net,
        loader=loader,
        device=device,
    )

    perturbed_dataset: list[Batch]
    perturbed_dataset = perturb_inr_all_batches(
        dataset=loader.dataset,
        perturbation=1e-6,
    )

    if args.debug:
        perturbed_dataset = perturbed_dataset[:10]

    # Create torch gometric loader
    perturbed_loader = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset,
        tmp_dir="analysis/tmp_dir",
        conf=args.conf,
        device=device,
    )

    # Forward pass
    zs, _, _ = collect_latents(
        model=net,
        loader=loader,
        device=device,
    )

    zs_perturbed, _, _ = collect_latents(
        model=net,
        loader=perturbed_loader,
        device=device,
    )

    # Delete tmp dir
    remove_tmp_torch_geometric_loader(
        tmp_dir="analysis/tmp_dir",
    )


if __name__ == "__main__":
    main()