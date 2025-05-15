import torch
import argparse
import os

from src.utils.helpers import set_seed
from src.scalegmn.inr import INR
from analysis.utils import (
    collect_latents,
    load_orbit_dataset_and_model,
    create_tmp_torch_geometric_loader,
    remove_tmp_torch_geometric_loader,
    perturb_inr_all_batches,
)

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

    perturbed_dataset = perturb_inr_all_batches(
        dataset=loader.dataset,
        perturbation=1e-6,
    )

    if args.debug:
        perturbed_dataset = perturbed_dataset[:10]

    # Create torch gometric loader
    new_loader = create_tmp_torch_geometric_loader(
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
        loader=new_loader,
        device=device,
    )

    # Delete tmp dir
    remove_tmp_torch_geometric_loader(
        tmp_dir="analysis/tmp_dir",
    )




if __name__ == "__main__":
    main()