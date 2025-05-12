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
    p.add_argument("--outdir", type=str, default="latent/resources/interpolation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def collect_latents(model, loader, device):
    """Collects the latent codes from the model encoder for all samples in the dataset.
    Args:
        model: The model to use for encoding.
        loader: The data loader for the dataset.
        device: The device to use for computation.
    Returns:
        zs: The latent codes (tensor of shape [N, latent_dim]).
        ys: The labels (tensor of shape [N]).
        wbs: The raw INR parameters (list of tensors).
    """
    zs, ys, wbs = [], [], []
    model.eval()
    for batch, wb in tqdm(loader, desc="Collecting latents"):
        batch = batch.to(device)
        z = model.encoder(batch)  # [B, latent_dim]
        zs.append(z.cpu())
        ys.append(batch.label.cpu())
        wbs.append(wb)  # raw INR params, useful for reconstructions
    return torch.cat(zs), torch.cat(ys), wbs


if __name__ == "__main__":
    args = get_args()

    # Initialize
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    # Load the orbit dataset
    # Artificially overwrite the dataset path
    conf["data"]["dataset_path"] = args.dataset_path
    conf["data"]["split_path"] = args.split_path
    split_set = dataset(
        conf["data"],
        split="test",
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    loader = torch_geometric.loader.DataLoader(
        split_set, batch_size=conf["batch_size"], shuffle=False
    )
    print("Loaded", len(split_set), "samples in the dataset.")

    # Load the model
    decoder_hidden_dim_list = [
        conf["scalegmn_args"]["d_hid"] * elem
        for elem in conf["decoder_args"]["d_hidden"]
    ]
    conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()
    
    net = get_autoencoder(conf, autoencoder_type="inr").to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()


    zs, _, _ = collect_latents(net, loader, device)
    # zs has dims [DATASET_SIZE, latent_dim]

    # Compute euclidean distance matrix between the latent vectors
    dist_matrix = torch.cdist(zs, zs, p=2)
    dist_matrix = dist_matrix.cpu().numpy()

    # Plot the distance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Euclidean Distance Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "euclidean_distance_matrix.png"), dpi=300)
    plt.close()
    print("Euclidean distance matrix saved to:", os.path.join(args.outdir, "euclidean_distance_matrix.png"))
