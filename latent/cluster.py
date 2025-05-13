import argparse
import os
import yaml
import numpy as np
import torch
import torch_geometric
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset


def get_device(preferred_device=None):
    """
    Determine the computation device, checking CUDA, MPS (Apple), then CPU.
    """
    if preferred_device:
        return preferred_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_latents(conf_path, ckpt, split, batch_size, seed=0, device=None):
    """
    Load model and dataset, compute latent codes and labels.
    Returns zs (N x D) numpy array and ys (N,) numpy array.
    """
    # Load config and set seed
    conf = yaml.safe_load(open(conf_path))
    conf = overwrite_conf(conf, {"debug": False})
    set_seed(seed)

    # Choose device
    device = get_device(device)

    # Build dataset and loader
    split_set = dataset(
        conf["data"],
        split=split,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    loader = torch_geometric.loader.DataLoader(
        split_set, batch_size=batch_size, shuffle=False
    )
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()

    # Load autoencoder
    net = get_autoencoder(conf, autoencoder_type="inr").to(device)

    # Load checkpoint and filter to encoder only (avoid decoder mismatch)
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    # Extract encoder weights
    # ensure only encoder weights are loader
    enc_state = {
        k.replace("encoder.", ""): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    net.encoder.load_state_dict(enc_state)
    net.eval()
    net.eval()

    # Collect latents
    zs_list, ys_list = [], []
    with torch.no_grad():
        for batch, wb in loader:
            batch = batch.to(device)
            z = net.encoder(batch)  # [B, latent_dim]
            zs_list.append(z.cpu().numpy())
            ys_list.append(batch.label.cpu().numpy())
    zs = np.concatenate(zs_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    return zs, ys


def kmeans_classify(
    zs: np.ndarray, ys: np.ndarray, n_clusters: int = 10, random_state: int = 0
):
    """
    Fit KMeans on latent codes zs and map clusters to true labels by majority vote.
    Returns accuracy, predictions, cluster IDs, and mapping dict.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = kmeans.fit_predict(zs)

    mapping = {}
    for c in range(n_clusters):  # produce mapping for each cluster
        mask = cluster_ids == c  # pick samples in cluster c
        if not mask.any():  # no samples in this cluster
            mapping[c] = -1
        else:
            mapping[c] = int(
                np.bincount(ys[mask]).argmax()
            )  # most common label in cluster c

    y_pred = np.array([mapping[c] for c in cluster_ids])  # map cluster IDs to labels
    acc = accuracy_score(ys, y_pred)  # compute accuracy
    return acc, y_pred, cluster_ids, mapping


def main():
    parser = argparse.ArgumentParser(
        description="Compute latents then run KMeans clustering."
    )
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to YAML config used in training"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Model checkpoint path (.pt)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--clusters", type=int, default=10, help="Number of KMeans clusters"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for dataloader"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Compute latents
    zs, ys = collect_latents(
        args.conf, args.ckpt, args.split, args.batch_size, seed=args.seed
    )

    # Save latents and labels
    # np.save("latents.npy", zs)
    # np.save("labels.npy", ys)
    # print(f"Saved latents.npy (shape {zs.shape}) and labels.npy (shape {ys.shape})")

    # Run clustering
    acc, y_pred, cluster_ids, mapping = kmeans_classify(
        zs, ys, n_clusters=args.clusters, random_state=args.seed
    )
    print(f"KMeans clustering accuracy: {acc * 100:.2f}%")
    print("Cluster to label mapping:")
    for cluster, label in mapping.items():
        print(f"  Cluster {cluster}: Label {label}")

    # Save clustering outputs
    # np.save("cluster_ids.npy", cluster_ids)
    # np.save("predicted_labels.npy", y_pred)
    # print("Saved cluster_ids.npy and predicted_labels.npy")


if __name__ == "__main__":
    main()
