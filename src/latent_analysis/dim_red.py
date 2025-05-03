import torch
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import argparse
import os
import yaml
import torch_geometric
import matplotlib.pyplot as plt

from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset


# --------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conf",
        type=str,
        # required=True,
        default="configs/mnist_cls/scalegmn_autoencoder.yml",
        help="YAML config used during training"
    )
    p.add_argument(
        "--ckpt",
        type=str,
        # required=True,
        default="models/mnist_cls/scalegmn_autoencoder/scalegmn_autoencoder_mnist_cls.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--outdir", type=str, default="resources/manifold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="If >0, apply PCA to this dimension before UMAP / t-SNE",
    )
    return p.parse_args()


# --------------------------------------------------
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
    for batch, wb in loader:
        batch = batch.to(device)
        z = model.encoder(batch)  # [B, latent_dim]
        zs.append(z.cpu())
        ys.append(batch.label.cpu())
        wbs.append(wb)  # raw INR params, useful for reconstructions
    return torch.cat(zs), torch.cat(ys), wbs


# --------------------------------------------------
def dimensionality_reduction(z, labels, method="umap", pca_dim=None, **kwargs):
    Z = z.numpy()
    if pca_dim is not None and pca_dim > 0 and Z.shape[1] > pca_dim:
        Z = PCA(n_components=pca_dim).fit_transform(Z)
    if method == "umap":
        reducer = umap.UMAP(**kwargs)
    elif method == "tsne":
        reducer = TSNE(**kwargs)
    else:
        raise ValueError(method)

    Y = reducer.fit_transform(Z)
    return Y


def scatter(Y, labels, title, out_png):
    plt.figure(figsize=(6, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# --------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # ---------------- Load conf & build dataset ----------
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_set = dataset(
        conf["data"],
        split=args.split,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    loader = torch_geometric.loader.DataLoader(
        split_set, batch_size=conf["batch_size"], shuffle=False
    )
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()

    # ---------------- Model ------------------------------
    net = get_autoencoder(conf, autoencoder_type="inr").to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # ---------------- Collect latents --------------------
    global zs  # used inside decode_grid
    zs, ys, wbs = collect_latents(net, loader, device)
    print(
        f"Collected {len(zs)} latent codes of dim {zs.shape[1]} from {args.split} split."
    )

    # ---------------- Dim-red ----------------------------

    emb_umap = dimensionality_reduction(
        zs,
        ys,
        method="umap",
        pca_dim=args.pca_dim,
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
    )
    emb_tsne = dimensionality_reduction(
        zs,
        ys,
        method="tsne",
        pca_dim=args.pca_dim,
        perplexity=10,
        init="pca",
        learning_rate="auto",
    )

    scatter(emb_umap, ys, f"UMAP - {args.split}", os.path.join(args.outdir, "umap.png"))
    scatter(emb_tsne, ys, f"t-SNE - {args.split}", os.path.join(args.outdir, "tsne.png"))
    print("Saved 2-D scatter plots to", args.outdir)


if __name__ == "__main__":
    main()
