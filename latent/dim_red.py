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


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conf",
        type=str,
        # required=True,
        default="configs/mnist_rec/scalegmn_autoencoder_ablation.yml",
        help="YAML config used during training",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        # required=True,
        default="models/mnist_rec_scale/scalegmn_autoencoder_ablation/scalegmn_autoencoder_baseline_mnist_rec_ablation.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--outdir", type=str, default="latent/resources/manifold_ablation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class DimensionalityReducer:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def full_pipeline(self, model, loader, device, save_path=None):
        zs, ys, _ = self.collect_latents(model, loader, device)
        embeddings = self.fit_transform(zs)

        if save_path is not None:
            self.scatter(
                embeddings,
                ys,
                f"UMAP - {args.split}",
                os.path.join(save_path),
            )
            print("Saved 2-D scatter plots to", save_path)

    def fit_transform(self, z):
        Z = z.numpy()
        if self.method == "umap":
            reducer = umap.UMAP(**self.kwargs)
        elif self.method == "tsne":
            reducer = TSNE(**self.kwargs)
        else:
            raise ValueError(self.method)

        Y = reducer.fit_transform(Z)
        return Y

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

    def scatter(Y, labels, title, out_png):
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        # Add legend
        legend1 = plt.legend(
            *scatter.legend_elements(), title="Classes", loc="upper right"
        )
        plt.gca().add_artist(legend1)
        plt.savefig(out_png, dpi=300)
        plt.close()


# --------------------------------------------------
def main(conf):
    # Load config
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    decoder_hidden_dim_list = [
        conf["scalegmn_args"]["d_hid"] * elem
        for elem in conf["decoder_args"]["d_hidden"]
    ]
    conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list

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
    # net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # ------------------ Dimensionality reduction -----------
    umap_reducer = DimensionalityReducer(
        method=umap,
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )

    tsne_reducer = DimensionalityReducer(
        method=TSNE,
        n_components=2,
        perplexity=10,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    # Run UMAP
    umap_reducer.full_pipeline(
        model=net,
        loader=loader,
        device=device,
        save_path=os.path.join(args.outdir, "umap"),
    )

    # Run t-SNE
    tsne_reducer.full_pipeline(
        model=net,
        loader=loader,
        device=device,
        save_path=os.path.join(args.outdir, "tsne"),
    )


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # ---------------- Load conf & build dataset ----------
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    main(conf=conf)
