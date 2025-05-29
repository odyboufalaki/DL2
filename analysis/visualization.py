import json
import random
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
from src.utils.orbit_dataset import generate_orbit_dataset
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset
from analysis.utils.utils_sgmn import collect_latents as collect_latents

class DimensionalityReducer:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def full_pipeline(
        self,
        zs,
        ys,
        save_path=None,
    ):
        embeddings = self.fit_transform(zs)

        if save_path is not None:
            self.scatter(
                embeddings,
                ys,
                f"UMAP",
                os.path.join(save_path),
            )
            print("Saved 2-D scatter plots to", save_path)

    def fit_transform(self, z):
        """
        Applies dimensionality reduction to the input data using the specified method.

        Parameters:
            z (torch.Tensor): A PyTorch tensor containing the input data to be reduced. 
                              It will be converted to a NumPy array for processing.

        Returns:
            numpy.ndarray: A NumPy array containing the transformed data in the reduced dimensional space.

        Raises:
            ValueError: If the specified method is not supported.

        Notes:
            - Supported methods for dimensionality reduction are:
              - "umap": Uses the UMAP (Uniform Manifold Approximation and Projection) algorithm.
              - "tsne": Uses the t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm.
            - Additional parameters for the dimensionality reduction method can be passed 
              via the `self.kwargs` dictionary.
        """
        Z = z.numpy()
        if self.method == "umap":
            reducer = umap.UMAP(**self.kwargs)
        elif self.method == "tsne":
            reducer = TSNE(**self.kwargs)
        else:
            raise ValueError(f"Method {self.method} not implemented.")

        Y = reducer.fit_transform(Z)
        return Y

    def scatter(
        self, 
        Y,
        Y_orbit,
        labels, 
        title, 
        out_png,
    ):
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
        scatter_orbit = plt.scatter(Y_orbit[:, 0], Y_orbit[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8, marker='x')
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        # Add legend
        # Create a single legend combining both scatter plots
        handles = scatter.legend_elements()[0] + scatter_orbit.legend_elements()[0]
        labels = [f"Class {i}" for i in range(len(scatter.legend_elements()[0]))] + \
                ["Orbit for class ?" for _ in range(len(scatter_orbit.legend_elements()[0]))]
        plt.legend(handles, labels, title="Classes & Orbit", loc="upper right")
        plt.savefig(out_png, dpi=300)
        plt.close()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conf",
        type=str,
        default="configs/mnist_rec/scalegmn_autoencoder.yml",
        help="YAML config used during training",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--outdir", type=str, default="analysis/resources/visualization")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", type=bool, default=False)
    p.add_argument(
        "--tmp_dir", 
        type=str, 
        default="analysis/tmp_dir",
        help="Directory to store the orbit dataset temporarily"
    )
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # ---------------- Load conf & build dataset ----------
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

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
    if args.debug:
        loader = [loader[0]]
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()

    # ---------------- Orbit dataset ------------------------------ 
    # Sample random INRs from the test set
    splits = json.load(open(conf["data"]["split_path"]))
    inr_path = random.sample(splits["test"]["path"], 1)[0]

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
        
    generate_orbit_dataset(
        output_dir=args.tmp_dir + "/orbit",
        inr_path=inr_path,
        device=device,
        dataset_size=args.dataset_size,
        transform_type=args.orbit_transformation,
    )

    net, loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )


    # ---------------- Model ------------------------------
    net = get_autoencoder(conf, autoencoder_type="inr").to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # ------------------ Dimensionality reduction -----------
    umap_reducer = DimensionalityReducer(
        method="umap",
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
    )

    tsne_reducer = DimensionalityReducer(
        method="tsne",
        n_components=2,
        perplexity=10,
        init="pca",
        learning_rate="auto",
    )

    # Collect latents
    zs, ys, _ = collect_latents(
        model=net,
        loader=loader,
        device=device,
    )

    # Run UMAP
    umap_reducer.full_pipeline(
        model=net,
        zs=zs,
        ys=ys,
        device=device,
        save_path=os.path.join(args.outdir, "umap"),
    )

    # Run t-SNE
    tsne_reducer.full_pipeline(
        model=net,
        zs=zs,
        ys=ys,
        device=device,
        save_path=os.path.join(args.outdir, "tsne"),
    )

    delete_orbit_dataset(args.tmp_dir + "/orbit")


if __name__ == "__main__":
    main()
