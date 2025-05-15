import json
import os
import torch
import yaml
import torch_geometric

from tqdm import tqdm

from src.utils.helpers import overwrite_conf
from src.data import dataset
from src.scalegmn.autoencoder import get_autoencoder
from src.data.base_datasets import Batch
from src.scalegmn.inr import INR


@torch.no_grad()
def collect_latents(
    model: torch.nn.Module,
    loader: torch_geometric.loader.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Collects the latent codes from the model encoder for all samples in the dataset.
    Args:
        model (torch.nn.Module): The model to use for encoding.
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        device (torch.device): The device to use for computation.
    Returns:
        tuple: A tuple containing:
            - zs (torch.Tensor): The latent codes (tensor of shape [N, latent_dim]).
            - ys (torch.Tensor): The labels (tensor of shape [N]).
            - wbs (list[torch.Tensor]): The raw INR parameters (list of tensors).
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


def load_orbit_dataset_and_model(
    conf: str,
    dataset_path: str,
    split_path: str,
    device: torch.device,
    ckpt_path: str = None,
    return_model: bool = True,
) -> tuple[torch.nn.Module, torch_geometric.loader.DataLoader]:
    """
    Loads the orbit dataset and the pre-trained model.

    Args:
        conf (str): Path to the configuration YAML file.
        dataset_path (str): Path to the dataset directory.
        split_path (str): Path to the dataset split file.
        ckpt_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto.
        debug (bool): If True, loads only a subset of the dataset for debugging.

    Returns:
        tuple: A tuple containing:
            - net (torch.nn.Module): The loaded model.
            - loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
    """
    conf = yaml.safe_load(open(conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    # Overwrite the dataset path
    conf["data"]["dataset_path"] = dataset_path

    # Load the orbit dataset
    conf["data"]["split_path"] = split_path
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
    
    if return_model:
        net = get_autoencoder(conf, autoencoder_type="inr").to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()

    return (net, loader) if return_model else loader

def perturb_inr_batch(wb: Batch, perturbation: float) -> Batch:
    """Perturb the INR parameters in the batch by a small amount.

    Args:
        wb (Batch): The batch of INR parameters.
        perturbation (float): The amount to perturb the parameters by.

    Returns:
        Batch: The perturbed batch of INR parameters.
    """
    perturbed_weights = [
        stacked_tensors + perturbation * torch.randn_like(stacked_tensors)
        for stacked_tensors in wb.weights
    ]
    perturbed_biases = [
        stacked_tensors + perturbation * torch.randn_like(stacked_tensors)
        for stacked_tensors in wb.biases
    ]

    return Batch(
        weights=perturbed_weights,
        biases=perturbed_biases,
        label=wb.label,
    )

def create_tmp_torch_geometric_loader(
    dataset: list[INR],
    tmp_dir: str,
    conf: dict,
    device: torch.device,
) -> torch_geometric.loader.DataLoader:
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    print(f"Creating temporary torch geometric loader from checkpoints in {tmp_dir}...")

    splits_json = dict()
    splits_json["test"] = {"path": [], "label": []}
    for inr_id, inr in enumerate(dataset):
        # Save the transformed INR to the output directory
        output_path_dir = os.path.join(
            tmp_dir,
            str(inr_id),
            "checkpoints",
        )
        os.makedirs(output_path_dir, exist_ok=True)

        output_path = os.path.join(
            output_path_dir,
            "model_final.pth",
        )
        # {"test": {"path": ["data/mnist-inrs/mnist
        splits_json["test"]["path"].append(output_path)
        splits_json["test"]["label"].append("2")
        torch.save(inr.state_dict(), output_path)

    # Save the splits JSON file
    with open(os.path.join(tmp_dir, "splits.json"), "w") as f:
        json.dump(splits_json, f, indent=4)

    # Load the dataset
    loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=tmp_dir,
        split_path=os.path.join(tmp_dir, "splits.json"),
        device=device,
        return_model=False,
    )

    return loader

    
def remove_tmp_torch_geometric_loader(
    tmp_dir: str
):
    print(f"Removing temporary directory {tmp_dir}...")
    # Remove the temporary directory
    for inr_id in range(len(os.listdir(tmp_dir))):
        parent_dir = os.path.join(tmp_dir, str(inr_id))
        output_path_dir = os.path.join(
            parent_dir,
            "checkpoints",
        )
        if os.path.exists(output_path_dir):
            for file in os.listdir(output_path_dir):
                os.remove(os.path.join(output_path_dir, file))
            os.rmdir(output_path_dir)        
            os.rmdir(parent_dir)
            
    if os.path.exists(os.path.join(tmp_dir, "splits.json")):
        os.remove(os.path.join(tmp_dir, "splits.json"))
    os.rmdir(tmp_dir)


def instantiate_inr_batch(
    batch: Batch,
    device: torch.device,
) -> list[INR]:
    """Instantiate an INR object from the batch of parameters.

    Args:
        wb (Batch): The batch of INR parameters.

    Returns:
        INR: The instantiated INR object.
    """
    ## INR State dict
    # seq.0.weight: torch.Size([32, 2])
    # seq.0.bias: torch.Size([32])
    # seq.1.weight: torch.Size([32, 32])
    # seq.1.bias: torch.Size([32])
    # seq.2.weight: torch.Size([1, 32])
    # seq.2.bias: torch.Size([1])

    dataset = []
    batch_size = len(batch.weights[0])
    for inr_id in range(batch_size):
        inr = INR()
        state_dict = inr.state_dict()
        for i, (weight, bias) in enumerate(zip(batch.weights, batch.biases)):
            state_dict[f"seq.{i}.weight"] = weight[inr_id].squeeze(-1).transpose(-1, 0)
            state_dict[f"seq.{i}.bias"] = bias[inr_id].squeeze(-1).transpose(-1, 0)
        inr.load_state_dict(state_dict)
        inr.eval()
        dataset.append(inr.to(device))
    
    return dataset

if __name__ == "__main__":
    instantiate_inr_batch(None, "cpu")