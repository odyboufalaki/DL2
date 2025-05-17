import json
import os
from src.scalegmn.models import ScaleGMN
import torch
import yaml
import torch_geometric

from tqdm import tqdm

from src.utils.helpers import overwrite_conf
from src.data import dataset
from src.scalegmn.autoencoder import get_autoencoder
from src.data.base_datasets import Batch
from src.scalegmn.inr import INR
from PIL import Image
import torchvision.transforms as transforms


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
) -> tuple[ScaleGMN, torch_geometric.loader.DataLoader]:
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
        split_set, batch_size=conf["batch_size"], shuffle=False,
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


def perturb_inr_all_batches(
    loader: torch_geometric.loader.DataLoader,
    perturbation: float,
) -> list[Batch]:
    """Perturb the INR parameters in all batches of the dataset.

    Args:
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        perturbation (float): The amount to perturb the parameters by.

    Returns:
        list[Batch]: A list of perturbed batches of INR parameters.
    """
    perturbed_dataset = []
    for _, wb in loader:
        # Perturb weights and biases
        batch_perturbed = perturb_inr_batch(
            wb, perturbation=perturbation,
        )
        perturbed_dataset.append(batch_perturbed)
    return perturbed_dataset


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
) -> None:
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


def instantiate_inr_all_batches(
    loader: torch_geometric.loader.DataLoader,
    device: torch.device,
) -> list[INR]:
    """
    Instantiate INRs from the dataset.

    Args:
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        device (torch.device): The device to use for computation.
    Returns:
        list[INR]: A list of instantiated INR objects.
    """
    dataset = []
    for _, wb in loader:
        dataset.extend(instantiate_inr_batch(
            batch=wb,
            device=device,
        ))
    return dataset


def interpolate_inrs_batch(
    inr_batch_1: Batch,
    inr_batch_2: Batch,
    alpha: float,
    interpolation_type: str = "linear",
) -> Batch:
    """Interpolate between two batches of INRs.

    Args:
        inr_batch_1 (Batch): The first batch of INR parameters.
        inr_batch_2 (Batch): The second batch of INR parameters.
        alpha (float): The interpolation factor (0 <= alpha <= 1).
        num_samples (int): The number of samples to generate (currently unused).
        type (str): The type of interpolation to use.

    Returns:
        list[INR]: A list of interpolated INR objects.
    """
    # Interpolate between the two batches
    if interpolation_type == "linear":
        interpolated_weights = [
            (1 - alpha) * w1 + alpha * w2
            for w1, w2 in zip(inr_batch_1.weights, inr_batch_2.weights)
        ]
        interpolated_biases = [
            (1 - alpha) * b1 + alpha * b2
            for b1, b2 in zip(inr_batch_1.biases, inr_batch_2.biases)
        ]
    else:
        raise ValueError(f"Interpolation type {interpolation_type} not supported.")

    return Batch(
        weights=interpolated_weights,
        biases=interpolated_biases,
        label=inr_batch_1.label,
    )


def interpolate_inrs_all_batches(
    inr_batch_1: Batch,
    inr_batch_2: Batch,
    num_samples: int,
    interpolation_type: str = "linear",
) -> list[Batch]:
    """Interpolate between two batches of INRs.

    Args:
        inr_batch_1 (Batch): The first batch of INR parameters.
        inr_batch_2 (Batch): The second batch of INR parameters.
        num_samples (int): The number of samples to generate.
        type (str): The type of interpolation to use.

    Returns:
        list[INR]: A list of interpolated INR objects.
    """
    # Interpolate between the two batches
    if type == "linear":
        alpha_values = torch.linspace(0, 1, num_samples)
    else:
        raise ValueError(f"Interpolation type {type} not supported.")

    interpolated_inrs = []
    for alpha in alpha_values:
        interpolated_inrs.append(
            interpolate_inrs_batch(inr_batch_1, inr_batch_2, alpha, interpolation_type)
        )

    return interpolated_inrs


def load_ground_truth_image(
    image_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Load the ground truth image from the specified path.

    Args:
        image_path (str): The path to the image file.
        device (torch.device): The device to load the image onto.

    Returns:
        torch.Tensor: The loaded image as a tensor.
    """
    # Open the image using PIL
    image = Image.open(image_path).convert("L")

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)

    return image_tensor
