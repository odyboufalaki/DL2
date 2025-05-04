import torch
import yaml
import torch.nn.functional as F
import os
import random
from src.data import dataset
from tqdm import tqdm, trange
from torch.utils import data
import torch_geometric
import torch.distributed as dist
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.inr import INR, reconstruct_inr, make_functional
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import (
    overwrite_conf,
    count_parameters,
    set_seed,
    mask_input,
    mask_hidden,
    count_named_parameters,
)
from src.scalegmn.autoencoder import get_autoencoder
from src.data.base_datasets import Batch
from torchvision.utils import save_image
from src.scalegmn.autoencoder import create_batch_wb
import wandb

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def main(args=None):

    # Read base config file
    conf = yaml.safe_load(open(args.conf))
    # Overwrite base config with command-line arguments (if any)
    conf = overwrite_conf(conf, vars(args))

    # Initialize W&B (if enabled)
    #    - W&B automatically merges sweep parameters with the provided 'config'.
    #    - 'wandb.config' will hold the final, merged configuration.
    if conf.get("wandb", False):
        run = wandb.init(config=conf, **conf.get("wandb_args", {}))
        # Use wandb.config as the effective configuration
        effective_conf = wandb.config
    else:
        # If not using wandb, the original conf is the effective config
        effective_conf = conf
        run = None  # No wandb run object

    decoder_hidden_dim_list = [effective_conf["scalegmn_args"]["d_hid"]*elem for elem in effective_conf["decoder_args"]["d_hidden"]] 
    effective_conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    
    # Use 'effective_conf' consistently from here onwards
    torch.set_float32_matmul_precision("high")

    # print(yaml.dump(effective_conf, default_flow_style=False)) # Use effective_conf
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif (
        torch.cuda.is_available()
    ):  # Keep cuda check for cross-compatibility if needed elsewhere
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    # Wandb logging setup (already uses wandb.init implicitly if run exists)

    set_seed(effective_conf["train_args"]["seed"])  # Use effective_conf

    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    extra_aug = (
        effective_conf["data"].pop("extra_aug")
        if "extra_aug" in effective_conf["data"]
        else 0
    )  # Use effective_conf
    equiv_on_hidden = mask_hidden(effective_conf)  # Use effective_conf
    get_first_layer_mask = mask_input(effective_conf)  # Use effective_conf

    train_set = dataset(
        effective_conf["data"],  # Use effective_conf
        split="train",
        debug=effective_conf["debug"],  # Use effective_conf
        direction=effective_conf["scalegmn_args"]["direction"],  # Use effective_conf
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        return_wb=True,
    )
    effective_conf["scalegmn_args"][
        "layer_layout"
    ] = train_set.get_layer_layout()  # Use effective_conf

    # augment train set for the Augmented CIFAR-10 experiment
    if (
        extra_aug > 0 and effective_conf["data"]["dataset"] == "cifar_inr"
    ):  # Use effective_conf
        aug_dsets = []
        for i in range(extra_aug):
            aug_dsets.append(
                dataset(
                    effective_conf["data"],  # Use effective_conf
                    split="train",
                    debug=effective_conf["debug"],  # Use effective_conf
                    prefix=f"randinit_smaller_aug{i}",
                    direction=effective_conf["scalegmn_args"][
                        "direction"
                    ],  # Use effective_conf
                    equiv_on_hidden=equiv_on_hidden,
                    get_first_layer_mask=get_first_layer_mask,
                )
            )
        train_set = data.ConcatDataset([train_set] + aug_dsets)
        print(f"Augmented training set with {len(train_set)} examples.")

    val_set = dataset(
        effective_conf["data"],  # Use effective_conf
        split="val",
        debug=effective_conf["debug"],  # Use effective_conf
        # node_pos_embed=effective_conf['scalegmn_args']['graph_constructor']['node_pos_embed'], # Use effective_conf
        # edge_pos_embed=effective_conf['scalegmn_args']['graph_constructor']['edge_pos_embed'], # Use effective_conf
        direction=effective_conf["scalegmn_args"]["direction"],  # Use effective_conf
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        return_wb=True,
    )

    test_set = dataset(
        effective_conf["data"],  # Use effective_conf
        split="test",
        debug=effective_conf["debug"],  # Use effective_conf
        # node_pos_embed=effective_conf['scalegmn_args']['graph_constructor']['node_pos_embed'], # Use effective_conf
        # edge_pos_embed=effective_conf['scalegmn_args']['graph_constructor']['edge_pos_embed'], # Use effective_conf
        direction=effective_conf["scalegmn_args"]["direction"],  # Use effective_conf
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        return_wb=True,
    )

    print(f"Len train set: {len(train_set)}")
    print(f"Len val set: {len(val_set)}")
    print(f"Len test set: {len(test_set)}")

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=effective_conf["batch_size"],  # Use effective_conf
        shuffle=True,
        num_workers=effective_conf["num_workers"],  # Use effective_conf
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=effective_conf["batch_size"],  # Use effective_conf
        shuffle=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=effective_conf["batch_size"],  # Use effective_conf
        shuffle=True,
        num_workers=effective_conf["num_workers"],  # Use effective_conf
        pin_memory=True,
    )

    # =============================================================================================
    #   DEFINE MODEL
    # =============================================================================================
    # Get an instance of the autoencoder model
    net = get_autoencoder(
        model_args=effective_conf, autoencoder_type=effective_conf["train_args"]["reconstruction_type"]
    )  # Use effective_conf

    # cnt_p = count_parameters(net=net)
    # if effective_conf["wandb"]: # Use effective_conf
    #     wandb.log({'number of parameters': cnt_p}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    net = net.to(device)
    # =============================================================================================
    #   DEFINE LOSS
    # =============================================================================================
    criterion = select_criterion(
        effective_conf["train_args"]["loss"], {}
    )  # Use effective_conf

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    conf_opt = effective_conf["optimization"]  # Use effective_conf
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(
        model_params,
        optimizer_name=conf_opt["optimizer_name"],
        optimizer_args=conf_opt["optimizer_args"],
        scheduler_args=conf_opt["scheduler_args"],
    )
    # =============================================================================================
    # TRAINING LOOP
    # =============================================================================================
    #best_val_acc = -1
    #best_train_acc = -1
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    (
        best_test_results,
        best_val_results,
        best_train_results,
        best_train_results_TRAIN,
    ) = (None, None, None, None)
    #last_val_accs = []
    last_val_losses = []
    patience = effective_conf["train_args"]["patience"]  # Use effective_conf
    if extra_aug:
        # run experiment like in NFN to have comparable results.
        train_on_steps(
            net,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            scheduler,
            criterion,
            effective_conf,
            device,
        )
    else:
        for epoch in range(
            effective_conf["train_args"]["num_epochs"]
        ):  # Use effective_conf
            net.train()
            curr_loss = 0
            len_dataloader = len(train_loader)
            for i, (batch, wb) in enumerate(tqdm(train_loader)):
                step = epoch * len_dataloader + i
                batch = batch.to(device)
                # Move weights and biases to the target device
                weights_dev = [w.to(device) for w in wb.weights]
                biases_dev = [b.to(device) for b in wb.biases]

                optimizer.zero_grad()
                out = net(batch)

                # Reconstruct original images using tensors on the correct device - DO NOT SAVE
                original_imgs = test_inr(
                    weights_dev, biases_dev, permuted_weights=True,
                    pixel_expansion=effective_conf['train_args']['pixel_expansion']
                )

                # Reconstruct autoencoder images - SAVE THIS ONE
                if effective_conf["train_args"]["reconstruction_type"] == "inr":
                    w_recon, b_recon = create_batch_wb(
                        out
                    )  # Use default out_features=1
                    reconstructed_imgs = test_inr(
                        w_recon, b_recon, 
                        pixel_expansion=effective_conf['train_args']['pixel_expansion']
                    )
                elif effective_conf["train_args"]["reconstruction_type"] == "pixels":
                    reconstructed_imgs = out.view(
                        len(batch), *(tuple(effective_conf["data"]["image_size"]))
                    )  # Use effective_conf
                else:
                    raise ValueError(f"Unknown autoencoder type: {effective_conf['train_args']['reconstruction_type']}")
                #print(
                #    f"Original image shape: {original_imgs.shape}, Reconstructed image shape: {reconstructed_imgs.shape}"
                #)
                loss = criterion(reconstructed_imgs, original_imgs, weight=original_imgs if effective_conf["train_args"]["weigthed_loss"] else None)
                print(f"loss: {loss.item()}")

                curr_loss += loss.item()
                loss.backward()
                log = {}
                if effective_conf["optimization"]["clip_grad"]:  # Use effective_conf
                    log["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        net.parameters(),
                        effective_conf["optimization"]["clip_grad_max_norm"],
                    )  # Use effective_conf

                optimizer.step()

                if run:  # Check if wandb run exists
                    log[f"train/{effective_conf['train_args']['loss']}"] = (
                        loss.item()
                    )  # Use effective_conf
                    log["epoch"] = epoch

                if scheduler[1] is not None and scheduler[1] != "ReduceLROnPlateau":
                    if run:
                        log["lr"] = scheduler[0].get_last_lr()[
                            0
                        ]  # Check if wandb run exists
                    scheduler[0].step()

                if run:  # Check if wandb run exists
                    wandb.log(log, step=step)

            #############################################
            # VALIDATION
            #############################################
            #effective_conf["validate"] = False
            if effective_conf["validate"]:  # Use effective_conf
                print(f"\nValidation after epoch {epoch}:")
                val_loss_dict = evaluate(
                    net,
                    val_loader,
                    effective_conf["data"]["image_size"],
                    criterion,
                    device=device,
                    pixel_expansion=effective_conf["train_args"]["pixel_expansion"],  
                    effective_conf=effective_conf,  
                )
                """
                test_loss_dict = evaluate(
                    net,
                    test_loader,
                    effective_conf["data"]["image_size"],
                    criterion,
                    device=device,
                    pixel_expansion=effective_conf["train_args"]["pixel_expansion"],    
                    effective_conf=effective_conf,  
                )
                """
                val_loss = val_loss_dict["avg_loss"]
                #val_acc = val_loss_dict["avg_acc"]
                #test_loss = test_loss_dict["avg_loss"]
                #test_acc = test_loss_dict["avg_acc"]
                """
                train_loss_dict = evaluate(
                    net,
                    train_loader,
                    effective_conf["data"]["image_size"],
                    criterion,
                    train_set,
                    num_samples=len(val_set),
                    batch_size=effective_conf["batch_size"],
                    num_workers=effective_conf["num_workers"],
                    device=device,
                    pixel_expansion=effective_conf["train_args"]["pixel_expansion"],
                    effective_conf=effective_conf,  
                )  # Use effective_conf
                """

                best_val_criteria = val_loss <= best_val_loss
                if best_val_criteria:
                    best_val_loss = val_loss
                    #best_test_results = test_loss_dict
                    best_val_results = val_loss_dict
                    #best_train_results = train_loss_dict

                # Save the model
                if effective_conf["save_model"]["save_model"]:  # Use effective_conf
                    if (
                        best_val_criteria
                        and effective_conf["save_model"]["save_best_only"]
                    ):  # Use effective_conf
                        save_path = (
                            effective_conf["save_model"]["save_dir"]
                            + "/"
                            + effective_conf["save_model"]["save_name"]
                        )  # Use effective_conf
                        if not os.path.exists(
                            effective_conf["save_model"]["save_dir"]
                        ):  # Use effective_conf
                            os.makedirs(
                                effective_conf["save_model"]["save_dir"], exist_ok=True
                            )  # Use effective_conf
                        torch.save(net.state_dict(), save_path)

               # best_train_criteria = train_loss_dict["avg_loss"] <= best_train_loss
               # if best_train_criteria:
               #     best_train_loss= train_loss_dict["avg_loss"]
               #     best_train_results_TRAIN = train_loss_dict

                if run:  # Check if wandb run exists
                    log = {
                        #"train/avg_loss": train_loss_dict["avg_loss"],
                        #"train/acc": train_loss_dict["avg_acc"],
                        # "train/conf_mat": wandb.plot.confusion_matrix(
                        #     probs=None,
                        #     y_true=train_loss_dict["gt"],
                        #     preds=train_loss_dict["predicted"],
                        #     class_names=range(10),
                        # ), # Commented out as confusion matrix depends on task type
                        #"train/best_loss": best_train_results["avg_loss"],
                        #"train/best_acc": best_train_results["avg_acc"],
                        #"train/best_loss_TRAIN_based": best_train_results_TRAIN[
                        #    "avg_loss"
                        #],
                        #"train/best_acc_TRAIN_based": best_train_results_TRAIN[
                        #    "avg_acc"
                        #],
                        "val/loss": val_loss,
                        #"val/acc": val_acc,  # This is the metric needed for the sweep
                        "val/best_loss": best_val_results["avg_loss"],
                        #"val/best_acc": best_val_results["avg_acc"],
                        # "val/conf_mat": wandb.plot.confusion_matrix(
                        #     probs=None,
                        #     y_true=val_loss_dict["gt"],
                        #     preds=val_loss_dict["predicted"],
                        #     class_names=range(10),
                        # ), # Commented out as confusion matrix depends on task type
                        #"test/loss": test_loss,
                        #"test/acc": test_acc,
                        #"test/best_loss": best_test_results["avg_loss"],
                        #"test/best_acc": best_test_results["avg_acc"],
                        # "test/conf_mat": wandb.plot.confusion_matrix(
                        #     probs=None,
                        #     y_true=test_loss_dict["gt"],
                        #     preds=test_loss_dict["predicted"],
                        #     class_names=range(10),
                        # ), # Commented out as confusion matrix depends on task type
                        "epoch": epoch,
                    }

                    wandb.log(log)

                net.train()

                last_val_losses.append(val_loss)
                # Keep only the last accuracies
                if len(last_val_losses) > patience:
                    last_val_losses.pop(0)
                # Check if the accuracies are decreasing
                if len(last_val_losses) == patience and all(
                    x < y for x, y in zip(last_val_losses, last_val_losses[1:])
                ):
                    print(
                        f"Validation loss has been increasing for {patience} consecutive epochs:\n{last_val_losses}\nExiting."
                    )
                    if run:
                        wandb.finish()  # Finish W&B run
                    return 1  # Indicate early stopping

    if run:
        wandb.finish()  # Finish W&B run at the end


@torch.no_grad()
def evaluate(
    model,
    loader, 
    image_size,
    criterion,
    eval_dataset=None,
    num_samples=0,
    batch_size=0,
    num_workers=8,
    device=None,
    pixel_expansion=1,
    effective_conf=None,
   
):
    if eval_dataset is not None:
        # only when also evaluating on train split. Since it is a lot bigger, we only evaluate on a smaller subset.
        indices = random.sample(range(len(eval_dataset)), num_samples)
        subset_dataset = torch.utils.data.Subset(eval_dataset, indices)
        loader = torch_geometric.loader.DataLoader(
            dataset=subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            # sampler=sampler
        )

    model.eval()
    total_loss = 0.0
    total = 0.0
    for i, (batch, wb) in enumerate(tqdm(loader)):
        batch = batch.to(device)
        out = model(batch)
        #step = epoch * len_dataloader + i
        # Move weights and biases to the target device
        weights_dev = [w.to(device) for w in wb.weights]
        biases_dev = [b.to(device) for b in wb.biases]

        # Reconstruct original images using tensors on the correct device - DO NOT SAVE
        original_imgs = test_inr(
            weights_dev, biases_dev, permuted_weights=True, save=True, img_name="original_",
            pixel_expansion=pixel_expansion
        )

        # Reconstruct autoencoder images - SAVE THIS ONE
        if effective_conf["train_args"]["reconstruction_type"] == "inr":
            w_recon, b_recon = create_batch_wb(
                out
            )  # Use default out_features=1
            reconstructed_imgs = test_inr(
                w_recon, b_recon, save=True, img_name="inr_",
                pixel_expansion=pixel_expansion
            )
        elif effective_conf["train_args"]["reconstruction_type"] == "pixels":
            reconstructed_imgs = out.view(
                len(batch), *(tuple(image_size))
            )  # Use effective_conf
            save_image(
                [reconstructed_imgs[0].squeeze(-1).detach().cpu()], "pixels_auto_reconstructed.png"
            )
        else:
            raise ValueError(f"Unknown autoencoder type: {effective_conf['train_args']['reconstruction_type']}")

        loss = criterion(reconstructed_imgs, original_imgs)
        total_loss += loss.item() * len(batch)  # Scale loss up to total sum
        total += len(batch)                     # Total number of samples

    model.train()
    avg_loss = total_loss / total
   

    return dict(avg_loss=avg_loss) 


def train_on_steps(
    net,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    run_config,
    device,
):  # Renamed conf to run_config
    """
    Follow the same training procedure as in 'Zhou, Allan, et al. "Permutation equivariant neural functionals." NIPS (2024).'
    """
    # import wandb # Already imported globally

    def cycle(loader):
        while True:
            for blah in loader:
                yield blah

    best_val_acc = -1
    best_test_results, best_val_results = None, None

    train_iter = cycle(train_loader)
    outer_pbar = trange(
        0, run_config["train_args"]["max_steps"], position=0
    )  # Use run_config

    for step in outer_pbar:

        if (
            step > 0
            and step % 3000 == 0
            or step == run_config["train_args"]["max_steps"] - 1
        ):  # Use run_config
            val_loss_dict = evaluate(net, val_loader, device=device)
            test_loss_dict = evaluate(net, test_loader, device=device)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict

            if (
                run_config.get("wandb", False) and wandb.run is not None
            ):  # Use run_config and check wandb.run
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/acc": val_acc,  # This is the metric needed for the sweep
                        "val/best_loss": best_val_results["avg_loss"],
                        "val/best_acc": best_val_results["avg_acc"],
                        "test/loss": test_loss,
                        "test/acc": test_acc,
                        "test/best_loss": best_test_results["avg_loss"],
                        "test/best_acc": best_test_results["avg_acc"],
                        "step": step,
                        "epoch": step // len(train_loader),
                    }
                )

        net.train()
        # The batch now contains (graph_batch, (weights, biases)) tuple
        batch, wb = next(train_iter)  # Unpack the tuple
        batch = batch.to(device)  # Move graph batch to device
        # Move weights and biases to the target device - feels weird idk
        weights_dev = [w.to(device) for w in wb.weights]
        biases_dev = [b.to(device) for b in wb.biases]

        optimizer.zero_grad()
        # inputs = (batch) # Model expects the graph batch
        out = net(batch)

        # Reconstruct original images using tensors on the correct device - DO NOT SAVE
        original_imgs = test_inr(
            weights_dev, biases_dev, permuted_weights=True, save=False
        )

        # Reconstruct autoencoder images - SAVE THIS ONE
        if run_config["train_args"]["reconstruction_type"] == "inr":
            w_recon, b_recon = create_batch_wb(out)  # Use default out_features=1
            reconstructed_imgs = test_inr(
                w_recon, b_recon, save=True, img_name="autoencoder_recon"
            )  # Save with specific name
        elif run_config["train_args"]["reconstruction_type"] == "pixels":
            reconstructed_imgs = out.view(
                len(batch), *(tuple(run_config["data"]["image_size"]))
            )  # Use run_config
        else:
            raise ValueError(f"Unknown autoencoder type: {run_config['train_args']['reconstruction_type']}")

        loss = criterion(reconstructed_imgs, original_imgs)  # Use original_imgs
        loss.backward()

        log = {}
        if run_config["optimization"]["clip_grad"]:  # Use run_config
            log["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                net.parameters(), run_config["optimization"]["clip_grad_max_norm"]
            )  # Use run_config

        optimizer.step()

        if (
            run_config.get("wandb", False) and wandb.run is not None
        ):  # Use run_config and check wandb.run
            log[f"train/{run_config['train_args']['loss']}"] = (
                loss.item()
            )  # Use run_config
            log["step"] = step

        if scheduler[1] is not None and scheduler[1] != "ReduceLROnPlateau":
            if run_config.get("wandb", False) and wandb.run is not None:
                log["lr"] = scheduler[0].get_last_lr()[
                    0
                ]  # Use run_config and check wandb.run
            scheduler[0].step()

        if (
            run_config.get("wandb", False) and wandb.run is not None
        ):  # Use run_config and check wandb.run
            wandb.log(log, step=step + 1)

    # It's generally better practice to let the main function handle finishing the run
    # if run_config.get("wandb", False) and wandb.run is not None:
    #     wandb.finish()


if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    if not args.conf:
        args.conf = "configs/mnist_rec/scalegmn_autoencoder_sweep.yml"

    # No need to load config here, main function handles it
    # conf = yaml.safe_load(open(args.conf))

    main(args=args)
