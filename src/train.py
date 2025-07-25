import os
from typing import Optional, Tuple, BinaryIO, IO
import torch
import numpy as np
import yaml
from src.modules import Transformer
from src.optimizer import CrossEntropyLoss, GradientClipping, AdamW, CosineScheduler
import wandb

torch.manual_seed(42)


def data_loading(
    x: np.array,
    batch_size: int,
    context_length: int,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(x).to(device=device)
    indices = torch.randint(0, len(x) - context_length , (batch_size,), device=device).unsqueeze(-1) # (B, 1)
    offsets = torch.arange(context_length + 1, device=device).unsqueeze(0) # (1, seq_len)
    indices = offsets + indices
    batch = x[indices] # (B, seq_len)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    states = {}
    states["model"] = model.state_dict()
    states["optimizer"] = optimizer.state_dict()
    states["iteration"] = iteration
    torch.save(states, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    states = torch.load(src)
    model.load_state_dict(states["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(states["optimizer"])
    return states["iteration"]




def train(
    train_dataset: str, # path to .npy file
    config_path: str,
    load_checkpoint: Optional[str | os.PathLike | BinaryIO | IO[bytes]] = None,
    checkpoint_folder: Optional[str] = None,
    checkpoint_iters: Optional[int] = None,
    use_wandb: bool = False
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # load hyperparams
    device = config["device"]
    lr_min, lr_max, T_w, T_c, training_iterations, M, eps = config["optimizer"]["lr_min"], config["optimizer"]["lr_max"], config["optimizer"]["T_w"], config["optimizer"]["T_c"], config["optimizer"]["training_iterations"], config["optimizer"]["M"], config["optimizer"]["eps"]
    batch_size, context_length = config["batch_size"], config["model"]["context_length"]

    # prepare data
    data = np.load(train_dataset)

    model = torch.compile(Transformer(device=device, **config["model"]))
    optimizer = AdamW(model.parameters())

    if load_checkpoint is not None:
        load_checkpoint(load_checkpoint, model, optimizer)
    if use_wandb:
        wandb.init(project="Training")

    # train
    for t in range(1, training_iterations+1):
        optimizer.zero_grad()

        lr = CosineScheduler(t, lr_max, lr_min, T_w, T_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = data_loading(data, batch_size, context_length, device)
        outputs = model(inputs)
        loss = CrossEntropyLoss(outputs, targets)
        loss.backward()
        GradientClipping(model.parameters(), M, eps)
        optimizer.step()

        # logging
        if checkpoint_iters is not None and checkpoint_folder is not None and t != 0 and t % checkpoint_iters == 0:
            save_checkpoint(model, optimizer, t, f"{checkpoint_folder}/checkpoint_{t}")
        if checkpoint_folder is not None and t == training_iterations:
            save_checkpoint(model, optimizer, t, f"{checkpoint_folder}/checkpoint_{t}")
        if use_wandb:
            wandb.log({"loss": loss, "iteration": t})

if __name__ == "__main__":
    train(
        train_dataset="data/tokenized_data.npy",
        config_path="config.yaml", 
        checkpoint_folder="checkpoints",
        use_wandb=True,
    )