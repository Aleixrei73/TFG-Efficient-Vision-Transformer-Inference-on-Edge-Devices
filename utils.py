import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from torchinfo import summary
from functools import reduce
import copy
import torch.nn.utils.prune as prune

NUM_WORKERS = os.cpu_count()

class FlattenPermute(nn.Module):

    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x):
        x = self.flatten(x)
        return x.permute(0, 2, 1)

def get_n_encoders(model, n):
    
    layers = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if idx == 1:
            layers.append(module)
        elif idx == 3:
            layers.append(module)
            layers.append(FlattenPermute(start_dim=2, end_dim=3))
        elif name.count('.') == 2:
            num = int(name.split('_')[-1])
            if num < n:
                layers.append(module)
            else:
                return nn.Sequential(*layers)
    
    return nn.Sequential(*layers)
        

def get_module(model, name):
    names = name.split(sep='.')
    return reduce(getattr, names, model)

def create_loaders(path, transform, batch_size, num_workers=NUM_WORKERS, dtype=None):

    if dtype != None:
        transform = transforms.Compose([
            transform,
            transforms.ConvertImageDtype(dtype)
        ])

    dataset = torchvision.datasets.CIFAR10(Path(path), train=True, transform=transform, download=True)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    val_dataset = torchvision.datasets.CIFAR10(Path(path), train=False, transform=transform, download=False)
    class_names = dataset.classes
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )
    return train_dl, test_dl, val_dl, class_names

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def load_default_model():
    model_path = Path("model/ViT-Default-Best.pht")
    vit = torchvision.models.vit_b_16()
    vit.heads = nn.Linear(in_features=768, out_features=10)
    vit.load_state_dict(torch.load(model_path, weights_only=True))
    return vit

def sumarize(model):
    print(summary(model=model,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ))
    
    
def prune_vit_mlp(working_layers):
    
    pruned_remains = []
    
    for idx, line in enumerate(working_layers[0].weight):
        if (torch.any(line != 0)):
            pruned_remains.append(idx)

    pruned_weights = working_layers[0].weight[pruned_remains]
    pruned_bias = working_layers[0].bias[pruned_remains]

    working_layers[0] = nn.Linear(in_features=working_layers[0].weight.shape[1], out_features=len(pruned_remains))

    working_layers[0].weight = nn.Parameter(pruned_weights)
    working_layers[0].bias = nn.Parameter(pruned_bias)

    pruned_weights = working_layers[3].weight[:, pruned_remains]
    aux_bias = working_layers[3].bias

    working_layers[3] = nn.Linear(in_features=len(pruned_remains), out_features=pruned_weights.shape[0])
    working_layers[3].weight = nn.Parameter(pruned_weights)
    working_layers[3].bias = nn.Parameter(aux_bias)
    
    
def prune_vit(model,amount):
    
    model_pruned = copy.deepcopy(model)
    
    for name, module in model_pruned.named_modules():
        if isinstance(module, nn.Linear) and 'mlp' in name:
            prune.ln_structured(module, "weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
            
    mlp_blocks = []

    for name,module in model_pruned.named_modules():
        
        if 'mlp' in name and name[-1] == 'p':
            mlp_blocks.append(module)
            
    for working_layers in mlp_blocks:
        prune_vit_mlp(working_layers)
    
    for parameter in model_pruned.parameters():
        parameter.requires_grad = True
        
    return model_pruned