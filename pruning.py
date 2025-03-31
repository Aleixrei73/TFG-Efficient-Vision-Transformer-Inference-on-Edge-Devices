import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import (models, transforms)
import os
import utils
from pathlib import Path
import trainer
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

vit_default = utils.load_default_model()
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit_transforms = pretrained_vit_weights.transforms()
_, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=32)

for amount in np.linspace(0, 0.2, 10):
    pruned_model = utils.prune_vit(vit_default, amount)
    metrics = trainer.getMetrics(vit_default, val_dl, device, num_times=100, save_plots=True, model_title=f"ViT-Pruning-P{amount*100:.2f}")