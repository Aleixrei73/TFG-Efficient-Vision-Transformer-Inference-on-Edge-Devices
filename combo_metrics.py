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
import time
import matplotlib.pyplot as plt
import numpy as np
import simplify
import torch.nn.utils.prune as prune
import tome.patch
import sys

# total arguments
r = int(sys.argv[1])

print(r)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

vit_default = utils.load_default_model()
loss_fn = nn.CrossEntropyLoss()
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit_transforms = pretrained_vit_weights.transforms()

prunning = 0.5997
pruned_model = utils.prune_vit(vit_default, prunning)
tome.patch.swag(pruned_model)
pruned_model.r = r
pruned_model.load_state_dict(torch.load(f"model/Combination/Pruning0.6/ViT-Combine-Pruning-Merging{r}-Best.pht", weights_only=True))

for parameter in pruned_model.parameters():
    parameter.requires_grad = False
    
_, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=1)
loss_fn = nn.CrossEntropyLoss()
    
torch.manual_seed(0)
torch.cuda.manual_seed(0)
metrics = trainer.getMetrics(pruned_model, val_dl, device, num_times=2000, save_plots=True, model_title=f'ViT-Combine-Pruning-Merging{r}')