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

p = [5, 10, 15, 20, 25, 30]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

for pruning in p:
    
    print(f'r = {pruning:.2f}')

    vit_default = utils.load_default_model()
    loss_fn = nn.CrossEntropyLoss()
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    tome.patch.swag(vit_default)
    vit_default.r = pruning
    vit_default.load_state_dict(torch.load(f"model/Merging/ViT-Merging{pruning}-Best.pht", weights_only=True))

    for parameter in vit_default.parameters():
        parameter.requires_grad = False
           
    _, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=1, dtype=torch.bfloat16)
    loss_fn = nn.CrossEntropyLoss()
            
    vit_default = vit_default.to(torch.bfloat16)
            
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    metrics = trainer.getMetrics(vit_default, val_dl, device, num_times=2000, save_plots=True, model_title=f'ViT-Combine-Merging{pruning}-Quantized')

    _, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=129, dtype=torch.bfloat16)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    val_acc = trainer.evaluate(vit_default, val_dl, loss_fn, device)

    with open(f'summary/ViT-Combine-Merging{pruning}-Quantized/metrics.txt', "a") as f:
        f.write(f'\nAccuracy: {(val_acc*100):.2f}%')