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

r = [25, 30]
p = [0.4998]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

for pruning in p:
    print(f'prunning = {pruning:.2f}')
    for merging in r:
        print(f'r = {merging}')

        vit_default = utils.load_default_model()
        loss_fn = nn.CrossEntropyLoss()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        pruned_model = utils.prune_vit(vit_default, pruning)
        tome.patch.swag(pruned_model)
        pruned_model.r = merging
        pruned_model.load_state_dict(torch.load(f"model/Combination/Pruning{pruning:.1f}/ViT-Combine-Pruning{pruning:.2f}-Merging{merging}-Best.pht", weights_only=True))

        for parameter in pruned_model.parameters():
            parameter.requires_grad = False
            
        _, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=1)
        loss_fn = nn.CrossEntropyLoss()
            
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        metrics = trainer.getMetrics(pruned_model, val_dl, device, num_times=2000, save_plots=True, model_title=f'ViT-Combine-Pruning{pruning:.1f}-Merging{merging}')

        _, _, val_dl, _ = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=129)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        val_acc = trainer.evaluate(pruned_model, val_dl, loss_fn, device)

        with open(f'summary/ViT-Combine-Pruning{pruning:.1f}-Merging{merging}/metrics.txt', "a") as f:
            f.write(f'\nAccuracy: {(val_acc*100):.2f}%')