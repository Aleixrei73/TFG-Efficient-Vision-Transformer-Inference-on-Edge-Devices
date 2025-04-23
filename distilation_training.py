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
import timm
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
torch.cuda.manual_seed(0)

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
teacher = utils.load_default_model()
pretrained_vit_transforms = pretrained_vit_weights.transforms()

loss_fn = nn.CrossEntropyLoss()

for parameter in teacher.parameters():
    parameter.requires_grad = False
    
student = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=False)
# We need to change the head to match the number of classes to classify
student.head = nn.Linear(in_features=192, out_features=10)

for parameter in student.parameters():
    parameter.requires_grad = True
    
train_dl, test_dl, val_dl, class_names = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=64)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005/64)
writer = SummaryWriter()

trainer.trainKD(teacher,student,train_dl,test_dl,optimizer,loss_fn,epochs=100, writer=writer, model_name="Distilated", device=device)