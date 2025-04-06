import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import utils
import trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = vit_default = utils.load_default_model()
pretrained_vit_transforms = pretrained_vit_weights.transforms()

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = True

train_dl, test_dl, val_dl, class_names = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=32)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pretrained_vit.parameters(), lr=0.03, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000)

writer = SummaryWriter()

print("Beggining training")
pretrained_vit_results = trainer.train(model=pretrained_vit, train_dataloader=train_dl, 
                                       test_dataloader=test_dl, optimizer=optimizer, scheduler=scheduler,
                                       loss_fn=loss_fn, epochs=5, writer=writer, device=device)

writer.close()