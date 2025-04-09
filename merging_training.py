import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import utils
import trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import tome.patch
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_default = utils.load_default_model()
pretrained_vit_transforms = pretrained_vit_weights.transforms()
tome.patch.swag(vit_default)

for parameter in vit_default.parameters():
    parameter.requires_grad = True

train_dl, test_dl, val_dl, class_names = utils.create_loaders("data", transform=pretrained_vit_transforms, batch_size=32)

loss_fn = nn.CrossEntropyLoss()

merging_step = 5

for i in tqdm(range(1, 7)):
    
    writer = SummaryWriter()
    
    merging_model = copy.deepcopy(vit_default)
    
    merging_model.r = merging_step*i
    
    optimizer = torch.optim.SGD(merging_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000)
    
    target_dir_path = Path("model")
    target_dir_path.mkdir(parents=True,exist_ok=True)

            # Create model save path
    model_save_path = target_dir_path / f"ViT-Merging{merging_step*i}.pht"

            # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=merging_model.state_dict(),f=model_save_path)

    print(f"Beggining training of merging with r = {merging_step*i}")

    pretrained_vit_results, max_perf = trainer.train(model=merging_model, train_dataloader=train_dl,
                                        test_dataloader=test_dl, optimizer=optimizer, scheduler=scheduler,
                                        loss_fn=loss_fn, epochs=3, writer=writer, model_name=f"Merging{merging_step*i}" , device=device)
    print(f"Max performance: {max_perf}")
    writer.close()