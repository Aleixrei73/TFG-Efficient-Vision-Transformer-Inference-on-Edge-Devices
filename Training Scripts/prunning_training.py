import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import utils
import trainer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

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

prunning_step = 0.1
current_capacity = 1

prunned_model = pretrained_vit

for i in tqdm(range(1, 9)):
    
    writer = SummaryWriter()
    
    objective_pruning = (1 - (prunning_step*i))/current_capacity
    current_capacity = 1 - (prunning_step*i)
    needed_pruning = 1 - objective_pruning
    
    prunned_model = utils.prune_vit(prunned_model, needed_pruning)
    
    optimizer = torch.optim.SGD(prunned_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000)
    
    target_dir_path = Path("model")
    target_dir_path.mkdir(parents=True,exist_ok=True)

            # Create model save path
    model_save_path = target_dir_path / f"ViT-Prunning{needed_pruning:.2f}.pht"

            # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=prunned_model.state_dict(),f=model_save_path)

    print(f"Beggining training of prunning {prunning_step*i} with current {current_capacity} and needed {needed_pruning}")

    pretrained_vit_results, max_perf = trainer.train(model=prunned_model, train_dataloader=train_dl,
                                        test_dataloader=test_dl, optimizer=optimizer, scheduler=scheduler,
                                        loss_fn=loss_fn, epochs=4, writer=writer, model_name=f"PrunningDef{prunning_step*i:.2f}" , device=device)
    print(f"Max performance: {max_perf}")
    writer.close()