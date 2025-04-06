"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from pathlib import Path
import utils
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          writer: SummaryWriter,
          model_name: str,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    print("Generating comparative performance")
    _, current_max = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
    print(f"First performance: {current_max}")

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Acc/Test", test_acc, epoch)
        
        if test_acc > current_max:
            current_max = test_acc
            target_dir_path = Path("model")
            target_dir_path.mkdir(parents=True,exist_ok=True)

            # Create model save path
            model_save_path = target_dir_path / f"ViT-{model_name}-Best.pht"

            # Save the model state_dict()
            print(f"[INFO] Saving model to: {model_save_path}")
            torch.save(obj=model.state_dict(),f=model_save_path)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    writer.flush()
    return results, current_max

def evaluate(model, val_dataloader, loss_fn, device, model_path=None):


    print("Beginning evaluation...\n")

    model.to(device)

    _,val_acc = test_step(model=model,
          dataloader=val_dataloader,
          loss_fn=loss_fn,
          device=device)

    print(f'Validation accuracy: {(val_acc*100):.2f}')
    
    return val_acc


def plot_with_indicators(data,title,save_plots,plot_title,ylabel=None):
    
    x = np.array([i for i in range(len(data))])
    
    plt.figure()
    plt.plot(data, color="blue",label="Values",zorder=0)
    average = np.repeat(np.mean(data), len(data))
    std = np.repeat(np.std(data), len(data))
    plt.plot(average, color="Red", label="Mean",zorder=10)
    
    plt.fill_between(x, average-std, average + std, color="green", alpha=0.6,label="Std Dev",zorder=100)
    
    plt.title(title)
    plt.xlabel("Iteration")
    if ylabel is None: plt.ylabel("Time (ms/batch)")
    else: plt.ylabel(ylabel)
    
    plt.legend()
    
    if save_plots:
        
        directory_path = Path(f"summary/{plot_title}/graphs")
        directory_path.mkdir(parents=True, exist_ok=True)
        final_title = plot_title+"-"+title.replace(' ','-')
        plt.savefig(f"summary/{plot_title}/graphs/{final_title}.png")
    
    plt.show()

def getMetrics(model, val_dl, device, num_times=100, save_plots=False, model_title=None):
    
    if save_plots and model_title is None:
        print("Must set model_title when saving plots")
        return -1
    
    model = model.to(device)
    times_total = np.array([0]*num_times,dtype=np.float64)
    times_memory = np.array([0]*num_times,dtype=np.float64)
    times_inference = np.array([0]*num_times,dtype=np.float64)
    peak_memory = np.array([0]*num_times, dtype=np.float64)
    
    
    model.eval()
    
    print("Doing warm-up runs...")
    
    with torch.inference_mode():
        # We add 10 runs as warm-up runs
        for i in range(10):
            
            batch = next(iter(val_dl))
            X, y = batch
            
            X = X.to(device)
            y = y.to(device)
            torch.cuda.synchronize()
            
            res = model(X)
            torch.cuda.synchronize()
    
    print("Ended warm-up, beginning true runs...")
    
    with torch.inference_mode():
        # We add 10 runs as warm-up runs
        for i in tqdm(range(num_times)):
            
            # Reset of memory stats to accurate results
            torch.cuda.reset_peak_memory_stats()
            batch = next(iter(val_dl))
            X, y = batch
            
            # Compute memory management time
            time_init = time.time()
            X = X.to(device)
            y = y.to(device)
            torch.cuda.synchronize()
            times_memory[i] = (time.time() - time_init)*1000
            
            # Compute inference time
            time_init = time.time()
            res = model(X)
            torch.cuda.synchronize()
            times_inference[i] = (time.time() - time_init)*1000 
            
            # Computing total time and memory peak
            times_total[i] = times_memory[i]+times_inference[i]
            peak_memory[i] = torch.cuda.max_memory_allocated()/(1024*1024)
            
    
    plot_with_indicators(times_total,"Total time",save_plots,model_title)
    plot_with_indicators(times_memory,"Memory management time",save_plots,model_title)
    plot_with_indicators(times_inference,"Inference time",save_plots,model_title)
    plot_with_indicators(peak_memory,"Memory usage",save_plots,model_title, ylabel="Memory usage (MB)")
    
    if save_plots:
        with open(f"summary/{model_title}/metrics.txt", "w+") as f:
            f.write(f'Mean total time over {num_times} executions: {np.mean(times_total)} +- {np.std(times_total)} ms/batch \n'
                    f'Mean memory time over {num_times} executions: {np.mean(times_memory)} +- {np.std(times_memory)} ms/batch \n'
                    f'Mean inference time over {num_times} executions: {np.mean(times_inference)} +- {np.std(times_inference)} ms/batch \n'
                    f'Mean memory over {num_times} executions: {np.mean(peak_memory)} +- {np.std(times_total)} MB')
    
    print(f'Mean total time over {num_times} executions: {np.mean(times_total)} ms/batch \n'
          f'Mean memory time over {num_times} executions: {np.mean(times_memory)} ms/batch \n'
          f'Mean inference time over {num_times} executions: {np.mean(times_inference)} ms/batch \n'
          f'Mean memory over {num_times} executions: {np.mean(peak_memory)} MB')
    
    return {"Total latency" : times_total, "Memory latency" : times_memory, "Inference Latency" : times_inference, "Memory" : peak_memory}