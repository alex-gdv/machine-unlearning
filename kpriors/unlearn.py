from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import argparse
import torch
import os
import copy

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression
from model.adamreg import AdamReg
from kpriors.selector import select_memory_points


KPRIORS_MEMORY_FRACTION = 0.05


def output_statistics(settings, metrics, size, table_out=False):
    if table_out:
        print(tabulate(list(settings.items()) + [(k, v/size) for k, v in metrics.items()]), flush=True)
    else:
        print(*[(k, v/size) for k, v in metrics.items()], flush=True)


parser = argparse.ArgumentParser()
# unlearning arguments
parser.add_argument("--base_experiment", type=str, required=True)
parser.add_argument("--base_checkpoint", type=str, required=True)
# training arguments
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint_epoch", type=str, required=False)
parser.add_argument("--checkpoint_freq", type=int, default=5)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load base data and model
base_train_dataset = UTKFaceRegression("data/train.json")
base_train_dataloader = DataLoader(base_train_dataset, batch_size=args.batch_size, shuffle=True)
base_model = ResNet50Regression() 
base_checkpoint = torch.load(
    f"./checkpoints/{args.base_experiment}/epoch_{args.base_checkpoint}.pt"
)
base_model.load_state_dict(base_checkpoint["model_state_dict"])

# select memory points
memory_size = int(len(base_train_dataloader) * args.batch_size * KPRIORS_MEMORY_FRACTION)
memory = select_memory_points(base_train_dataloader, base_model, memory_size, device)

model = copy.deepcopy(base_model)
model = model.to(device)

# load forget data
forget_dataset = UTKFaceRegression("data/forget.json")
forget_dataloader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)

# load validation data
val_dataset = UTKFaceRegression("data/val.json")
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

criterion = torch.nn.MSELoss(reduction="sum")

os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)

if args.checkpoint_epoch is None:
    optimizer = AdamReg(model)
    optimizer.previous_weights = base_model.parameters()
    start_epoch = 0
# need to fix checkpoints with custom optimizer
else:
    checkpoint = torch.load(
        f"./checkpoints/{args.experiment}/epoch_{args.checkpoint_epoch}.pt", 
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1


for epoch in range(start_epoch, args.epochs+1):
    modes = ["forget", "validation"] if epoch % args.val_freq == 0 else ["train"]
    for mode in modes:
        if mode == "forget":
            model.train()
            dataloader = forget_dataloader
            size = len(forget_dataset)
        elif mode == "validation":
            model.eval()
            dataloader = val_dataloader
            size = len(val_dataset)

        settings = {"epoch": epoch, "mode": mode}
        epoch_metrics = {}
        for batch, (inputs, labels) in enumerate(dataloader):
            batch_metrics = {}
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            def closure_forget():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = -loss
                return loss
            
            def closure_memory():
                memory_inputs = memory["inputs"]
                memory_inputs = memory_inputs.to(device)

                

            batch_metrics["loss"] = loss.item()
            for window in [1, 5, 10]:
                batch_metrics[f"within_{window}"] = (torch.abs(outputs - labels) < window).sum().item()

            epoch_metrics["loss"] = epoch_metrics.get("loss", 0.) + batch_metrics["loss"]
            for window in [1, 5, 10]:
                epoch_metrics[f"within_{window}"] = epoch_metrics.get(f"within_{window}", 0.) + batch_metrics[f"within_{window}"]

            if mode == "train":
                loss.backward()
                optimizer.step()
            
            if batch % 50 == 0:
                print(f"EPOCH {epoch} BATCH {batch}")
                output_statistics(settings=settings, metrics=batch_metrics, size=len(inputs))
        
        print(f"EPOCH {epoch}")
        output_statistics(settings=settings, metrics=epoch_metrics, size=size, table_out=True)
        
        if epoch % args.checkpoint_freq == 0:
            if not os.path.isdir(""):
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, 
                    f"./checkpoints/{args.experiment}/epoch_{epoch}.pt"
                )