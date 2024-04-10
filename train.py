from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import argparse
import torch
import os

from model.dataset import UTKFaceRegression, UTKFaceOrdinal
from model.model import ResNet50Regression, ResNet50Ordinal


def output_statistics(settings, metrics, size, table_out=False):
    if table_out:
        print(tabulate(list(settings.items()) + [(k, v/size) for k, v in metrics.items()]), flush=True)
    else:
        print(*[(k, v/size) for k, v in metrics.items()], flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint_epoch", type=str, required=False)
parser.add_argument("--checkpoint_freq", type=int, default=5)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.encoding == "ordinal":
    num_classes = 100
    train_dataset = UTKFaceOrdinal("data/train.json", num_classes=num_classes)
    val_dataset = UTKFaceOrdinal("data/val.json", num_classes=num_classes)
    model = ResNet50Ordinal(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
else:
    train_dataset = UTKFaceRegression("data/train.json")
    val_dataset = UTKFaceRegression("data/val.json")
    model = ResNet50Regression()
    criterion = torch.nn.MSELoss(reduction="sum")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)        
model = model.to(device)

os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)
if args.checkpoint_epoch is None:
    optimizer = torch.optim.Adam(model.parameters())
    start_epoch = 0
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
    modes = ["train", "validation"] if epoch % args.val_freq == 0 else ["train"]
    for mode in modes:
        if mode == "train":
            model.train()
            dataloader = train_dataloader
            size = len(train_dataset)
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

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)            

            if args.encoding == "ordinal":
                outputs = outputs.round().argmin(dim=1)
                labels = labels.round().argmin(dim=1)

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