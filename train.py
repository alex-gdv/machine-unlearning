from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch
import os

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=False)
parser.add_argument("--checkpoint_freq", type=int, default=2)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = UTKFaceRegression("data/train.json")
val_dataset = UTKFaceRegression("data/val.json")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)        

os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)
if args.checkpoint is None:
    model = ResNet50Regression()
    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 0
else:
    checkpoint = torch.load(args.checkpoint)

    model = ResNet50Regression().load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam().load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

criterion = torch.nn.MSELoss(reduction="sum")
model = model.to(device)

for epoch in range(start_epoch, args.epochs):
    for mode in ["train", "validation"]:
        if mode == "train":
            model.train()
            dataloader = train_dataloader
            size = len(train_dataset)
        elif mode == "validation":
            if epoch % args.val_freq == 0:
                model.eval()
                dataloader = val_dataloader
                size = len(val_dataset)
            else:
                continue

        settings = {"epoch": epoch, "mode": mode}
        metrics = {}
        for batch, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc=mode, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)            
            metrics["loss"] = metrics.get("loss", 0.) + loss.item()
            for window in [1, 5, 10]:
                metrics[f"within_{window}"] = metrics.get(f"within_{window}", 0.) + (torch.abs(outputs - labels) < window).sum()

            if mode == "train":
                loss.backward()
                optimizer.step()
        
        table = list(settings.items()) + [(k, v/size) for k, v in metrics.items()]
        print(tabulate(table))

        if epoch % args.checkpoint_freq == 0:
            if not os.path.isdir(""):
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            f"./checkpoints/{args.experiment}/epoch_{epoch}.pt")