from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch
import os

from model.dataset import UTKFaceRegression, UTKFaceOrdinal
from model.model import ResNet50Regression


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint_epoch", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

torch.no_grad()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.encoding == "ordinal":
    train_dataset = UTKFaceOrdinal("data/test.json")
else:
    test_dataset = UTKFaceRegression("data/test.json")

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
size = len(test_dataloader)

checkpoint = torch.load(f"./checkpoints/{args.experiment}/epoch_{args.checkpoint_epoch}.pt")
model = ResNet50Regression()
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

criterion = torch.nn.MSELoss(reduction="sum")
metrics = {}
for batch, (inputs, labels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="test", leave=False):
    inputs = inputs.to(device)
    labels = labels.to(device).float()

    outputs = model(inputs)

    loss = criterion(outputs, labels)            
    metrics["loss"] = metrics.get("loss", 0.) + loss.item()
    for window in [1, 5, 10]:
        metrics[f"within_{window}"] = metrics.get(f"within_{window}", 0.) + (torch.abs(outputs - labels) < window).sum()

table =[(k, v/size) for k, v in metrics.items()]
print(tabulate(table))