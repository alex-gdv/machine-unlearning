from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch
import os

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression
from util import output_statistics


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--checkpoint_epoch", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

torch.no_grad()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_dataset = UTKFaceRegression("data/test.json")
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

checkpoint = torch.load(f"./checkpoints/{args.experiment}/epoch_{args.checkpoint_epoch}.pt")
model = ResNet50Regression()
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

criterion = torch.nn.MSELoss(reduction="sum")
epoch_metrics = {}
for batch, (inputs, labels) in enumerate(test_dataloader):
    batch_metrics = {}
    inputs = inputs.to(device)
    labels = labels.to(device).float()

    outputs = model(inputs)

    loss = criterion(outputs, labels)

    batch_metrics["loss"] = loss.item()
    for window in [1, 5, 10]:
        batch_metrics[f"within_{window}"] = (torch.abs(outputs - labels) < window).sum()

    epoch_metrics["loss"] = epoch_metrics.get("loss", 0.) + batch_metrics["loss"]
    for window in [1, 5, 10]:
        epoch_metrics[f"within_{window}"] = epoch_metrics.get(f"within_{window}", 0.) + batch_metrics[f"within_{window}"]

    if batch % 10 == 0:
        print(f"BATCH {batch}")
        output_statistics(metrics=batch_metrics, size=len(inputs))

print("TEST RESULTS:")
output_statistics(metrics=epoch_metrics, size=len(test_dataloader))