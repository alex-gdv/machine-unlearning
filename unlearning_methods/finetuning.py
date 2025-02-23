from torch.utils.data import DataLoader
import argparse
import torch
import os

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression
from util import output_statistics


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--load_model", type=str, required=True, help="trained model to finetune on retain set")
parser.add_argument("--checkpoint_freq", type=int, default=5)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = UTKFaceRegression("data/retain.json")
val_dataset = UTKFaceRegression("data/val.json")
model = ResNet50Regression()
criterion = torch.nn.MSELoss(reduction="sum")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)        
model = model.to(device)

os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)
checkpoint = torch.load(
    args.load_model, 
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

for epoch in range(0, args.epochs+1):
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