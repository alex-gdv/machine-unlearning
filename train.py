from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch

from model.dataset import UTKFace
from model.model import ResNet50


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--checkpoint_freq", type=int, default=5)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--encoding", default="regression", choices=["ordinal", "regression"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = UTKFace("data/train.json")
val_dataset = UTKFace("data/val.json")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)        

if args.checkpoint_path is None:
    model = ResNet50()
    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 0
# else:
#     checkpoint = torch.load(args.checkpoint_path)

#     model = ResNet50().load_state_dict(checkpoint["model_state_dict"])
#     optimizer = torch.optim.Adam().load_state_dict(checkpoint["optimizer_state_dict"])

#     start_epoch = checkpoint["epoch"] + 1

criterion = torch.nn.MSELoss()
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

        # if epoch % args.checkpoint_freq == 0:
        #     torch.save({
        #                 'epoch': epoch,
        #                 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict()}, 
        #                 f"./checkpoints/{model_name}")
