import torch
from torch.utils.data import DataLoader
import sys
import datetime

from model.dataset import UTKFace
from model.model import ResNet50


def train():
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UTKFace(
        root_dir="./data/data_clean",
        device=device
    )

    

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)        

    model = ResNet50().to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()

    # load checkpoint
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        checkpoint = torch.load(f"./checkpoints/{model_name}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        s_epoch = checkpoint["epoch"] + 1
    else:
        s_epoch = 0
        model_name = "model_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".pt"

    # training loop
    n_epochs = 100

    for epoch in range(s_epoch, n_epochs):
        for mode in ["train", "validation"]:
            if mode == "train":
                model.train()
                dataloader = train_dataloader
                size = train_size
            elif mode == "validation":
                model.eval()
                dataloader = validation_dataloader
                size = validation_size

            print(f"MODE {mode}", flush=True)

            total_mse = 0.
            total_mae = 0.
            total_correct_10 = 0
            total_correct_5 = 0
            total_correct_1 = 0
            for batch, data in enumerate(dataloader):
                optimizer.zero_grad()

                inputs, labels = data
                outputs = model(inputs)

                loss_mse = criterion_mse(outputs, labels)
                loss_mae = criterion_mae(outputs, labels)
                correct_10 = torch.abs(outputs - labels < 10).sum()
                correct_5 = torch.abs(outputs - labels < 5).sum()
                correct_1 = torch.abs(outputs - labels < 1).sum()


                total_mse += loss_mse.item()
                total_mae += loss_mae.item()
                total_correct_10 += correct_10
                total_correct_5 += correct_5
                total_correct_1 += correct_1


                if mode == "train":
                    loss_mse.backward()
                    optimizer.step()

                if batch % 100 == 0:
                    print(f"BATCH {batch} MSE {loss_mse.item()} MAE {loss_mae.item()} ACCURACY {correct_5/len(inputs)}")

            print(f"MODE {mode} EPOCH {epoch} AVG MSE {total_mse/len(dataloader)} AVG MAE {total_mae/len(dataloader)} AVG ACCURACY {total_correct_10/size} {total_correct_5/size} {total_correct_1/size}")
            print("="*100)

            if epoch % 2 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                            f"./checkpoints/{model_name}")
