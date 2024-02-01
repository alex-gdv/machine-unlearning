import torch
from torch.utils.data import DataLoader

from dataset import UTKFace
from model import ResNet50


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UTKFace(
        root_dir="./data/data_clean",
        device=device
    )

    train_size = int(len(dataset)*0.7)
    validation_size = int(len(dataset)*0.1)
    test_size = len(dataset) - train_size - validation_size

    # set seed for deterministic split
    torch.manual_seed(0)
    train_dataset, validation_dataset, _ = torch.utils.data.random_split(
        dataset,
        [train_size, validation_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    model = ResNet50(
        nr_classes=dataset.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    n_epochs = 100
    
    for epoch in range(n_epochs):
        for mode in ["train"]:#, "validation"]:
            if mode == "train":
                model.train()
                dataloader = train_dataloader
            elif mode == "validation":
                model.eval()
                dataloader = validation_dataloader

            print(f"MODE {mode}", flush=True)

            total_loss = 0.
            total_correct = 0.
            for batch, data in enumerate(dataloader):
                optimizer.zero_grad()

                inputs, labels = data
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                correct = (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

                total_loss += loss.item()
                total_correct += correct

                if mode == "train":
                    loss.backward()
                    optimizer.step()

                if batch % 1 == 0:
                    print(f"BATCH {batch} LOSS {loss.item()} ACCURACY: {correct/len(inputs)}")

            print(f"MODE {mode} EPOCH {epoch} AVG LOSS {total_loss/len(dataloader)} AVG ACCURACY: {total_correct/train_size}")
            print("="*100)
