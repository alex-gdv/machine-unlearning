from torch.utils.data import DataLoader
import argparse
import torch
import os

from unlearning_methods.util import output_statistics


def gradient_ascent(model, train_dataloader, val_dataloader, forget_dataloader, device, args, unlearning_error=None):
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(0, args.epochs+1):
        modes = ["train", "unlearning", "validation"] if epoch % args.val_freq == 0 else ["train", "unlearning"]
        for mode in modes:
            if mode == "train":
                model.train()
                dataloader = train_dataloader
                size = len(train_dataloader.dataset)
            elif mode == "unlearning":
                model.train()
                dataloader = forget_dataloader
                size = len(forget_dataloader.dataset)
            elif mode == "validation":
                model.eval()
                dataloader = val_dataloader
                size = len(val_dataloader.dataset)

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
                elif mode == "unlearning":
                    loss *= -1 # ascend on the loss value
                    loss.backward()
                    optimizer.step()
                
                if unlearning_error is not None:
                    epoch_metrics["unlearning_error"] = epoch_metrics.get(unlearning_error, 0.0) + unlearning_error.calculate_error(
                        model=model, 
                        criterion=criterion, 
                        dataloader=dataloader, 
                        step=len(dataloader)*epoch+batch
                    )
                
                if batch % 50 == 0:
                    print(f"EPOCH {epoch} BATCH {batch}")
                    output_statistics(settings=settings, metrics=batch_metrics, size=len(inputs))
            
            print(f"EPOCH {epoch}")
            output_statistics(settings=settings, metrics=epoch_metrics, size=size, table_out=True)
            
            if epoch % args.checkpoint_freq == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, 
                    f"./checkpoints/{args.experiment}/epoch_{epoch}.pt"
                )