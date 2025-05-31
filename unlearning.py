from torch.utils.data import DataLoader
import argparse
import torch
import os

from model.dataset import UTKFaceRegression
from model.model import ResNetRegression
from unlearning_methods.unlearning_error import UnlearningError

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--unlearning_type", nargs="?", choices=["finetuning", "gradient_ascent"])
parser.add_argument("--load_model", type=str, required=True, help="trained model to finetune on retain set")
parser.add_argument("--checkpoint_freq", type=int, default=5)
parser.add_argument("--val_freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--reset_layers", type=int, default=0, help="num layers to reset")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = UTKFaceRegression("data/retain.json")
val_dataset = UTKFaceRegression("data/val.json")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)   
     
os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)

model = ResNetRegression()
checkpoint = torch.load(
    args.load_model, 
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

exact_unlearn_model = ResNetRegression()
exact_unlearn_checkpoint = torch.load(
    "checkpoints/exact_unlearning.pt", 
    map_location=device
)
exact_unlearn_model.load_state_dict(exact_unlearn_checkpoint["model_state_dict"])
exact_unlearn_model.to(device)

if args.reset_layers > 0:
    model.reset_first_k_layers(k=args.reset_layers)

if args.unlearning_type == "finetuning":
    from unlearning_methods.finetuning import finetune
    
    finetune(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        device=device, 
        args=args,
        unlearning_error=UnlearningError(exact_unlearn_model=exact_unlearn_model, args=args)
    )

elif args.unlearning_type == "gradient_ascent":
    from unlearning_methods.gradient_ascent import gradient_ascent
    
    forget_dataset = UTKFaceRegression("data/forget.json")
    forget_dataloader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    
    gradient_ascent(
        model, 
        train_dataloader, 
        val_dataloader, 
        forget_dataloader, 
        device, 
        args,
        unlearning_error=UnlearningError(exact_unlearn_model=exact_unlearn_model, args=args)
    )