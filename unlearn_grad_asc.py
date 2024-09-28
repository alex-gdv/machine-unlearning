from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import argparse
import copy
import random

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression


# todo: thin these down
std_reg = 1.0
parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')
parser.add_argument('--unlearn_batch', default=114, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning epochs')
parser.add_argument('--checkpoint_freq',default=5, type=int,help='frequency of checkpointing')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# gather training and val datasets
train_dataset = UTKFaceRegression("data/retain.json")
val_dataset = UTKFaceRegression("data/val.json")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

# gather forget dataset
forget_dataset = UTKFaceRegression("data/forget.json")
forget_dataloader = DataLoader(forget_dataset, batch_size=len(forget_dataset), shuffle=True)

# randomly select a subset of images to gather hessian data for unlearning loss
hessian_loader = random.sample(train_dataset, k=512)
hessian_loader = torch.utils.data.DataLoader(hessian_loader, batch_size=512, shuffle=False, num_workers=2)

# get model from best checkpoint
model_pretrained = ResNet50Regression()
model_pretrained = model_pretrained.to(device)
checkpoint = torch.load(
    f"./checkpoints/{args.experiment}/epoch_best.pt", 
    map_location=device
)
model_pretrained.load_state_dict(checkpoint["model_state_dict"])
criterion = torch.nn.MSELoss()

# get gradients of the pretrained model on forget set
for X, y_true in enumerate(forget_dataloader):
    y_true = y_true.to(device)
    X = X.to(device)
    
    y_pred = model_pretrained(X)
    forget_loss = criterion(y_pred, y_true)
    
    forget_loss.backward(retain_graph=True)
    forget_grads = torch.autograd.grad(forget_loss, [param for param in model_pretrained.parameters()], create_graph = True)

model = copy.deepcopy(model_pretrained)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(0, args.finetune_epochs):
    for batch, (X, y_true) in enumerate(train_dataloader):
        print(f"epoch {epoch} batch {batch}")
        y_true = y_true.to(device)
        X = X.to(device)
        model.train()
        
        y_pred = model(X)
        
        optimizer.zero_grad()
        loss = criterion(y_pred, y_true)

        loss.backward()
        optimizer.step()
        
        # apply gradient ascent for unlearning
        model_unlearned = copy.deepcopy(model)        
        old_params = {}
        for i, (name, params) in enumerate(model.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += (epoch+1) * args.lr * forget_grads[i]    
        for name, params in model_unlearned.named_parameters():
            params.data.copy_(old_params[name])