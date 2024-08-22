from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import copy
import numpy as np
from PyHessian.pyhessian import hessian
import os

from model.dataset import UTKFaceRegression
from model.model import ResNet50Regression


def weights_to_list_fast(weights):
  with torch.no_grad():
    weights_list = []
    for weight in weights:
      list_t = weight.view(-1).tolist()
      weights_list = weights_list + list_t

    return weights_list


def std_loss(x,y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1))/(len(x.view(-1)))
    loss = loss + std_reg*avg_std
    return loss

std_reg =1.0
parser = argparse.ArgumentParser(description='Finetuning for verification error')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--model', default='resnet', type=str, help='resnet or vgg')
parser.add_argument('--loss_func', default='regular', type=str, help='loss function: regular,hessian, hessianv2, std_loss')
parser.add_argument('--dataset', default = 'cifar10', type=str, help ='cifar10, cifar100')
parser.add_argument('--pretrain_epochs', default=10, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='pretraining batch size')
parser.add_argument('--finetune_batch_size', default=32, type=int, help='finetuning batch size')
parser.add_argument('--unlearn_batch', default=114, type=int, help='what batch of data should be unlearned')
parser.add_argument('--finetune_epochs', default=1, type=int, help='number of finetuning epochs')
parser.add_argument('--checkpoint_freq',default=5, type=int,help='frequency of checkpointing')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = UTKFaceRegression("data/retain.json")
val_dataset = UTKFaceRegression("data/val.json")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)   

print('==> Preparing Hessian data..')
# ! UNDERSTAND WHAT THIS IS BEFORE CONTINUING WITH IT
# trainset_list = list(trainset)
# hessian_loader = trainset_list[:512]
# hessian_loader = torch.utils.data.DataLoader(hessian_loader, batch_size=512, shuffle=False, num_workers=2)

# print('==> Preparing Finetuning data...')
# batch_star = trainset_list[args.finetune_batch_size * args.unlearn_batch: args.finetune_batch_size * (args.unlearn_batch+1)]
# data_no_unlearned = trainset_list[:args.finetune_batch_size * args.unlearn_batch] + trainset_list[args.finetune_batch_size * (args.unlearn_batch+1):]
# unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size=args.finetune_batch_size, shuffle=False, num_workers=2)


#Getting model
model = ResNet50Regression()
model = model.to(device)

os.makedirs(f"./checkpoints/{args.experiment}", exist_ok=True)
if args.checkpoint_epoch is None:
    optimizer = torch.optim.Adam(model.parameters())
    start_epoch = 0
else:
    checkpoint = torch.load(
        f"./checkpoints/{args.experiment}/epoch_{args.checkpoint_epoch}.pt", 
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

# ! WHAT'S THE POINT OF THIS?
#saving the weights of the pretrained model
M_pretrain = copy.deepcopy(model)
w_pretrain_weights_tensor = [param for param in M_pretrain.parameters()]
w_pretrain_weights = weights_to_list_fast(w_pretrain_weights_tensor)

print('==> Beginning iteration over T=0 to T=500...')
data_ret = {}

# ! DON'T THINK WE NEED TWO MODELS
#1. Initialize your 2 models: M and M'
M = copy.deepcopy(M_pretrain)

M.train()

#initialize the loss functions, optimizers and schedulers for both models
optimizer = optim.SGD(M.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.MSELoss(reduction="sum")

# ! NO LR SCHEDULER TO BEGIN WITH - ADD THIS BACK LATER
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 

#the data that we are finetuning on (with x* at the beginning)
# ! FIGURE OUT HOW TO WRANGLE WITH THIS
data = batch_star + data_no_unlearned
data_loader = torch.utils.data.DataLoader(data, batch_size = args.finetune_batch_size, shuffle = False, num_workers = 2)

sigma_list  = []
print('T = ',len(data_loader))

#we need some lists for statistics
sigma_list = []
delta_weights_list = []
rolling_unl_error_list = [0]
ver_error_list = []

for ep in range(args.finetune_epochs):
    for main_idx,(inputs, targets) in enumerate(data_loader):
        print('Epoch: ',ep, 't:',main_idx)
        actual_idx = (ep*len(data_loader)) + main_idx
        inputs, targets = inputs.to(device), targets.to(device)
        M.train()
        
        if (actual_idx!=0) and (actual_idx)%args.checkpoint_freq ==0:
            M_interim = copy.deepcopy(M)
        
        optimizer.zero_grad()
        outputs_M = M(inputs)
        loss_M = criterion(outputs_M, targets)

        loss_M.backward()
        optimizer.step()
        curr_weights = w_pretrain_weights
        curr_error = 0
        M_interim = copy.deepcopy(M_pretrain)

        M.eval()
        #print('==> Getting Unlearning error...')

        for i,(img,label) in enumerate(hessian_loader):
            img = img.cuda()
            label = label.cuda()
            break
        if args.loss_func == 'regular':
            hessian_comp = hessian(M, hessian_criterion, data=(img, label), cuda=True)
        if args.loss_func == 'std':
            hessian_comp = hessian(M, std_loss, data=(img, label), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        sigma = np.sqrt(top_eigenvalues[-1])
        sigma_list.append(sigma)
        
        #very important: if you are now at the start of a new checkpoint, you want to update the weights to compare to:
        #if main_idx!=0 and main_idx%args.checkpoint_freq ==0:
        if (actual_idx!=0) and (actual_idx)%args.checkpoint_freq ==0:
            curr_weights = w_M_weights
            curr_error = rolling_unl_error
        
        #Now, save the weights of both M_(N+t) and M'_(N+t)
        M_weights_tensor = [param for param in M.parameters()]
        w_M_weights = weights_to_list_fast(M_weights_tensor)

        M_retrain_tensor = [param for param in M_retrained.parameters()]
        w_M_retrain_weights = weights_to_list_fast(M_retrain_tensor)

        #Now, get M''_(N+t)
        M_unlearned = copy.deepcopy(M)
        optimizer_unlearned = optim.SGD(M_unlearned.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        scheduler_unlearned = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_unlearned, T_max=200)

        M_interim.train()

        for i,(img,label) in enumerate(unlearned_loader):
            img = img.cuda()
            label = label.cuda()
            output_pre = M_interim(img)
            if args.loss_func == 'regular':
                loss_unl = criterion(output_pre, label)
            if args.loss_func == 'std':
                loss_unl = std_loss(output_pre,label)
            loss_unl.backward(retain_graph=True)
            grads = torch.autograd.grad(loss_unl, [param for param in M_interim.parameters()],create_graph = True)
        old_params = {}
        for i, (name, params) in enumerate(M.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += (ep+1) * args.lr * grads[i]
        for name, params in M_unlearned.named_parameters():
            params.data.copy_(old_params[name])
        M_unlearned_tensor = [param for param in M_unlearned.parameters()]
        w_M_unlearned_weights = weights_to_list_fast(M_unlearned_tensor)

            
        delta_weights = np.linalg.norm((np.array(w_M_weights) - np.array(curr_weights)))
        t_num = actual_idx%args.checkpoint_freq
        if main_idx ==0 and ep ==0:
            rolling_unl_error = 0
        if actual_idx%args.checkpoint_freq ==0 and actual_idx !=0:
            rolling_error = curr_error
        if actual_idx%args.checkpoint_freq != 0:
            rolling_unl_error = curr_error+ (args.lr * args.lr) * (t_num) * delta_weights * (sum(sigma_list[-t_num:])/len(sigma_list[-t_num:]))


        rolling_unl_error_list.append(rolling_unl_error)


ret = {}
ret['sigma'] = sigma_list
ret['rolling unlearning error'] = rolling_unl_error_list


import pickle
if not os.path.isdir('final_checkpoint_correlation_results'):
    os.mkdir('final_checkpoint_correlation_results')
path = f'./final_checkpoint_correlation_results/{args.model}_{args.dataset}_{args.loss_func}_{args.pretrain_batch_size}_{args.pretrain_epochs}_{args.finetune_batch_size}_{args.finetune_epochs}_{args.checkpoint_freq}.p'

pickle.dump(ret, open(path, 'wb'))
