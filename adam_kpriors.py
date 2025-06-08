import torch
from torch.optim.adam import Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy

from model.model import ResNet50Regression

from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import argparse
import torch
import os
import copy

from model.dataset import UTKFaceRegression
from kpriors.memory_selector import select_memory_points


class AdamKpriors(Adam):
    def __init__(self, params, original_memory_outputs, prior_prec, original_params):
        super(AdamKpriors, self).__init__(params)
        self.original_memory_outputs = original_memory_outputs
        self.prior_prec = prior_prec
        self.original_params = original_params

    def step(self, closure_forget, closure_memory):
        self._cuda_graph_capture_health_check()

        with torch.enable_grad():
            loss_forget = closure_forget()
        
        # K-priors: Calculate difference between original model outputs
        # and unlearning model outputs.
        unlearning_memory_outputs = closure_memory()
        delta_memory_ouputs = unlearning_memory_outputs.detach() - self.original_memory_outputs

        # Adam optimiser code.
        with torch.no_grad():
            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]

                self._init_group(
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                )

                # K-priors: Compute the vector-Jacobian product (VJP).
                vjp_grads = torch.autograd.grad(
                    unlearning_memory_outputs,
                    params_with_grad,
                    grad_outputs=delta_memory_ouputs,    
                )

                for i, param in enumerate(params_with_grad):
                    grad = grads[i]
                    exp_avg = exp_avgs[i]
                    exp_avg_sq = exp_avg_sqs[i]
                    step_t = state_steps[i]
                    original_param = self.original_params[i]

                    # K-priors: Add VJP to gradient.
                    vjp_grad = vjp_grads[i]
                    grad = grad.add(vjp_grad.detach())

                    # update step
                    step_t += 1

                    # K-priors: Add previous weights to gradient.
                    if self.prior_prec != 0:
                        grad = grad.add(original_param, alpha=-self.prior_prec)
                        grad = grad.add(param, alpha=self.prior_prec)

                    # Decay the first and second moment running average coefficient
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    bias_correction1 = 1 - beta1**step_t
                    bias_correction2 = 1 - beta2**step_t

                    step_size = lr / bias_correction1

                    bias_correction2_sqrt = bias_correction2**0.5

                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                    param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss_forget


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_train_dataset = UTKFaceRegression("data/train.json")
base_train_dataloader = DataLoader(base_train_dataset, batch_size=4, shuffle=True)

# load forget data
forget_dataset = UTKFaceRegression("data/forget.json")
forget_dataloader = DataLoader(forget_dataset, batch_size=4, shuffle=True)

model = ResNet50Regression() 
base_checkpoint = torch.load(
    f"./checkpoints/0.0.1/epoch_100.pt",
    map_location=device,
    weights_only=False
)
model.load_state_dict(base_checkpoint["model_state_dict"])
model.to(device)

criterion = torch.nn.MSELoss(reduction="sum")

# select memory points
memory_size = 16 # int(len(base_train_dataloader) * 4 * 0.05)
memory = select_memory_points(base_train_dataloader, model, memory_size, device)

# We define prior precision for the original and unlearning models.
# The authors use prior precision values ranging from 5 to 50.
# They use a value of 5 for image classification with an MLP.
PRIOR_PREC = 5

original_params = [p.detach().clone() for p in model.parameters()]

optimizer = AdamKpriors(
    model.parameters(),
    memory["outputs"],
    PRIOR_PREC,
    original_params   
)
for batch, (inputs, labels) in enumerate(forget_dataloader):
    batch_metrics = {}
    inputs = inputs.to(device)
    labels = labels.to(device).float()

    def closure_forget():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = -loss
        loss.backward()
        return loss.detach()

    def closure_memory():        
        outputs = model(memory["inputs"])
        return outputs

    loss_forget = optimizer.step(closure_forget, closure_memory)
    if batch % 10 == 0:
        print(loss_forget)
