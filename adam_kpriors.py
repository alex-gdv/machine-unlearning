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

def parameters_grads_to_vector(parameters):
    vec = []
    for param in parameters:
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)

class AdamKpriors(Adam):
    def __init__(self, params):
        super(AdamKpriors, self).__init__(params)

    def step(self, closure_forget, closure_memory):
        # need to count how many datapoints in total (batch + memory) and divide gradient by this amount

        parameters = self.param_groups[0]["params"]
        loss = closure_forget()
        loss.backward()
        grad = parameters_grads_to_vector(parameters)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load forget data
forget_dataset = UTKFaceRegression("data/forget.json")
forget_dataloader = DataLoader(forget_dataset, batch_size=32, shuffle=True)

base_model = ResNet50Regression() 
base_checkpoint = torch.load(
    f"./checkpoints/0.0.1/epoch_100.pt"
)
base_model.load_state_dict(base_checkpoint["model_state_dict"])

criterion = torch.nn.MSELoss(reduction="sum")

model = copy.deepcopy(base_model)
model = model.to(device)

optimizer = AdamKpriors(base_model.parameters())

for batch, (inputs, labels) in enumerate(forget_dataloader):
    batch_metrics = {}
    inputs = inputs.to(device)
    labels = labels.to(device).float()

    def closure_forget():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = -loss
        return loss

    optimizer.step()
    break
