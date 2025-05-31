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
    def __init__(self, params):
        super(AdamKpriors, self).__init__(params)

    def step(self, closure_forget, closure_memory):
        self._cuda_graph_capture_health_check()

        with torch.enable_grad():
            loss_forget = closure_forget()
            loss_forget.backward()
        
        loss_memory = closure_memory()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            params = self.group["params"]
            for i, param in enumerate(params):
                if i < 10:
                    print(param.grad)
                # grad = grads[i] if not maximize else -grads[i]
                # exp_avg = exp_avgs[i]
                # exp_avg_sq = exp_avg_sqs[i]
                # step_t = state_steps[i]

                # # update step
                # step_t += 1

                # if weight_decay != 0:
                #     grad = grad.add(param, alpha=weight_decay)

                # device = param.device

                # # Decay the first and second moment running average coefficient
                # exp_avg.lerp_(grad, 1 - beta1)

                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                # step = _get_value(step_t)

                # bias_correction1 = 1 - beta1**step
                # bias_correction2 = 1 - beta2**step

                # step_size = lr / bias_correction1

                # bias_correction2_sqrt = bias_correction2**0.5

                # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                # param.addcdiv_(exp_avg, denom, value=-step_size)


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

criterion = torch.nn.MSELoss(reduction="sum")

# select memory points
# memory_size = int(len(base_train_dataloader) * 4 * 0.05)
# memory = select_memory_points(base_train_dataloader, model, memory_size, device)

# optimizer = AdamKpriors(model.parameters())
# for batch, (inputs, labels) in enumerate(forget_dataloader):
#     batch_metrics = {}
#     inputs = inputs.to(device)
#     labels = labels.to(device).float()

#     def closure_forget():
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss = -loss
#         loss.backward()
#         return loss.detach()

#     def closure_memory():
#         optimizer.zero_grad()
        
#         outputs = model(memory["inputs"])
#         return outputs

#     optimizer.step(closure_forget, closure_memory)
#     break
