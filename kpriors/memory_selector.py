"""Source:
@article{khan2021knowledge,
  title = {Knowledge-Adaptation Priors},
  author = {Khan, Mohammad Emtiyaz and Swaroop, Siddharth},
  journal = {Advances in Neural Information Processing Systems},
  year = {2021}
}
GitHub:
https://github.com/team-approx-bayes/kpriors

"""

import torch
import torch.nn.functional as F
import numpy as np


# what is the role of the softmax function here???
def softmax_jacobian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s


def select_memory_points(dataloader, model, memory_size, device):
    with torch.no_grad():
        model.to(device)

        top_scores = torch.Tensor([], device="cpu")
        top_inputs = torch.Tensor([]).to(device)
        top_labels = torch.Tensor([]).to(device)
        top_soft_labels = torch.Tensor([]).to(device)

        for batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            lamb = softmax_jacobian(outputs)
            if device.type == "cuda":
                lamb = lamb.cpu()
            lamb = lamb.detach()
            top_scores = torch.cat([top_scores, lamb])
            top_inputs = torch.cat([top_inputs, inputs])
            top_labels = torch.cat([top_labels, labels])
            top_soft_labels = torch.cat([top_soft_labels, outputs])
            if top_scores.shape[0] > memory_size:
                _, indices = top_scores.sort(descending=True)
                top_scores = top_scores[indices[:memory_size]]
                top_inputs = top_inputs[indices[:memory_size]]
                top_labels = top_labels[indices[:memory_size]]
                top_soft_labels = top_soft_labels[indices[:memory_size]]

        memory = {}
        memory["inputs"] = top_inputs.to(device)
        memory["labels"] = top_labels.to(device)
        memory["soft_labels"] = top_soft_labels.to(device)

        return memory
