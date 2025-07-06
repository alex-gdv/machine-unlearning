"""Code adapted from https://github.com/team-approx-bayes/kpriors."""

import torch


def select_memory_points(dataloader, model, memory_size, device):
    top_norms = torch.Tensor([], device="cpu")
    top_inputs = torch.Tensor([]).to(device)
    top_outputs = torch.Tensor([]).to(device)

    criterion = torch.nn.MSELoss(reduction="none")

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        norms = []
        for i in range(loss.size()[0]):
            model.zero_grad()
            loss[i].backward(retain_graph=True)

            grad_vector = torch.cat([
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ])

            norm = torch.norm(grad_vector, p=2)
            norms.append(norm)

        norms = torch.Tensor(norms, device="cpu")
        top_norms = torch.cat([top_norms, norms])
        top_inputs = torch.cat([top_inputs, inputs])
        top_outputs = torch.cat([top_outputs, outputs])

        if top_norms.size()[0] > memory_size:
            print("hello")
            _, indices = top_norms.sort(descending=True)
            top_norms = top_norms[indices[:memory_size]]
            top_inputs = top_inputs[indices[:memory_size]]
            top_outputs = top_outputs[indices[:memory_size]]

    memory = {}
    memory["inputs"] = top_inputs
    memory["outputs"] = top_outputs

    return memory
