import torch
from torch.optim.adam import Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from model.model import ResNet50Regression


class AdamKpriors(Adam):
    def __init__(self, params):
        super(AdamKpriors, self).__init__(params)

    def step(self, closure_forget, closure_memory):

        parameters = self.param_groups[0]["params"]
        p = parameters_to_vector(parameters)
        self.state["mu"] = p.clone().detach()
        mu = self.state["mu"]
        


base_model = ResNet50Regression() 
base_checkpoint = torch.load(
    f"./checkpoints/0.0.1/epoch_100.pt"
)
base_model.load_state_dict(base_checkpoint["model_state_dict"])

optimizer = AdamKpriors(base_model.parameters())
optimizer.step()
