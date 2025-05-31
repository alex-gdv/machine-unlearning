import numpy as np
import torch

from unlearning_methods.util import weights_to_list
from model.model import ResNetRegression


def get_verification_error(exact_unlearning_model, approx_unlearning_model):
    exact_unlearning_weights = weights_to_list([param for param in exact_unlearning_model.parameters()])
    approx_unlearning_weights = weights_to_list([param for param in approx_unlearning_model.parameters()])

    return np.linalg.norm((np.array(exact_unlearning_weights) - np.array(approx_unlearning_weights)))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_unlearning_model", type=str, required=True)
    parser.add_argument("--approx_unlearning_model", type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    exact_unlearning_model = ResNetRegression()
    exact_unlearning_model = exact_unlearning_model.to(device)
    exact_unlearning_checkpoint = torch.load(
        args.exact_unlearning_model, 
        map_location=device
    )
    exact_unlearning_model.load_state_dict(exact_unlearning_checkpoint["model_state_dict"])
    
    approx_unlearning_model = ResNetRegression()
    approx_unlearning_model = approx_unlearning_model.to(device)
    approx_unlearning_checkpoint = torch.load(
        args.approx_unlearning_model, 
        map_location=device
    )
    approx_unlearning_model.load_state_dict(approx_unlearning_checkpoint["model_state_dict"])
    
    print(get_verification_error(exact_unlearning_model, approx_unlearning_model))
    