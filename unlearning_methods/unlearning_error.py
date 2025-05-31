from pyhessian import hessian 
import torch
import numpy as np

from unlearning_methods.util import weights_to_list

class UnlearningError:
    def __init__(self, exact_unlearn_model, args):
        self.lr = args.lr
        self.exact_unlearn_weights = weights_to_list( [param for param in exact_unlearn_model.parameters()])
        self.sigmas = []

    def calculate_error(self, model, criterion, dataloader, step):        
        inputs, targets = dataloader[0]
        hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        
        sigma = np.sqrt(top_eigenvalues[-1]) # standard deviation of the hessian values in the direction with most variance
        self.sigmas.append(sigma)

        # save model weights
        approx_unlearn_weights = weights_to_list([param for param in model.parameters()])
        delta_weights = np.linalg.norm((np.array(self.exact_unlearn_weights) - np.array(approx_unlearn_weights))) # verification loss
                
        # lr^2 * num_steps -> how many backprop steps this is accounting for 
        #               * diff in weights -> penalise going too far from original model as a proxy for exact unlearn
        #               * std of hessian -> converging or not
        return (self.lr * self.lr) * (step) * delta_weights * (sum(self.sigmas[-step:])/len(self.sigmas[-step:]))
