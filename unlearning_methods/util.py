from tabulate import tabulate
import torch


def output_statistics(settings, metrics, size, table_out=False):
    if table_out:
        print(tabulate(list(settings.items()) + [(k, v/size) for k, v in metrics.items()]), flush=True)
    else:
        print(*[(k, v/size) for k, v in metrics.items()], flush=True)


def weights_to_list(weights):
  with torch.no_grad():
    weights_list = []
    for weight in weights:
      list_t = weight.view(-1).tolist()
      weights_list = weights_list + list_t

    return weights_list