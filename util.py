from tabulate import tabulate


def output_statistics(settings, metrics, size, table_out=False):
    if table_out:
        print(tabulate(list(settings.items()) + [(k, v/size) for k, v in metrics.items()]), flush=True)
    else:
        print(*[(k, v/size) for k, v in metrics.items()], flush=True)
