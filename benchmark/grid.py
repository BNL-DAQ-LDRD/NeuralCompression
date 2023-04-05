"""
Generate command with a parameter grid
"""
from collections import OrderedDict
from itertools import product

def form_command(prefix, param_dict):
    """
    form command with prefix and a dictionary of parameters
    """
    cmd = [prefix]
    for key, val in param_dict.items():
        if isinstance(val, bool):
            if  val:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key} {val}')
    return ' '.join(cmd)


def form_grid(params):
    """
    Given parameters choices, generate a list of parameter dictionaries
    Example: params = {'a': [1, 2], 'b': [3, 4]}
        form_grid(params) = [
            {'a': 1, 'b': 3},
            {'a': 1, 'b': 4},
            {'a': 2, 'b': 3},
            {'a': 2, 'b': 4}
        ]
    We can use sklearn.model_selection.ParameterGrid, but Ray doesn't
    like it :).
    """
    params_ordered = OrderedDict(params)
    keys = params_ordered.keys()
    param_grid = []
    for vals in product(*params_ordered.values()):
        param_grid.append(dict(zip(keys, vals)))
    return param_grid


def main():
    """
    Given script name (prefix), and parameter choices,
    generate a list of commands and write them to a bash
    file to be run afterward.
    """
    prefix = 'python gpu_inference.py'
    params = {
        'data_size': [512],
        'batch_size': [2, 4, 8, 16, 32, 64, 128],
        'pin_memory': [True, False],
        'num_workers': [1, 2, 3, 4, 6, 8, 12],
        'benchmark': [True, False],
        'prefetch_factor': [1, 2]
    }
    param_grid = form_grid(params)

    cmds = []
    for i, param in enumerate(param_grid):
        cmd = form_command(prefix, param)
        cmd += f' --result_fname results/result_{i}.csv'
        cmds.append(cmd)

    with open('commands.sh', 'w') as file_handle:
        file_handle.write('\n'.join(cmds))


if __name__ == '__main__':
    main()
