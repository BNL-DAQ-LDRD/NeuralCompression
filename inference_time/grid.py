"""
Generate command with a parameter grid
"""
from sklearn.model_selection import ParameterGrid

def form_command(prefix, param_dict):
    cmd = [prefix]
    for key, val in param_dict.items():
        if type(val) == bool:
            if  val:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key} {val}')
    return ' '.join(cmd)


params = {
    'data_size': [512],
    'batch_size': [2, 4, 8, 16, 32, 64, 128],
    'pin_memory': [True, False],
    'num_workers': [1, 2, 3, 4, 6, 8, 12],
    'benchmark': [True, False],
    'prefetch_factor': [1, 2]
}



grid = list(ParameterGrid(params))
prefix = 'python inference_time.py'
cmds = [form_command(prefix, g) + f' --result_fname results/result_{i}.csv' for i, g in enumerate(grid)]
with open('commands.sh', 'w') as fh:
    fh.write('\n'.join(cmds))
