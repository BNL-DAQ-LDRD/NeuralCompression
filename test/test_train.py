#!/usr/bin/env python
from neuralcompress.procedures.train import train

# Construct encoder network
conv_1 = {
    'out_channels': 8,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1]
}
conv_2 = {
    'out_channels': 16,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1]
}
conv_3 = {
    'out_channels': 32,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1]
}
conv_4 = {
    'out_channels': 32,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1]
}

# Construct decoder network
deconv_1 = {
    'out_channels': 16,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_2 = {
    'out_channels': 8,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_3 = {
    'out_channels': 4,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_4 = {
    'out_channels': 2,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
conv_args_list   = [conv_1, conv_2, conv_3, conv_4]
deconv_args_list = [deconv_1, deconv_2, deconv_3, deconv_4]

config = {
    'data_path': '/data/datasets/sphenix/highest_framedata_3d/outer',
    'data': {
        'batch_size' : 32,
        'train_sz'   : 960,
        'valid_sz'   : 320,
        'test_sz'    : 320,
        'is_random'  : True,
    },
    'model': {
        'name'             : 'bcae',
        'image_channels'   : 1,
        'code_channels'    : 8,
        'conv_args_list'   : conv_args_list,
        'deconv_args_list' : deconv_args_list,
        'activ'            : {
            'name'           : 'leakyrelu',
            'negative_slope' : .2
        },
        'norm': 'instance',
    },
    'device': 'cuda',
    'optimizer': {
        'name' : 'AdamW',
        'lr'   : 0.01,
    },
    'scheduler': {
        'name'      : 'step',
        'step_size' : 20,
        'gamma'     : .95,
        'verbose'   : True,
    },
    'loss': {
        'transform'        : 'bcae',
        'weight_pow'       : .1,
        'clf_threshold'    : .5,
        'target_threshold' : 64,
        'gamma'            : 2,
        'eps'              : 1e-8,
        'lambda'           : 20000, # initial lambda,
        'verbose'          : True,
    },
    'epochs': 2000,
    'save_path': ( # must be an absolute path
        '/home/yhuang2/PROJs/NeuralCompression_results/checkpoints/'
    ),
    'save_freq': 20
}

train(config)
