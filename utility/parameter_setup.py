import sys, os
import yaml
import pandas as pd
import numpy as np
import ast
import argparse

# ------------- Constants relevant to parameters ------------------

# A sensible set of default parameters (can run an experiment without setting these params)

# To avoid printing warnings multiple times
warning_1_printed = False
warning_2_printed = False

default_params = {
    'num_classes': -1,  # Required param. Added here to specify type (int)
    'batch_size': 256,
    'dataset_percent': 1.0,
    'device': 'cuda',
    # 'device': 'cpu',
    # 'distributed_training': 1,
    'distributed_training': 0,
    'early_stop_patience': 10,
    'epochs': 5,
    'exp_id': 'default_exp_id',
    'freeze_conv_base': 0,
    'kaiming_init': 0,
    'learning_rate': 0.1,
    'loss_function': 'cross_entropy',
    'lr_scheduler_gamma': 1.0,
    'lr_scheduler_num_steps': 1,
    'mixed_precision': 1,
    # 'mixed_precision': 0,
    # 'model': 'resnet18',
    'momentum': 0.9,    # For SGD
    'weight_decay': 0.0,    # For SGD, Adam
    'optimizer': 'sgd',
    'gradient_clipping_value': 0.0,     # 0.0 means no clipping
    'output_dir': 'output',
    'pretrained': 0,
    'random_seed': 1234567,
    'test_batch_size': 256,
    'test_loss_function': 'cross_entropy',
    'train_type': 'standard',
    'transfer_type': 'from_scratch',
    'verbose': 1,
    'ann_input_nodes': -1,  # Added here to specify type (int)
    'load_attack_data': 1,
    # 'attack_data_dir': 'data/attack_data_unnormalized/cnn_mnist_st_ig_attacks_eval_chen-et-al-eval_ig-100',
    'attack_data_dir': 'yyy',
    'severity': -1,
    'checkpoint_every_n_epochs': 10,    # Give -1 or 0 to disable checkpointing
    'job_id': '',   # To get job_id from job script
    'image_size': -1,   # Ability specify a non-default resize for some datasets (CIFAR, ImageNet, Waterbirds, CelebA)
    'attack_batch_size': -1,
    'normalization': 'none',    # 'none', 'global', 'per_image'
    'wrn_widen_factor': -1,  # For wide_resnet
    'wrn_depth': -1,  # For wide_resnet
    'eval_top_K': 100,  # 100 for small datasets, must specify about 1000 for large datasets
}

# Required params or the bare minimum needed to specify an experiment (cannot run an experiment without these)

required_params = [
    'classifier',
    'data_dir',
    'dataset',
    'input_shape',
    'model',
    'num_classes',
]

allowed_values = {
    'dataset': ['cifar10', 'cifar10-c', 'cifar10-p', 'cifar10_augmented', 'cifar100', 'cifar100_augmented', 'mnist',
                'mnist-c', 'fashion_mnist', 'fashion_mnist-c', 'flower102', 'svhn', 'restricted_imagenet',
                'restricted_imagenet_balanced', 'rimb-c', 'imagenet', 'waterbirds', 'celebA'],

    'model': ['small_cnn', 'resnet18', 'ann', 'wide_resnet'],
    'transfer_type': ['from_scratch', 'fixed_feature', 'full_network'],
    'device': ['cpu', 'cuda'],
    'optimizer': ['sgd', 'adam', 'adagrad'],

    'loss_function': ['cross_entropy', 'gradient_regularization', 'gradient_attack_regularization', 'ig_regularization',
                      'eg_regularization', 'layer_gradient_regularization', 'layer_ig_regularization',
                      'layer_gradient_attack_regularization', 'changing', 'every_other', 'ig_regularization--percent_lambda'],

    'normalization': ['none', 'global', 'per_image']
}

# -----------------------------------------------------------------
# Parameter checking functions


# def get_params_filename_from_cmd_args():
#     num_args = len(sys.argv)
#     if num_args != 2:
#         print('Usage:')
#         print('python train_dnn_mnist my_config.yaml')
#         sys.exit(1)
#
#     filename = sys.argv[1]
#     if not os.path.isfile(filename):
#         print(f'Config File {filename} does not exist')
#         sys.exit(1)
#
#     return filename


def get_cmd_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', required=True)

    args, unknown = parser.parse_known_args()
    args = vars(args)

    # print(args)

    param_file = args['params']

    if not os.path.isfile(param_file):
        print(f'Config File {param_file} does not exist')
        sys.exit(1)

    return param_file, unknown


def parse_params(params):
    # Parse lists, convert to ints
    for key, val in params.items():
        try:
            if type(val) == str and val[0] == '[':
                params[key] = ast.literal_eval(val)

            if type(val) == str and val[0] == '{':
                params[key] = ast.literal_eval(val)

            if type(val) == np.int64 or type(val) == np.int32:
                params[key] = int(val)

            if key in default_params:
                correct_type = type(default_params[key])
                params[key] = correct_type(val)
        except:
            print(f'Error parsing param. key: {key}, val:{val}')

    return params


def get_params_from_file(params_filename):
    if params_filename.endswith('.yaml'):
        file = open(params_filename, "r")
        params = yaml.load(file, Loader=yaml.FullLoader)
        all_params = [params]

    elif params_filename.endswith('.xlsx'):
        df = pd.read_excel(params_filename, engine='openpyxl')
        df = df[df['run'] == 1]

        # assert df.shape[0] == 1, 'Exactly one row in the params Excel file must have run = 1'
        # params = df.iloc[0].to_dict()

        all_params = []
        for i in range(len(df)):
            params = df.iloc[i].to_dict()
            params = parse_params(params)
            all_params.append(params)

    else:
        assert False, 'Unsupported params file type'

    return all_params


def assert_dependencies(params):
    global warning_1_printed
    global warning_2_printed

    if params['device'] == 'cpu':
        assert not params['distributed_training'], 'Distributed training is not applicable in CPU'
        assert not params['mixed_precision'], 'Mixed precision training is not applicable in CPU'
    elif params['device'] == 'cuda':
        if not params['distributed_training'] and not warning_1_printed:
            print('\n************** WARNING: Running on CUDA with no distributed training **************\n')
            warning_1_printed = True
        if not params['mixed_precision'] and not warning_2_printed:
            print('\n************** WARNING: Running on CUDA with no mixed precision **************\n')
            warning_2_printed = True

    if params['transfer_type'] == 'fixed_feature':
        assert params['freeze_conv_base'] == 1
    elif params['transfer_type'] == 'full_network':
        assert params['freeze_conv_base'] == 0
    elif params['transfer_type'] == 'from_scratch':
        assert params['freeze_conv_base'] == 0 \
               or params['pretrained'] == 1     # To allow running a2, a3, a4, a5 with transfer_type = from_scratch

    if params['loss_function'] == 'changing':
        assert 'changing_loss_functions' in params \
            and params['changing_loss_functions'] != 'x', 'Must specify the loss function change schedule'


def sanitize_params(params):
    # Ensure all required params are given
    for p in required_params:
        assert p in params, f'Required parameter "{p}" is missing'

    # Set any default parameters
    for key, val in default_params.items():
        if key not in params:
            params[key] = val

    # Ensure the parameter values are valid/ allowed
    for key, val in allowed_values.items():
        pval = params[key]
        assert pval in val, f'Invalid parameter value "{key}: {pval}". Allowed values are {val}'

    # Ensure proper dependencies between params
    assert_dependencies(params)

    # Remove irrelevant params (given as 'x')
    for k in list(params.keys()):
        if params[k] == 'x':
            del params[k]


def override_cmd_line_args(params, other_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', required=True)

    all_params = {**params, **default_params}   # Merge the two dicts, so we can support all params as cmd line args

    for key, val in all_params.items():
        # print('key: ', key, 'val: ', val)
        parser.add_argument(f'--{key}', type=type(val), required=False)

    args = parser.parse_args()

    args = vars(args)
    # print(args)

    for key, val in args.items():
        # assert key in all_params, f'Invalid command line arg given: {key}'    # Ensure the param is valid
        if key in all_params and val is not None:
            params[key] = val

    return params
