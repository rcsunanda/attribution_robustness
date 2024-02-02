import hashlib
import os
from pprint import pformat

import torch
import numpy as np
from tqdm import tqdm

from utility import parameter_setup, data_prep, common, attribution_robustness_metrics, misc
from models import model_utility
from int_tl import ifia_attack, gradient_attack, feature_significance
from utility import visualization

# To run this script, simply set the following 2 params to the correct parameter file and run the following command:
# python -m int_tl.generate_attacks

load_params_from_file = True
params_filename = 'param_files/attack_evaluations.xlsx'

# Following params are used only when above load_params_from_file is False
# We keep it here for reference
# code_params = {
#     # 'dataset': 'restricted_imagenet_balanced',
#     # 'data_dir': '/home/sunanda/research/Datasets/imagenet/ILSVRC/Data/CLS-LOC',
#     # 'num_classes': 14,
#     # 'model_state_path': 'saved_models/b1_resnet_rimb_st_Feb14-15_05_30/best_model_state.pt',
#     # 'model': 'resnet18',
#
#     'dataset': 'mnist',
#     'data_dir': 'data',
#     'num_classes': 10,
#     'model_state_path': 'saved_models/cnn_mnist_st_unnormalized_e200/final_model_state.pt',
#     'model': 'small_cnn',
#
#     'dataset_percent': 1.0,
#     'batch_size': 512,
#     'test_batch_size': 512,
#     'transfer_type': 'from_scratch',
#
#     # Model & training
#     # 'model': 'small_cnn',
#     'pretrained': 0,
#     'kaiming_init': 0,
#     'freeze_conv_base': 0,
#     'distributed_training': 0,
#     'classifier': [],
#     'device': 'cuda:0',   # eg: cpu, cuda:0, cuda:1
#     # 'device': 'cpu',   # eg: cpu, cuda:0, cuda:1
# }


def save_dataset_to_disk(XX, labels, XX_attacks, directory):
    directory = f'{directory}'

    print(f'Saving dataset to directory: {directory}')

    torch.save(XX, f'{directory}/XX.pt')
    torch.save(labels, f'{directory}/labels.pt')
    torch.save(XX_attacks, f'{directory}/XX_attacks.pt')


def load_dataset_from_disk(params):
    directory = f"{params['attack_data_dir']}/{params['exp_id']}"
    assert os.path.exists(directory), f'Dataset directory does not exist: {directory}'

    print(f'Loading dataset from directory: {directory}')

    XX = torch.load(f'{directory}/XX.pt')
    labels = torch.load(f'{directory}/labels.pt')
    XX_attacks = torch.load(f'{directory}/XX_attacks.pt')

    print(f'XX.shape: {XX.shape}')
    print(f'XX_attacks.shape: {XX_attacks.shape}')
    print(f'labels.shape: {labels.shape}')

    return XX, labels, XX_attacks


def calculate_accuracy(network, X, y, batch_size):
    num_samples = len(X)
    correct_preds = 0

    for i in range(0, num_samples, batch_size):
        X_batch = X[i: i + batch_size]
        y_batch = y[i: i + batch_size]

        # inputs = inputs.to(self.device_obj, non_blocking=True)
        # labels = labels.to(self.device_obj, non_blocking=True)

        with torch.no_grad():  # We do not intend to call backward(), as we don't backprop and optimize
            outputs = network(X_batch)
            _, y_pred = torch.max(outputs.data, 1)

            correct_preds += (y_pred == y_batch).sum().item()

    test_accuracy = correct_preds / num_samples

    return test_accuracy


def generate_ig_attacks(network, testset, params):
    print('generate_ig_attacks')

    epsilon = params['ig_epsilon']
    alpha = params['ig_alpha']
    num_iter = params['ig_adv_num_iter']
    adv_ig_steps = params['ig_adv_ig_steps']
    k_top = params['ig_k_top']
    random_start = False

    print('Generating attack examples')
    print(f'Params. epsilon: {epsilon}, alpha: {alpha}, num_iter: {num_iter}, adv_ig_steps: {adv_ig_steps}, '
          f'k_top: {k_top}, random_start: {random_start}')

    XX = []
    labels = []
    XX_attacks = []

    for X_batch, y_batch, *metadata in tqdm(testset):
        X_batch = X_batch.to(params['device'])
        y_batch = y_batch.to(params['device'])

        for x, y in zip(X_batch, y_batch):
            # print('\n----------------------------------\n')
            x_attack = ifia_attack.create_topk_ifia_attack(network, x, y, epsilon, alpha, num_iter, adv_ig_steps, k_top, random_start)
            # print(f'x_attack.shape: {x_attack.shape}')
            XX.append(x)
            labels.append(y)
            XX_attacks.append(x_attack)

    XX = torch.stack(XX)
    labels = torch.stack(labels)
    XX_attacks = torch.stack(XX_attacks)

    print(f'XX.shape: {XX.shape}')
    print(f'XX_attacks.shape: {XX_attacks.shape}')
    print(f'labels.shape: {labels.shape}')

    return XX, labels, XX_attacks


def evaluate_attacks_ig_maps(network, XX, yy, XX_attacks, params):
    print('evaluate_ig_attacks')

    eval_top_K = 100    # No. of top features to consider for intersection metric
    batch_size = params['test_batch_size']
    ig_steps = 100

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)

    print(XX.device)
    print(f'Computing IG maps of original examples. ig_steps: {ig_steps}')
    XX_ig = feature_significance.compute_integrated_gradients_in_batches(network, XX, yy, baselines, ig_steps, batch_size, True)

    print(f'Computing IG maps of attack examples. ig_steps: {ig_steps}')
    XX_attacks_ig = feature_significance.compute_integrated_gradients_in_batches(network, XX_attacks, yy, baselines, ig_steps, batch_size, True)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    # Important, the following sum + normalization is also required for correct metric calculation (Kendall etc)

    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Sum across channels
    XX_ig = XX_ig / torch.sum(XX_ig)  # Normalize the heatmap

    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Sum across channels
    XX_attacks_ig = XX_attacks_ig / torch.sum(XX_attacks_ig)  # Normalize the heatmap


    print(f'Computing metrics. eval_top_K: {eval_top_K}')

    spearman_list = []
    kendall_list = []
    intersection_list = []

    for x_ig, x_attack_ig in zip(XX_ig, XX_attacks_ig):
        spearman = attribution_robustness_metrics.compute_spearman_rank_correlation(x_ig, x_attack_ig)
        kendall = attribution_robustness_metrics.compute_kendalls_correlation(x_ig, x_attack_ig)
        inersection = attribution_robustness_metrics.compute_topK_intersection(x_ig, x_attack_ig, K=eval_top_K)

        spearman_list.append(spearman)
        kendall_list.append(kendall)
        intersection_list.append(inersection)

    avg_spearman = np.mean(spearman_list)
    avg_kendall = np.mean(kendall_list)
    avg_intersection = np.mean(intersection_list)

    # print('spearman_list: ', spearman_list)

    print(f'avg_spearman: {avg_spearman: .4f}')
    print(f'avg_kendall: {avg_kendall: .4f}')
    print(f'avg_intersection: {avg_intersection * 100: .2f} %')

    return avg_spearman, avg_kendall, avg_intersection


def visualize_ig_maps(network, XX, labels, XX_attacks, params):
    print('visualize_ig_maps')

    print(f'Orig XX checksum \t\t: {misc.tensor_checksum(XX)}')
    print(f'Orig XX_attacks checksum \t: {misc.tensor_checksum(XX_attacks)}')

    num_samples = 6
    ig_steps = 100
    batch_size = params['test_batch_size']

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)

    XX = XX[:num_samples]
    yy = labels[:num_samples]
    XX_attacks = XX_attacks[:num_samples]

    XX_ig = feature_significance.compute_integrated_gradients_in_batches(network, XX, yy, baselines, ig_steps, batch_size)
    XX_attacks_ig = feature_significance.compute_integrated_gradients_in_batches(network, XX_attacks, yy, baselines, ig_steps, batch_size)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Channels first

    print(f'XX checksum \t\t: {misc.tensor_checksum(XX)}')
    print(f'XX_attacks checksum \t: {misc.tensor_checksum(XX_attacks)}')
    print('')
    print(f'XX_ig checksum \t\t: {misc.tensor_checksum(XX_ig)}')
    print(f'XX_attacks_ig checksum \t: {misc.tensor_checksum(XX_attacks_ig)}')

    plot1_args = []
    for x, x1, x2, x3 in zip(XX, XX_ig, XX_attacks, XX_attacks_ig):
        plot1_args.append({'x': x.cpu(), 'plot_func': visualization.plot_grayscale})
        plot1_args.append({'x': x1.cpu()})
        plot1_args.append({'x': x2.cpu(), 'plot_func': visualization.plot_grayscale})
        plot1_args.append({'x': x3.cpu()})

    epsilon = params['ig_epsilon']
    alpha = params['ig_alpha']
    num_iter = params['ig_adv_num_iter']
    adv_ig_steps = params['ig_adv_ig_steps']

    param_str = f'Params. epsilon {epsilon}, alpha {alpha}, num_iter {num_iter}, adv_ig_steps {adv_ig_steps}'

    visualization.plot_grid(num_samples, 4, f'IG_attack_maps-{param_str}', visualization.plot_feature_sig, plot1_args)

    # visualize_histograms(XX, XX_attacks)

    common.save_all_figures(params['output_dir'])

    return plot1_args


def visualize_histograms(XX, XX_attacks):
    print('visualize_histograms')

    num_samples = len(XX)

    XX = XX.cpu()
    XX_attacks = XX_attacks.cpu()

    X_hist = []
    X_bins = []
    X_attacks_hist = []
    X_attacks_bins = []

    for x, x_attack in zip(XX, XX_attacks):
        # print('---------- x -----------')
        # print(x)
        # print('########### x_attack ############')
        # print(x_attack)

        x_hist, x_bins = np.histogram(x, bins=10)
        x_attack_hist, x_attack_bins = np.histogram(x_attack, bins=10)

        # print('\n---------------------------------' * 2)
        # print('@@@@@@@@ x_hist, x_bins @@@@@@@@@')
        # print('x_hist: ', x_hist)
        # print('x_bins: ', x_bins)
        # print('!!!!!!!!! x_attack_hist, x_attack_bins !!!!!!!!!!!')
        # print('x_attack_hist: ', x_attack_hist)
        # print('x_attack_bins: ', x_attack_bins)

        X_hist.append(x_hist)
        X_bins.append(x_bins)
        X_attacks_hist.append(x_attack_hist)
        X_attacks_bins.append(x_attack_bins)

    plot1_args = []
    for x, xh, xb, xa, xah, xab in zip(XX, X_hist, X_bins, XX_attacks, X_attacks_hist, X_attacks_bins):
        plot1_args.append({'x': x})
        plot1_args.append({'hist': xh, 'bins': xb, 'plot_func': visualization.plot_histogram})
        plot1_args.append({'x': xa})
        plot1_args.append({'hist': xah, 'bins': xab, 'plot_func': visualization.plot_histogram})

    visualization.plot_grid(num_samples, 4, 'Histograms', visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')


def evaluate_accuracy(network, XX, labels, XX_attacks, params):
    print('evaluate_accuracy')

    batch_size = params['test_batch_size']

    standard_accuracy = calculate_accuracy(network, XX, labels, batch_size)
    attack_accuracy = calculate_accuracy(network, XX_attacks, labels, batch_size)

    print(f'standard_accuracy: {standard_accuracy * 100: .2f} %')
    print(f'attack_accuracy: {attack_accuracy * 100: .2f} %')


def main():
    # --------------------- Setup ---------------------

    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    if load_params_from_file:
        print(f'Loading parameters from {params_filename}')
        all_params = parameter_setup.get_params_from_file(params_filename)
        assert len(all_params) == 1, 'Only one set of parameters is supported'
        exp_params = all_params[0]
        parameter_setup.sanitize_params(exp_params)
    else:
        exp_params = code_params

    common.init_logging(exp_params)

    print('Experiment parameters')
    print(pformat(exp_params))

    print('Loading model: ', exp_params['model_state_path'])

    network, _ = model_utility.load_model(exp_params, set_weights=True)

    state_dict = network.state_dict()
    checksum = hashlib.md5(str(state_dict).encode()).hexdigest()
    print(f'Model neural network checksum: {checksum}')

    network.eval()  # switch to evaluate mode (!! very important !!)


    # ----------------- Generate attack examples -----------------
    # Uncomment & run this block to generate attack examples & save to disk

    print('Preparing dataset: ', exp_params['dataset'])

    trainset, testset = data_prep.load_data(exp_params)

    XX, labels, XX_attacks = generate_ig_attacks(network, testset, exp_params)

    save_dataset_to_disk(XX, labels, XX_attacks, exp_params['output_dir'])


    # # ----------------- Evaluate attack examples -----------------
    # # Uncomment & run this block to load saved attack examples & evaluate
    #
    # XX, labels, XX_attacks = load_dataset_from_disk(exp_params)
    #
    # visualize_ig_maps(network, XX, labels, XX_attacks, exp_params)
    #
    # evaluate_accuracy(network, XX, labels, XX_attacks, exp_params)
    #
    # evaluate_attacks_ig_maps(network, XX, labels, XX_attacks, exp_params)


if __name__ == '__main__':
    main()

