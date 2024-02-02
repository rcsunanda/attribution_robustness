# Run with the following command
# python -m int_tl.feature_significance_tests

import warnings
import time
from pprint import pformat

warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning msgs from tensorflow imports

from tensorboardX import SummaryWriter

import os
import torch
import numpy as np
from captum.attr import Saliency, IntegratedGradients
# from path_explain import PathExplainerTorch
# from path_explain import PathExplainerTF
from attributionpriors.attributionpriors.pytorch_ops_cpu import \
    AttributionPriorExplainer as AttributionPriorExplainerCPU
from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer as AttributionPriorExplainerGPU

from utility import parameter_setup, data_prep, common, misc, attribution_robustness_metrics
from models import model_utility
from int_tl import feature_significance, ig_attack, gradient_attack, ifia_attack
from utility import visualization

params = {
    # # CIFAR-10, ResNet-18
    # 'dataset': 'cifar10_augmented',
    # 'data_dir': 'data',
    # 'num_classes': 10,
    # # 'model_state_path': 'saved_models/cifar10/z1_resnet_cifar10_from_scratch/final_model_state.pt',
    # 'model_state_path': 'saved_models/cifar10/z1_resnet_cifar10_pretrained/final_model_state.pt',

    # # CIFAR-10, Small CNN
    # 'dataset': 'cifar10_augmented',
    # 'data_dir': 'data',
    # 'num_classes': 10,
    # 'model_state_path': 'saved_models/cifar10/small_cnn_cifar10-new/final_model_state.pt',

    # ImageNet, ResNet-18
    'dataset': 'restricted_imagenet_balanced',
    'data_dir': 'data/imagenet/ILSVRC/Data/CLS-LOC',
    'num_classes': 14,
    # 'model_state_path': 'saved_models/rimb_st/resnet_rimb_st_size128/final_model_state.pt',
    # 'model_state_path': 'saved_models/rimb_st_per-image-normalized/resnet_rimb_st_size128_perimgnorm/final_model_state.pt',
    'model_state_path': 'saved_models/rimb_st_per-image-normalized/resnet_rimb_st_size128_dataloader-perimgnorm/final_model_state.pt',

    # # MNIST, Small CNN
    # 'dataset': 'mnist',
    # 'data_dir': 'data',
    # 'num_classes': 10,
    # 'model_state_path': 'saved_models/cnn_mnist_st_unnormalized_e200/final_model_state.pt',

    # 'dataset_percent': 0.05,
    'dataset_percent': 0.5,
    'batch_size': 64,
    'test_batch_size': 32,
    'transfer_type': 'from_scratch',

    # Model & training

    # ResNet18 for CIFAR-10 & ImageNet
    'model': 'resnet18',
    'pretrained': 0,
    'kaiming_init': 0,
    'freeze_conv_base': 0,
    'distributed_training': 0,
    'image_size': 128,
    'input_shape': [3, 128, 128],   # For ImageNet
    # 'input_shape': [3, 32, 32],     # For CIFAR-10
    'classifier': [],
    'device': 'cuda:0',  # eg: cpu, cuda:0, cuda:1
    # 'device': 'cpu',  # eg: cpu, cuda:0, cuda:1

    # # Small CNN for CIFAR-10
    # 'model': 'small_cnn',
    # 'pretrained': 0,
    # 'kaiming_init': 0,
    # 'freeze_conv_base': 0,
    # 'distributed_training': 0,
    # 'input_shape': [3, 32, 32],
    # 'conv_base': [{"type": "Conv2d", "in_channels": 3, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 1}, {"type": "MaxPool2d", "kernel_size": 2, "stride": 2}, {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 1}, {"type": "MaxPool2d", "kernel_size": 2, "stride": 2}],
    # 'classifier': [{"type": "ReLU", "out_features": 1024}],
    # # 'device': 'cuda:0',   # eg: cpu, cuda:0, cuda:1
    # 'device': 'cpu',   # eg: cpu, cuda:0, cuda:1

    # # Small CNN for MNIST
    # 'model': 'small_cnn',
    # 'pretrained': 0,
    # 'kaiming_init': 0,
    # 'freeze_conv_base': 0,
    # 'distributed_training': 0,
    # 'input_shape': [1, 28, 28],
    # 'conv_base': [{"type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 5, "stride": 1, "padding": 1}, {"type": "MaxPool2d", "kernel_size": 2, "stride": 2}, {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 1}, {"type": "MaxPool2d", "kernel_size": 2, "stride": 2}],
    # 'classifier': [{"type": "ReLU", "out_features": 1024}],
    # # 'device': 'cuda:0',   # eg: cpu, cuda:0, cuda:1
    # 'device': 'cpu',   # eg: cpu, cuda:0, cuda:1
}

tb_writer = None    # Initialize in main

# n first examples with correct predictions
def get_correct_pred_examples(data_loader, network, num_samples):
    X = []
    y = []

    for inputs, labels in data_loader:
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])

        with torch.no_grad():  # We do not intend to call backward(), as we don't backprop and optimize
            outputs = network(inputs)
            _, y_pred = torch.max(outputs.data, 1)

            idx = y_pred == labels  # examples with correct predictions

            X.extend([x for x in inputs[idx]])
            y.extend([y for y in labels[idx]])

            if len(X) >= num_samples:
                break

    X = torch.stack(X[:num_samples])
    y = torch.stack(y[:num_samples])

    return X, y


def compare_gradients_single_vs_tensor_1(network, testset):
    print('Comparing gradients of a single example impl vs. tensor gradient impl')

    X, y = get_correct_pred_examples(testset, network, num_samples=1)

    X_grad_single = feature_significance._compute_gradients_single_input(network, X, y)

    X_grad = feature_significance.compute_gradients(network, X, y)

    # print(f'different_elements_count = {X_grad_single.numel() - torch.isclose(X_grad_single, X_grad, atol=1e-06).sum()}')

    assert torch.allclose(X_grad_single, X_grad, atol=1e-06)

    print('Test passed')


def compare_gradients_single_vs_tensor_2(network, testset):
    print('Comparing gradients of tensor gradients vs. gradient of each example separately')

    XX, yy = get_correct_pred_examples(testset, network, num_samples=256)

    XX_grad_all = feature_significance.compute_gradients(network, XX, yy)

    # Call multiple times
    XX_grad_all_2 = feature_significance.compute_gradients(network, XX, yy)
    assert torch.allclose(XX_grad_all, XX_grad_all_2, atol=1e-05)

    XX = XX.detach().clone()
    yy = yy.detach().clone()

    for i in range(len(XX)):
        # print(f'i = {i}')
        X = XX[i].reshape(1, *XX[i].shape)
        y = yy[i].reshape(1)

        X_grad = feature_significance.compute_gradients(network, X, y)

        XX_grad_i = XX_grad_all[i].reshape(1, *XX_grad_all[i].shape)

        # print(X_grad.shape)
        # print(XX_grad_i.shape)
        # print(X_grad)
        # print(XX_grad_i)

        diff_elements_count = X_grad.numel() - torch.isclose(X_grad, XX_grad_i, atol=1e-03, rtol=1e-02).sum()
        # print(f'different_elements_count = {diff_elements_count}')

        assert diff_elements_count / X_grad.numel() <= 0.08  # Less than 8% emelements are different

        # assert torch.allclose(X_grad, XX_grad_i, atol=1e-04, rtol=1e-03)

    print('Test passed')


def compare_gradients_custom_vs_captum(network, testset):
    print('Comparing gradients of my tensor gradient impl vs. captum library gradient')

    saliency = Saliency(network)

    XX, yy = get_correct_pred_examples(testset, network, num_samples=15)
    # print(yy)

    XX_grad = feature_significance.compute_gradients(network, XX, yy)

    XX_grad_captum = saliency.attribute(XX, yy.detach(), abs=False)

    assert torch.allclose(XX_grad, XX_grad_captum, atol=1e-04)

    print('Test passed')


def compare_integrated_gradients_custom_vs_captum(network, testset):
    print('Comparing integrated gradients of my tensor gradient impl vs. captum library gradient')

    ig = IntegratedGradients(network, multiply_by_inputs=True)

    # Large no. of examples and large steps will cause GPU out-of-memory error - ToDo: Implement batched IG
    XX, yy = get_correct_pred_examples(testset, network, num_samples=32)
    # print(yy)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 10

    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_ig_captum = ig.attribute(XX, target=yy, baselines=baselines, n_steps=steps, method='riemann_left')
    XX_ig_captum = XX_ig_captum.float()

    # print(XX_ig)
    # print(XX_ig_captum)

    # Debugging
    # additional_different_elements_count = additional.numel() - torch.isclose(additional, additional_captum, atol=1e-03, rtol=1e-02).sum()
    # print(f'additional_different_elements_count = {additional_different_elements_count}')
    # assert torch.allclose(additional, additional_captum, atol=1e-02)

    diff_elements_count = XX_ig.numel() - torch.isclose(XX_ig, XX_ig_captum, atol=1e-03, rtol=1e-02).sum()
    print(f'different_elements_count = {diff_elements_count}')

    assert torch.allclose(XX_ig, XX_ig_captum, atol=1e-02)

    print('Test passed')


def compare_visualization_ig_custom_vs_captum(network, testset):
    print('Comparing IG visualization of my tensor gradient impl vs. captum library gradient')

    ig = IntegratedGradients(network, multiply_by_inputs=True)

    # Large no. of examples (eg: 16) and large steps (eg: 24) will cause GPU out-of-memory error - ToDo: Implement batched IG
    XX, yy = get_correct_pred_examples(testset, network, num_samples=8)
    # print(yy)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_ig_of_loss = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines, m=steps)

    XX_ig_captum = ig.attribute(XX, target=yy, baselines=baselines, n_steps=steps, method='riemann_left',
                                return_convergence_delta=False)
    XX_ig_captum = XX_ig_captum.float()

    print(f'XX_ig.shape: {XX_ig.shape}')
    print(f'XX_ig_captum.shape: {XX_ig_captum.shape}')

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_ig_captum = torch.sum(torch.abs(XX_ig_captum), dim=1)  # Channels first
    XX_ig_of_loss = torch.sum(torch.abs(XX_ig_of_loss), dim=1)  # Channels first

    # --------- Visualize images and IG maps in same plot grid (2 plot grids for Custom IG and Captum IG) -----------

    plot1_args = []
    plot2_args = []
    plot3_args = []
    for x, x1, x2, x3 in zip(XX, XX_ig, XX_ig_captum, XX_ig_of_loss):
        plot1_args.append({'x': x.cpu()})
        plot1_args.append({'x': x1.cpu()})

        plot2_args.append({'x': x.cpu()})
        plot2_args.append({'x': x2.cpu()})

        plot3_args.append({'x': x.cpu()})
        plot3_args.append({'x': x3.cpu()})

    visualization.plot_grid(4, 4, 'Custom_IG_maps', visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 4, 'Captum_Lib_IG_maps', visualization.plot_feature_sig, plot2_args)

    visualization.plot_grid(4, 4, 'batch_loss_IG_maps', visualization.plot_feature_sig, plot3_args)

    common.save_all_figures('output/tests')


def visualize_topk_ifia_attacks(network, testset):
    print('visualize_topk_ifia_attacks')

    # # ImageNet parameters - Unnormalized
    # epsilon = 8
    # alpha = 2
    # num_iter = 100
    # adv_ig_steps = 100
    # num_samples = 4
    # k_top = 1000

    # # ImageNet parameters. Normalized - This set works when we modify the IFIA attack to ignore prediction matching
    # epsilon = 0.3
    # alpha = 0.01
    # num_iter = 20
    # adv_ig_steps = 20
    # num_samples = 4  # Over 4 causes GPU out-of-memory errors on the 2080 RTX
    # k_top = 1000

    # ImageNet parameters. Normalized - This set works with the IFIA attack that matches predictions
    epsilon = 0.6
    alpha = 0.01
    num_iter = 50
    adv_ig_steps = 40
    num_samples = 4  # Over 4 causes GPU out-of-memory errors on the 2080 RTX
    k_top = 1000

    # # MNIST parameters
    # epsilon = 0.2
    # alpha = 0.01
    # num_iter = 2
    # adv_ig_steps = 20
    # num_samples = 48
    # k_top = 200

    random_start = False

    XX, yy = get_correct_pred_examples(testset, network, num_samples=num_samples)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    params_str = f'epsilon_{epsilon}--alpha_{alpha}--num_iter_{num_iter}--adv_ig_steps_{adv_ig_steps}--k_top_{k_top}'

    print(f'Computing IFIA attacks with params {params_str}')

    XX_attacks = []
    for x, y in zip(XX, yy):
        x_attack = ifia_attack.create_topk_ifia_attack(network, x, y, epsilon, alpha, num_iter, adv_ig_steps, k_top, random_start)
        # print(f'x_attack.shape: {x_attack.shape}')
        XX_attacks.append(x_attack)

        # Compare x and x_attack
        # diff_elements_count = x.numel() - torch.isclose(x, x_attack, atol=1e-03, rtol=1e-02).sum()
        # print(f'different_elements_count = {diff_elements_count}')
        # print(x)
        # print(x_attack)
        # print ('--------------------------------')

    XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples
    print(f'XX_attacks.shape: {XX_attacks.shape}')

    XX_attacks_ig = feature_significance.compute_integrated_gradients(network, XX_attacks, yy, baselines, m=steps)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Channels first

    # XX_attacks = torch.zeros(XX.shape).to(XX.device)
    # XX_attacks_ig = torch.zeros(XX.shape).to(XX.device)

    print(f'num_iter: {num_iter}. XX_attacks checksum: {misc.tensor_checksum(XX_attacks)}')
    print(f'num_iter: {num_iter}. XX_attacks_ig checksum: {misc.tensor_checksum(XX_attacks_ig)}')

    plot1_args = []
    for x, x1, x2, x3 in zip(XX, XX_ig, XX_attacks, XX_attacks_ig):
        plot1_args.append({'x': x.cpu()})
        plot1_args.append({'x': x1.cpu()})
        plot1_args.append({'x': x2.cpu()})
        plot1_args.append({'x': x3.cpu()})

    fig = visualization.plot_grid(4, 4, f'IFIA_attacks--{params_str}', visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')

    global tb_writer
    tb_writer.add_figure('IFIA_attacks', fig, 0)
    print('Figure saved to Tensorboard (Images tab)')


def visualize_batch_loss_ig_attacks(network, testset):
    print('visualize_batch_loss_ig_attacks')

    # # # ImageNet parameters - set 1
    # epsilon = 8
    # alpha = 2
    # num_iter = 16
    # adv_ig_steps = 20
    # batch_size = 16  # Over 4 causes GPU out-of-memory errors on the 2080 RTX

    # ImageNet parameters - set 2
    epsilon = 0.1
    alpha = 0.008
    num_iter = 20
    adv_ig_steps = 20
    batch_size = 16  # Over 4 causes GPU out-of-memory errors on the 2080 RTX

    # # CIFAR-10 parameters
    # epsilon = 0.6
    # alpha = 0.01
    # num_iter = 16
    # adv_ig_steps = 50
    # batch_size = 4

    # #MNIST parameters
    # epsilon = 0.3
    # alpha = 0.01
    # num_iter = 2
    # adv_ig_steps = 20
    # batch_size = 48

    params_str = f'epsilon_{epsilon}--alpha_{alpha}--num_iter_{num_iter}--adv_ig_steps_{adv_ig_steps}--batch_size_{batch_size}'

    print(f'Computing IG batch_loss attacks with params. {params_str}')

    XX, yy = get_correct_pred_examples(testset, network, num_samples=batch_size)

    XX_attacks = ig_attack.compute_pgd_attacks(network, XX, yy, epsilon, alpha, num_iter, adv_ig_steps)

    print(f'XX_attacks.shape: {XX_attacks.shape}')

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 4
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)
    # XX_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines, m=steps)
    # XX_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines, m=steps)

    XX_attacks_ig = feature_significance.compute_integrated_gradients(network, XX_attacks, yy, baselines, m=steps)
    # XX_attacks_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX_attacks, yy, baselines, m=steps)
    # XX_attacks_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX_attacks, yy, baselines, m=steps)

    total_elements = XX_attacks_ig.numel()
    nan_count = torch.sum(torch.isnan(XX_attacks_ig)).item()
    inf_count = torch.sum(torch.isinf(XX_attacks_ig)).item()

    # Count masked elements in the PyTorch array
    # masked_count = torch.sum(torch.eq(XX_attacks_ig, -1))

    print('nan_count: ', nan_count, ', inf_count: ', inf_count, ', total_elements: ', total_elements)
    # print(XX_attacks_ig)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Channels first

    print('all_sum: ', torch.sum(XX_attacks_ig))

    print(f'num_iter: {num_iter}. XX_attacks checksum: {misc.tensor_checksum(XX_attacks)}')
    print(f'num_iter: {num_iter}. XX_attacks_ig checksum: {misc.tensor_checksum(XX_attacks_ig)}')

    # num_samples = min(8, len(XX))     # First n samples to visualize
    start_idx = 5     # Start at this index
    num_samples = 4     # n samples to visualize

    plot1_args = []
    for i in range(start_idx, start_idx + num_samples):
        plot1_args.append({'x': XX[i].cpu()})
        plot1_args.append({'x': XX_ig[i].cpu()})
        plot1_args.append({'x': XX_attacks[i].cpu()})
        plot1_args.append({'x': XX_attacks_ig[i].cpu()})

    fig = visualization.plot_grid(num_samples, 4, f'batch_loss_ig_attacks--{params_str}',
                                  visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')

    global tb_writer
    tb_writer.add_figure('batch_loss_ig_attacks', fig, 0)
    print('Figure saved to Tensorboard (Images tab)')


def compare_batch_loss_ig_loop_vs_vectorized(network, testset):
    print('compare_batch_loss_ig_loop_vs_vectorized')

    batch_size = 12

    XX, yy = get_correct_pred_examples(testset, network, num_samples=batch_size)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 4

    t0 = time.time()
    XX_ig_loop = feature_significance.compute_integrated_gradients_of_loss_refactored(network, XX, yy, baselines, m=steps, create_graph=True)
    print(f'compute_integrated_gradients_of_loss_refactored: time taken: {time.time() - t0:.1f} secs')

    t1 = time.time()
    XX_ig_vectorized = feature_significance.compute_integrated_gradients_of_loss_vectorized(network, XX, yy, baselines, m=steps, create_graph=True)
    print(f'compute_integrated_gradients_of_loss_vectorized: time taken: {time.time() - t1:.1f} secs')

    # To see if one method gives an output that is an exact multiple of the other
    # division = XX_ig_loop / XX_ig_vectorized
    # print(division)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig_loop = torch.sum(torch.abs(XX_ig_loop), dim=1)  # Channels first
    XX_ig_vectorized = torch.sum(torch.abs(XX_ig_vectorized), dim=1)  # Channels first

    num_samples = 4     # First n samples to visualize

    plot1_args = []
    for i in range(num_samples):
        plot1_args.append({'x': XX[i].cpu(), 'plot_func': visualization.plot_grayscale})
        plot1_args.append({'x': XX_ig_loop[i].cpu()})
        plot1_args.append({'x': XX_ig_vectorized[i].cpu()})

    visualization.plot_grid(num_samples, 3, 'IG_maps (batch loss: loop vs vectorized)', visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')

    assert torch.allclose(XX_ig_loop, XX_ig_vectorized, atol=1e-04)     # atol=1e-04 may be too strict


def visualize_batch_loss_ig_attacks_iterations(network, testset):
    print('visualize_batch_loss_ig_attacks')
    global tb_writer

    # # ImageNet parameters
    # epsilon = 8
    # alpha = 2
    # num_iter = 10
    # adv_ig_steps = 8
    # batch_size = 48

    # MNIST parameters
    epsilon = 0.3
    alpha = 0.01
    # num_iters_list = [0, 1, 5, 10, 20, 50]
    # num_iters_list = [0, 1, 5, 10, 20, 30, 40, 50, 70, 80, 90, 100]
    num_iters_list = [0, 1, 5]
    # num_iters_list = [200]
    adv_ig_steps = 20
    batch_size = 48

    XX, yy = get_correct_pred_examples(testset, network, num_samples=batch_size)

    images = {'natural': XX}  # key: 'natural' or 'attack_iters_n', value: images (natural or attacks)
    heatmaps = {}  # key, val same as above

    # Compute the attacks for given list of iterations
    for iters in num_iters_list:
        print('--------> Computing attacks for iters: ', iters)
        XX_attacks = ig_attack.compute_pgd_attacks(network, XX, yy, epsilon, alpha, iters, adv_ig_steps, random_start=True)
        images[f'attack_iters_{iters}'] = XX_attacks

    # Compute the heatmaps for all images (natural and attacks)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    for i, (key, img_batch) in enumerate(images.items()):
        img_batch_ig = feature_significance.compute_integrated_gradients_of_loss_refactored(network, img_batch, yy, baselines, m=steps)

        # Need to sum channels to get per-pixel attributions that can be visualized with a color map
        img_batch_ig = torch.sum(torch.abs(img_batch_ig), dim=1)  # Channels first

        heatmaps[key] = img_batch_ig

        print(f'\tkey: {key}. img_batch: {misc.tensor_checksum(img_batch)}, img_batch_sum: {torch.sum(img_batch)}')

        tb_writer.add_text('img_batch_checksums', f'{misc.tensor_checksum(img_batch)}', i)

        # print(f'key: {key}. img_batch_ig: {misc.tensor_checksum(img_batch_ig)}')

    assert len(images) == len(heatmaps)

    num_samples = 4     # First n samples to visualize

    plot1_args = []
    for i in range(num_samples):
        keys_1 = []
        for key1, image_batch in images.items():
            plot1_args.append({'x': image_batch[i].cpu(), 'plot_func': visualization.plot_grayscale})
            keys_1.append(key1)

        keys_2 = []
        for key2, heatmap_batch in heatmaps.items():
            plot1_args.append({'x': heatmap_batch[i].cpu()})
            keys_2.append(key2)

        # Verify correct alignment of visualizations
        for key1, key2 in zip(keys_1, keys_2):
            assert key1 == key2

    rows = 2 * num_samples
    cols = len(num_iters_list) + 1   # +1 for original image
    assert rows * cols == len(plot1_args)

    visualization.plot_grid(rows, cols, 'IG_attack_iterations_maps', visualization.plot_feature_sig, plot1_args)

    common.save_all_figures('output/tests')

    print('Reporting metrics to TensorBoard')
    eval_top_K = 100

    natural_XX_ig = heatmaps['natural']

    for iter in num_iters_list:
        XX_attacks_ig = heatmaps[f'attack_iters_{iter}']
        XX_attacks = images[f'attack_iters_{iter}']

        # Compute metrics
        ig_norm_sum = ig_attack.compute_ig_norm_sum(network, XX, yy, XX_attacks, steps)
        tb_writer.add_scalar('ig_x_attack', ig_norm_sum, iter)

        spearman = attribution_robustness_metrics.compute_spearman_rank_correlation(natural_XX_ig, XX_attacks_ig)
        kendall = attribution_robustness_metrics.compute_kendalls_correlation(natural_XX_ig, XX_attacks_ig)
        inersection = attribution_robustness_metrics.compute_topK_intersection(natural_XX_ig, XX_attacks_ig, K=eval_top_K)

        tb_writer.add_scalar('spearman', spearman, iter)
        tb_writer.add_scalar('kendall', kendall, iter)
        tb_writer.add_scalar('intersection', inersection, iter)


def visualize_gradient_attacks(network, testset):
    print('visualize_gradient_attacks')

    epsilon = 8
    alpha = 2
    num_iter = 5

    XX, yy = get_correct_pred_examples(testset, network, num_samples=4)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_grad = feature_significance.compute_gradients(network, XX, yy)

    XX_attacks = []
    for x, y in zip(XX, yy):
        x_attack = gradient_attack.create_gradient_attack_example(network, x, y, epsilon, alpha, num_iter)
        # print(f'x_attack.shape: {x_attack.shape}')
        XX_attacks.append(x_attack)

        # Compare x and x_attack
        # diff_elements_count = x.numel() - torch.isclose(x, x_attack, atol=1e-03, rtol=1e-02).sum()
        # print(f'different_elements_count = {diff_elements_count}')
        # print(x)
        # print(x_attack)
        # print ('--------------------------------')

    XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples
    print(f'XX_attacks.shape: {XX_attacks.shape}')

    XX_attacks_ig = feature_significance.compute_integrated_gradients(network, XX_attacks, yy, baselines, m=steps)

    XX_attacks_grad = feature_significance.compute_gradients(network, XX_attacks, yy)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Channels first

    # XX_attacks = torch.zeros(XX.shape).to(XX.device)
    # XX_attacks_ig = torch.zeros(XX.shape).to(XX.device)

    plot1_args = []
    plot2_args = []

    for x, x1, x2, x3 in zip(XX, XX_grad, XX_attacks, XX_attacks_grad):
        plot1_args.append({'x': x.cpu()})
        plot1_args.append({'x': x1.cpu()})
        plot1_args.append({'x': x2.cpu()})
        plot1_args.append({'x': x3.cpu()})

    for x, x1, x2, x3 in zip(XX, XX_ig, XX_attacks, XX_attacks_ig):
        plot2_args.append({'x': x.cpu()})
        plot2_args.append({'x': x1.cpu()})
        plot2_args.append({'x': x2.cpu()})
        plot2_args.append({'x': x3.cpu()})

    visualization.plot_grid(4, 4, 'Gradient_attack_grad_maps', visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 4, 'Gradient_attack_IG_maps', visualization.plot_feature_sig, plot2_args)

    common.save_all_figures('output/tests')


def visualize_expected_gradients_path_explainer(network, testset):
    explainer = PathExplainerTorch(network)

    num_refs = 16  # This is the k value    # k <= batch_size is good enough

    XX, yy = get_correct_pred_examples(testset, network, num_samples=16)

    # baseline_XX, yy = get_correct_pred_examples(testset, network, num_samples=100)

    print(f'Running path_explainer EG attributions. num_refs: {num_refs}')

    # Baseloine is the same as input batch, as reference examples can be chosen from the input batch
    # (see Expected Gradients paper)
    XX_eg = explainer.attributions(input_tensor=XX,
                                   baseline=XX,
                                   num_samples=num_refs,
                                   use_expectation=True,
                                   output_indices=yy)

    print(f'XX_eg.shape: {XX_eg.shape}')

    print('Computing IG maps (custom)')

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)
    # XX_ig = XX

    plot1_args = []
    plot2_args = []
    for x, x1, x2 in zip(XX, XX_eg, XX_ig):
        plot1_args.append({'x': x.cpu()})
        plot1_args.append({'x': x1.cpu()})

        plot2_args.append({'x': x.cpu()})
        plot2_args.append({'x': x2.cpu()})

    visualization.plot_grid(4, 8, 'Expected_Gradient_maps', visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 8, 'Custom_IG_maps', visualization.plot_feature_sig, plot2_args)

    common.save_all_figures('output/tests')


def visualize_expected_gradients(network, testset):
    num_refs = 4  # This is the k value

    XX, yy = get_correct_pred_examples(testset, network, num_samples=64)

    print('Running EG attributions')

    # explainer = AttributionPriorExplainerCPU(XX, 5, k=num_refs)

    explainer = AttributionPriorExplainerGPU(XX, 5, k=num_refs)

    attributions = explainer.shap_values(network, input_tensor=XX)

    print(f'attributions.shape: {attributions.shape}')


def compare_integrated_of_loss_original_vs_refactored(network, testset):
    print('compare_integrated_of_loss_original_vs_refactored')

    n = 0
    for inputs, labels in testset:
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])

        baseline = torch.zeros((1, *inputs[0].shape)).to(inputs.device)
        steps = 24

        X_ig_orig = feature_significance.compute_integrated_gradients_of_loss_original(network, inputs, labels, baseline, steps)

        X_ig_refactored = feature_significance.compute_integrated_gradients_of_loss_refactored(network, inputs, labels, baseline, steps)

        assert torch.allclose(X_ig_orig, X_ig_refactored, atol=1e-04)     # atol=1e-04 may be too strict

        n = n +1

    print(f'Test passed after comparison of {n} batches')


def compare_gradients_of_loss_original_vs_refactored(network, testset):
    print('compare_gradients_of_loss_original_vs_refactored')

    n = 0
    for inputs, labels in testset:
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])

        X_ig_orig = feature_significance.compute_gradients_of_loss_original(network, inputs, labels)

        X_ig_refactored = feature_significance.compute_gradients_of_loss_refactored(network, inputs, labels)

        assert torch.allclose(X_ig_orig, X_ig_refactored, atol=1e-04)  # atol=1e-04 may be too strict

        n = n + 1

    print(f'Test passed after comparison of {n} batches')


def main():
    # -----------------------------------------------------
    # Setup

    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(os.getcwd())

    # parameter_setup.sanitize_params(params)  # Set default params

    print('Loading model')

    network, _ = model_utility.load_model(params, set_weights=True)
    network.eval()  # switch to evaluate mode (!! very important !!)

    print('Preparing data')

    trainset, testset = data_prep.load_data(params)

    print('Initializing tensorboard')
    global tb_writer
    tb_writer = SummaryWriter('output/tests/tensorboard', comment='BBBBBBBB', flush_secs=3)

    tb_writer.add_text('Experiment params', pformat(params).replace('\n', '<br>'), -1)

    # -----------------------------------------------------
    # Run tests

    print('Running tests')

    # compare_gradients_single_vs_tensor_1(network, testset)
    # compare_gradients_single_vs_tensor_2(network, testset)

    # compare_gradients_custom_vs_captum(network, testset)

    # compare_integrated_gradients_custom_vs_captum(network, testset)

    # compare_visualization_ig_custom_vs_captum(network, testset)

    # visualize_topk_ifia_attacks(network, testset)

    # visualize_batch_loss_ig_attacks(network, testset)

    # visualize_batch_loss_ig_attacks_iterations(network, testset)

    # visualize_gradient_attacks(network, testset)

    # visualize_expected_gradients_path_explainer(network, testset)

    # visualize_expected_gradients(network, testset)

    # compare_batch_loss_ig_loop_vs_vectorized(network, testset)

    # compare_integrated_of_loss_original_vs_refactored(network, testset)

    compare_gradients_of_loss_original_vs_refactored(network, testset)

    tb_writer.close()


if __name__ == '__main__':
    main()
