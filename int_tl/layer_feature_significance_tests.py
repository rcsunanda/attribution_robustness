# Run with the following command
# python -m int_tl.feature_significance_tests

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning msgs from tensorflow imports

import os
import time
import torch
import numpy as np
from captum.attr import Saliency, IntegratedGradients
from path_explain import PathExplainerTorch
# from path_explain import PathExplainerTF
# from attributionpriors.attributionpriors.pytorch_ops_cpu import AttributionPriorExplainer as AttributionPriorExplainerCPU
# from attributionpriors.attributionpriors.pytorch_ops import AttributionPriorExplainer as AttributionPriorExplainerGPU

from utility import parameter_setup, data_prep, common
from models import model_utility
from int_tl import layer_feature_significance, layer_ig_attack, layer_gradient_attack
from utility import visualization
from int_tl import feature_significance     # Need plotting functions


params = {
    # Dataset
    # 'dataset': 'cifar10_augmented',
    # 'data_dir': 'data',
    # 'num_classes': 10,
    # 'model_state_path': 'saved_models/z1_resnet_cifar10_kaiming0_Jun22-12_12_17/best_model_state.pt',

    'dataset': 'restricted_imagenet_balanced',
    'data_dir': '/home/sunanda/research/Datasets/imagenet/ILSVRC/Data/CLS-LOC',
    'num_classes': 14,
    'model_state_path': 'saved_models/b1_resnet_rimb_st_Feb14-15_05_30/best_model_state.pt',

    'dataset_percent': 0.1,
    'batch_size': 64,
    'test_batch_size': 64,
    'transfer_type': 'from_scratch',

    # Model & training
    'model': 'resnet18',
    'pretrained': 0,
    'kaiming_init': 0,
    'freeze_conv_base': 0,
    'distributed_training': 0,
    'classifier': [],
    # 'device': 'cuda:0',   # eg: cpu, cuda:0, cuda:1
    'device': 'cpu',   # eg: cpu, cuda:0, cuda:1
}


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

    X_grad = feature_significance.compute_layer_gradients(network, X, y)

    # print(f'different_elements_count = {X_grad_single.numel() - torch.isclose(X_grad_single, X_grad, atol=1e-06).sum()}')

    assert torch.allclose(X_grad_single, X_grad, atol=1e-06)

    print('Test passed')


def compare_gradients_single_vs_tensor_2(network, testset):
    print('Comparing gradients of tensor gradients vs. gradient of each example separately')

    XX, yy = get_correct_pred_examples(testset, network, num_samples=256)

    XX_grad_all = feature_significance.compute_layer_gradients(network, XX, yy)

    # Call multiple times
    XX_grad_all_2 = feature_significance.compute_layer_gradients(network, XX, yy)
    assert torch.allclose(XX_grad_all, XX_grad_all_2, atol=1e-05)

    XX = XX.detach().clone()
    yy = yy.detach().clone()

    for i in range(len(XX)):
        # print(f'i = {i}')
        X = XX[i].reshape(1, *XX[i].shape)
        y = yy[i].reshape(1)

        X_grad = feature_significance.compute_layer_gradients(network, X, y)

        XX_grad_i = XX_grad_all[i].reshape(1, *XX_grad_all[i].shape)

        # print(X_grad.shape)
        # print(XX_grad_i.shape)
        # print(X_grad)
        # print(XX_grad_i)

        diff_elements_count = X_grad.numel() - torch.isclose(X_grad, XX_grad_i, atol=1e-03, rtol=1e-02).sum()
        # print(f'different_elements_count = {diff_elements_count}')

        assert diff_elements_count / X_grad.numel() <= 0.08     # Less than 8% emelements are different

        # assert torch.allclose(X_grad, XX_grad_i, atol=1e-04, rtol=1e-03)

    print('Test passed')


def compare_layer_gradients_custom_vs_captum(network, testset):
    print('Comparing gradients of my tensor gradient impl vs. captum library gradient')

    block_idx = 4
    sub_block_idx = 1

    saliency = Saliency(network)

    XX, yy = get_correct_pred_examples(testset, network, num_samples=15)
    # print(yy)

    XX_grad = layer_feature_significance.compute_layer_gradients(network, XX, yy, block_idx, sub_block_idx)

    XX_hook_grad = layer_feature_significance.compute_layer_gradients_with_hook(network, XX, yy, block_idx, sub_block_idx)

    assert torch.allclose(XX_grad, XX_hook_grad, atol=1e-04)


    # XX_grad_captum = saliency.attribute(XX, yy.detach(), abs=False)

    # assert torch.allclose(XX_grad, XX_grad_captum, atol=1e-04)

    print('Test passed')


def gradients_speed_test(network, testset):
    print('Gradient computation speed test')
    XX, yy = get_correct_pred_examples(testset, network, num_samples=15)

    block_idx = 6
    sub_block_idx = 1

    n = 1000

    t0 = time.time()
    for i in range(n):
        XX_grad = layer_feature_significance.compute_layer_gradients(network, XX, yy, block_idx, sub_block_idx)

    print(f'Compute gradients with autograd. Time taken: {time.time() - t0: .2f} secs')

    t0 = time.time()
    for i in range(n):
        XX_hook_grad = layer_feature_significance.compute_layer_gradients_with_hook(network, XX, yy, block_idx, sub_block_idx)

    print(f'Compute gradients with hooks. Time taken: {time.time() - t0: .2f} secs')


def compare_integrated_gradients_custom_vs_captum(network, testset):
    print('Comparing integrated gradients of my tensor gradient impl vs. captum library gradient')

    ig = IntegratedGradients(network, multiply_by_inputs=True)

    # Large no. of examples and large steps will cause GPU out-of-memory error - ToDo: Implement batched IG
    XX, yy = get_correct_pred_examples(testset, network, num_samples=32)
    # print(yy)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 10

    block_idx = 4
    sub_block_idx = 1

    XX_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, baselines,
                                                                          block_idx, sub_block_idx, m=steps)

    XX_ig_captum = ig.attribute(XX, target=yy, baselines=baselines, n_steps=steps, method='riemann_left')
    XX_ig_captum = XX_ig_captum.float()

    print(f'XX_ig.shape: {XX_ig.shape}')
    print(f'XX_ig_captum.shape: {XX_ig_captum.shape}')

    # Debugging
    # additional_different_elements_count = additional.numel() - torch.isclose(additional, additional_captum, atol=1e-03, rtol=1e-02).sum()
    # print(f'additional_different_elements_count = {additional_different_elements_count}')
    # assert torch.allclose(additional, additional_captum, atol=1e-02)

    # diff_elements_count = XX_ig.numel() - torch.isclose(XX_ig, XX_ig_captum, atol=1e-03, rtol=1e-02).sum()
    # print(f'different_elements_count = {diff_elements_count}')
    #
    # assert torch.allclose(XX_ig, XX_ig_captum, atol=1e-02)

    print('Test passed')


def compare_visualization_ig_custom_vs_captum(network, testset):
    print('Comparing IG visualization of my tensor gradient impl vs. captum library gradient')

    ig = IntegratedGradients(network, multiply_by_inputs=True)

    # Large no. of examples and large steps will cause GPU out-of-memory error - ToDo: Implement batched IG
    XX, yy = get_correct_pred_examples(testset, network, num_samples=16)
    # print(yy)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    XX_ig = feature_significance.compute_layer_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_ig_captum = ig.attribute(XX, target=yy, baselines=baselines, n_steps=steps, method='riemann_left', return_convergence_delta=False)
    XX_ig_captum = XX_ig_captum.float()

    print(f'XX_ig.shape: {XX_ig.shape}')
    print(f'XX_ig_captum.shape: {XX_ig_captum.shape}')

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)     # Channels first
    XX_ig_captum = torch.sum(torch.abs(XX_ig_captum), dim=1)     # Channels first

    # # ---------------- Visualize in 3 separate plot grids ----------------
    #
    # plot_args = [{'x': x.cpu()} for x in XX]
    # visualization.plot_grid(4, 4, 'Original_images', feature_significance.plot_feature_sig, plot_args)
    #
    # plot_args = [{'x': m.cpu()} for m in XX_ig]
    # visualization.plot_grid(4, 4, 'Custom_IG_maps', feature_significance.plot_feature_sig, plot_args)
    #
    # plot_args = [{'x': m.cpu()} for m in XX_ig_captum]
    # visualization.plot_grid(4, 4, 'Captum_Lib_IG_maps', feature_significance.plot_feature_sig, plot_args)


    # --------- Visualize images and IG maps in same plot grid (2 plot grids for Custom IG and Captum IG) -----------

    plot1_args = []
    plot2_args = []
    for x, x1, x2 in zip(XX, XX_ig, XX_ig_captum):
        plot1_args.append({'x': x.cpu()})
        plot1_args.append({'x': x1.cpu()})

        plot2_args.append({'x': x.cpu()})
        plot2_args.append({'x': x2.cpu()})

    visualization.plot_grid(4, 8, 'Custom_IG_maps', visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 8, 'Captum_Lib_IG_maps', visualization.plot_feature_sig, plot2_args)

    common.save_all_figures('output/tests')


def visualize_activations_gradients_ig(network, testset):
    print('Visualizing feature maps of an intermediate layer')

    XX, yy = get_correct_pred_examples(testset, network, num_samples=16)
    # print(yy)

    block_idx = 7
    sub_block_idx = 1

    # -------- Compute intermediate layer activations ----------

    layer_activations = None    # Feature maps are saved here
    def _forward_hook_save(self, input, output):
        nonlocal layer_activations
        layer_activations = output

    network.conv_base[block_idx][sub_block_idx].conv2.register_forward_hook(_forward_hook_save)

    out = network(XX)

    print(f'XX.shape: {XX.shape}')  # (num_samples, num_channels, width, height)
    print(f'layer_activations.shape: {layer_activations.shape}')

    # Need to sum feature maps (channels) to get per-location activations that can be visualized with a color map
    layer_activation_mean = torch.mean(layer_activations, dim=1)     # Channels first
    print(f'layer_activation_mean.shape: {layer_activation_mean.shape}')

    # ----------------------------------------------------------

    # -------- Compute intermediate layer gradients ----------

    layer_gradients = layer_feature_significance.compute_layer_gradients(network, XX, yy, block_idx, sub_block_idx)
    print(f'layer_gradients.shape: {layer_gradients.shape}')  # (num_samples, num_channels, width, height)

    layer_gradients_mean = torch.mean(layer_gradients, dim=1)     # Channels first
    print(f'layer_gradients_mean.shape: {layer_gradients_mean.shape}')

    # ----------------------------------------------------------

    # -------- Compute intermediate layer integrated gradients (IG) ----------

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, baselines,
                                                                             block_idx, sub_block_idx, steps)
    print(f'layer_ig.shape: {layer_ig.shape}')  # (num_samples, num_channels, width, height)

    layer_ig_mean = torch.mean(layer_ig, dim=1)  # Channels first
    print(f'layer_ig_mean.shape: {layer_ig_mean.shape}')

    # ----------------------------------------------------------

    plot1_args = []
    plot2_args = []
    plot3_args = []
    for x, x1, x2, x3 in zip(XX, layer_activation_mean, layer_gradients_mean, layer_ig_mean):
        plot1_args.append({'x': x.cpu(), 'title': 'Input'})
        plot1_args.append({'x': x1.cpu(), 'title': 'Activ.'})

        plot2_args.append({'x': x.cpu(), 'title': 'Input'})
        plot2_args.append({'x': x2.cpu(), 'title': 'Grad.'})

        plot3_args.append({'x': x.cpu(), 'title': 'Input'})
        plot3_args.append({'x': x3.cpu(), 'title': 'IG'})

    visualization.plot_grid(4, 8, f'Layer_activations_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 8, f'Layer_gradients_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot2_args)

    visualization.plot_grid(4, 8, f'Layer_IG_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot3_args)

    common.save_all_figures('output/tests')


def visualize_layer_ig_attacks(network, testset):
    # epsilon = 100
    # alpha = 10
    # num_iter = 20

    epsilon = 8
    alpha = 2
    num_iter = 5
    adv_ig_steps = 5

    block_idx = 4
    sub_block_idx = 1

    XX, yy = get_correct_pred_examples(testset, network, num_samples=4)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_attacks = []
    for x, y in zip(XX, yy):
        x_attack = layer_ig_attack.create_layer_ig_attack_example(network, x, y, block_idx, sub_block_idx,
                                                                  epsilon, alpha, num_iter, adv_ig_steps)
        # print(f'x_attack.shape: {x_attack.shape}')
        XX_attacks.append(x_attack)

        # Compare x and x_attack
        # diff_elements_count = x.numel() - torch.isclose(x, x_attack, atol=1e-03, rtol=1e-02).sum()
        # print(f'different_elements_count = {diff_elements_count}')
        # print(x)
        # print(x_attack)
        # print ('--------------------------------')

    XX_attacks = torch.stack(XX_attacks)    # Turn the list of tensors into a single tensor with all attack examples
    print(f'XX_attacks.shape: {XX_attacks.shape}')

    XX_attacks_ig = feature_significance.compute_integrated_gradients(network, XX_attacks, yy, baselines, m=steps)

    # Need to sum channels to get per-pixel attributions that can be visualized with a color map
    XX_ig = torch.sum(torch.abs(XX_ig), dim=1)  # Channels first
    XX_attacks_ig = torch.sum(torch.abs(XX_attacks_ig), dim=1)  # Channels first

    # XX_attacks = torch.zeros(XX.shape).to(XX.device)
    # XX_attacks_ig = torch.zeros(XX.shape).to(XX.device)

    # ------- Layer IG maps of original X and attacks ------

    XX_layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, baselines,
                                                                             block_idx, sub_block_idx, steps)

    XX_layer_ig_mean = torch.mean(XX_layer_ig, dim=1)  # Channels first

    XX_attacks_layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX_attacks, yy, baselines,
                                                                                block_idx, sub_block_idx, steps)

    XX_attacks_layer_ig_mean = torch.mean(XX_attacks_layer_ig, dim=1)  # Channels first

    # --------------------------------------------------------

    # ------ Layer activations of original X and attacks -----

    layer_activations = None    # Feature maps are saved here
    def _forward_hook_save(self, input, output):
        nonlocal layer_activations
        layer_activations = output

    network.conv_base[block_idx][sub_block_idx].conv2.register_forward_hook(_forward_hook_save)

    out = network(XX)

    XX_layer_activations_mean = torch.mean(layer_activations, dim=1)     # Channels first

    out = network(XX_attacks)

    XX_attacks_layer_activations_mean = torch.mean(layer_activations, dim=1)     # Channels first

    # --------------------------------------------------------

    plot1_args = []
    for x, x1, x2, x3 in zip(XX, XX_ig, XX_attacks, XX_attacks_ig):
        plot1_args.append({'x': x.cpu(), 'title': 'Original'})
        plot1_args.append({'x': x1.cpu(), 'title': 'Orig_IG'})
        plot1_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot1_args.append({'x': x3.cpu(), 'title': 'Attack_IG'})

    plot2_args = []
    for x, x1, x2, x3 in zip(XX, XX_layer_ig_mean, XX_attacks, XX_attacks_layer_ig_mean):
        plot2_args.append({'x': x.cpu(), 'title': 'Original'})
        plot2_args.append({'x': x1.cpu(), 'title': 'Orig_L_IG'})
        plot2_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot2_args.append({'x': x3.cpu(), 'title': 'Attack_L_IG'})

    plot3_args = []
    for x, x1, x2, x3 in zip(XX, XX_layer_activations_mean, XX_attacks, XX_attacks_layer_activations_mean):
        plot3_args.append({'x': x.cpu(), 'title': 'Original'})
        plot3_args.append({'x': x1.cpu(), 'title': 'Orig_Act.'})
        plot3_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot3_args.append({'x': x3.cpu(), 'title': 'Attack_Act'})

    visualization.plot_grid(4, 4, f'Layer_IG_attack_input_IG_maps_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 4, f'Layer_IG_attack_layer_IG_maps_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot2_args)

    visualization.plot_grid(4, 4, f'Layer_IG_attack_layer_activations_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot3_args)

    common.save_all_figures('output/tests')


def visualize_layer_gradient_attacks(network, testset):
    epsilon = 8
    alpha = 2
    num_iter = 5
    adv_ig_steps = 5

    block_idx = 4
    sub_block_idx = 1

    XX, yy = get_correct_pred_examples(testset, network, num_samples=4)

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24
    XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)

    XX_attacks = []
    for x, y in zip(XX, yy):
        x_attack = layer_gradient_attack.create_layer_gradient_attack_example(network, x, y, epsilon, alpha, num_iter,
                                                                              block_idx, sub_block_idx)
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

    # ------- Layer IG maps of original X and attacks ------

    XX_layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX, yy, baselines,
                                                                                block_idx, sub_block_idx, steps)

    XX_layer_ig_mean = torch.mean(XX_layer_ig, dim=1)  # Channels first

    XX_attacks_layer_ig = layer_feature_significance.compute_layer_integrated_gradients(network, XX_attacks, yy,
                                                                                        baselines,
                                                                                        block_idx, sub_block_idx, steps)

    XX_attacks_layer_ig_mean = torch.mean(XX_attacks_layer_ig, dim=1)  # Channels first

    # --------------------------------------------------------

    # ------ Layer activations of original X and attacks -----

    layer_activations = None  # Feature maps are saved here

    def _forward_hook_save(self, input, output):
        nonlocal layer_activations
        layer_activations = output

    network.conv_base[block_idx][sub_block_idx].conv2.register_forward_hook(_forward_hook_save)

    out = network(XX)

    XX_layer_activations_mean = torch.mean(layer_activations, dim=1)  # Channels first

    out = network(XX_attacks)

    XX_attacks_layer_activations_mean = torch.mean(layer_activations, dim=1)  # Channels first

    # --------------------------------------------------------

    plot1_args = []
    for x, x1, x2, x3 in zip(XX, XX_ig, XX_attacks, XX_attacks_ig):
        plot1_args.append({'x': x.cpu(), 'title': 'Original'})
        plot1_args.append({'x': x1.cpu(), 'title': 'Orig_IG'})
        plot1_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot1_args.append({'x': x3.cpu(), 'title': 'Attack_IG'})

    plot2_args = []
    for x, x1, x2, x3 in zip(XX, XX_layer_ig_mean, XX_attacks, XX_attacks_layer_ig_mean):
        plot2_args.append({'x': x.cpu(), 'title': 'Original'})
        plot2_args.append({'x': x1.cpu(), 'title': 'Orig_L_IG'})
        plot2_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot2_args.append({'x': x3.cpu(), 'title': 'Attack_L_IG'})

    plot3_args = []
    for x, x1, x2, x3 in zip(XX, XX_layer_activations_mean, XX_attacks, XX_attacks_layer_activations_mean):
        plot3_args.append({'x': x.cpu(), 'title': 'Original'})
        plot3_args.append({'x': x1.cpu(), 'title': 'Orig_Act.'})
        plot3_args.append({'x': x2.cpu(), 'title': 'Attack'})
        plot3_args.append({'x': x3.cpu(), 'title': 'Attack_Act'})

    visualization.plot_grid(4, 4,
                            f'Layer_grad_attack_input_IG_maps_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot1_args)

    visualization.plot_grid(4, 4,
                            f'Layer_grad_attack_layer_IG_maps_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot2_args)

    visualization.plot_grid(4, 4,
                            f'Layer_grad_attack_layer_activations_(block_idx_{block_idx}-sub_block_idx_{sub_block_idx})',
                            visualization.plot_feature_sig, plot3_args)

    common.save_all_figures('output/tests')


def visualize_expected_gradients_path_explainer(network, testset):
    explainer = PathExplainerTorch(network)

    num_refs = 64   # This is the k value

    XX, yy = get_correct_pred_examples(testset, network, num_samples=16)

    # baseline_XX, yy = get_correct_pred_examples(testset, network, num_samples=100)

    print(f'Running path_explainer EG attributions. num_refs: {num_refs}')

    XX_eg = explainer.attributions(input_tensor=XX,
                                      baseline=XX,
                                      num_samples=num_refs,
                                      use_expectation=True,
                                      output_indices=yy)

    print(f'XX_eg.shape: {XX_eg.shape}')

    print('Computing IG maps (custom)')

    baselines = torch.zeros((1, *XX[0].shape)).to(XX.device)
    steps = 24

    # XX_ig = feature_significance.compute_integrated_gradients(network, XX, yy, baselines, m=steps)
    XX_ig = XX

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
    num_refs = 4   # This is the k value

    XX, yy = get_correct_pred_examples(testset, network, num_samples=64)

    print('Running EG attributions')

    # explainer = AttributionPriorExplainerCPU(XX, 5, k=num_refs)

    explainer = AttributionPriorExplainerGPU(XX, 5, k=num_refs)

    attributions = explainer.shap_values(network, input_tensor=XX)


    print(f'attributions.shape: {attributions.shape}')


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


    # -----------------------------------------------------
    # Run tests

    print('Running tests')

    # compare_gradients_single_vs_tensor_1(network, testset)
    # compare_gradients_single_vs_tensor_2(network, testset)

    # compare_layer_gradients_custom_vs_captum(network, testset)

    # visualize_activations_gradients_ig(network, testset)

    # gradients_speed_test(network, testset)

    # compare_integrated_gradients_custom_vs_captum(network, testset)

    # compare_visualization_ig_custom_vs_captum(network, testset)
    
    visualize_layer_gradient_attacks(network, testset)

    # visualize_layer_ig_attacks(network, testset)

    # visualize_expected_gradients_path_explainer(network, testset)

    # visualize_expected_gradients(network, testset)


if __name__ == '__main__':
    main()
