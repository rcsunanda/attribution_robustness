import os.path

from torchsummary import summary
import torch

from models.cnn import ConvolutionalNetwork
from models.ann import FullyConnectedNetwork
from models.resnets import ResNet18
# from models.wide_resnet import WideResNet
from models.wide_resnet_2 import WideResNet


def load_model(params, set_weights=False):
    assert os.path.exists(params['model_state_path']), f'Model state path does not exist: {params["model_state_path"]}'

    network = create_network(params)

    device = params['device']
    state_dict = load_state_dict(params['model_state_path'], device)

    if set_weights:
        if 'cuda' in device:
            device_obj = torch.device('cuda:0')  # We only run evaluation on one gpu
            torch.cuda.set_device(device_obj)
            network = network.to(device_obj)  # Move model to device upfront

        load_weights_to_network(params['model'], network, state_dict, load_classifier_weights=True)

    return network, state_dict


def create_network(params):
    if params['model'] == 'small_cnn':
        network = ConvolutionalNetwork(params)
    elif params['model'] == 'resnet18':
        network = ResNet18(params)
    elif params['model'] == 'wide_resnet':
        network = WideResNet(params)
    elif params['model'] == 'ann':
        network = FullyConnectedNetwork(params)
    else:
        assert False, f'Unknown model: {params["model"]}'

    return network


def load_state_dict(path, device):
    print('Loading state dict (model & scalar states)')

    if 'cuda' in device:
        map_loc = {'cuda:0': f'cuda:0'}  # We only run evaluation on one gpu
    else:
        assert device == 'cpu'
        map_loc = torch.device('cpu')

    state_dict = torch.load(path, map_location=map_loc)
    return state_dict


def load_weights_to_network(model_name, network, state_dict, load_classifier_weights=True):
    if model_name == 'small_cnn' or model_name == 'resnet18':
        _load_cnn_weights_to_network(network, state_dict, load_classifier_weights)
    elif model_name == 'ann':
        _load_ann_weights_to_network(network, state_dict)
    elif model_name == 'wide_resnet':
        print('Third')
        network.load_state_dict(state_dict['model_state_dict'])
    else:
        assert False


def _load_cnn_weights_to_network(network, state_dict, load_classifier_weights):
    conv_base_state_dict = {}
    classifier_state_dict = {}

    for key, val in state_dict['model_state_dict'].items():
        if 'conv_base' in key:
            new_key = key.replace('module.', '')
            new_key = new_key.replace('conv_base.', '')
            conv_base_state_dict[new_key] = val

        if 'classifier' in key:
            new_key = key.replace('module.', '')
            new_key = new_key.replace('classifier.', '')
            classifier_state_dict[new_key] = val

    network.conv_base.load_state_dict(conv_base_state_dict)

    if load_classifier_weights:
        network.classifier.load_state_dict(classifier_state_dict)


def _load_ann_weights_to_network(network, state_dict):
    state_dict = state_dict['model_state_dict']
    network.load_state_dict(state_dict)


def get_device(gpu_id, params):
    # Set device (CPU or GPU with correct ID)

    assert params['device'] in ['cpu', 'cuda'] or 'cuda' in params['device']

    if params['device'] == 'cuda':
        print(f'Running on GPU (CUDA)')
        assert torch.cuda.is_available()
        params['device'] = f'cuda:{gpu_id}'
        torch.cuda.set_device(params['device'])
    elif params['device'] == 'cpu':
        print('Running on CPU')

    device_obj = torch.device(params['device'])  # eg: cpu, cuda:0, cuda:1
    return device_obj


def visualize_network(network, dataset, input_shape, tb_writer):
    print(network)
    print("Number of model parameters = ", sum(p.numel() for p in network.parameters()))

    summary(network, tuple(input_shape), device='cpu')

    # X_samples = dataset.tensors[0][0:4]
    X_samples, y = next(iter(dataset))
    # X_samples = X_samples[0:4]
    # print(f'Checking forward pass on a batch of shape {X_samples.shape}')
    # out = network(X_samples)    # To check early for any memory/ other issues in forward pass
    # torchviz.make_dot(out).render("output/network_viz", format="png")

    # print('Adding network graph to TensorBoard')
    # tb_writer.add_graph(network, X_samples)

