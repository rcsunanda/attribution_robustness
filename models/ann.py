import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNetwork(nn.Module):
    """ A fully connected network with architecture given by params.
    logits output (not softmaxed)
    """
    def __init__(self, params):
        super(FullyConnectedNetwork, self).__init__()

        self.layers = nn.ModuleList()

        input_nodes = params['ann_input_nodes']
        output_nodes = params['num_classes']
        layer_nodes = params['ann_layer_units']
        activations = params['ann_layer_activations']
        dropouts = params['ann_layer_dropout_rates']

        assert len(layer_nodes) <= len(activations)
        assert len(layer_nodes) <= len(dropouts)

        prev_layer_nodes = input_nodes
        for i in range(len(layer_nodes)):
            nodes = layer_nodes[i]
            fc = nn.Linear(prev_layer_nodes, nodes)
            assert activations[i] == 'relu' # Only ReLU is supported for now
            relu = nn.ReLU()
            dropout = nn.Dropout(dropouts[i])
            prev_layer_nodes = nodes
            self.layers.extend([fc, relu, dropout])

        # Output layer
        out = nn.Linear(prev_layer_nodes, output_nodes)
        self.layers.append(out)
        # out_softmaxed = nn.Softmax(dim=1) # Should not have a softmax. Loss must be computed on logits
        # self.layers.append(out_softmaxed)

        # print(self.layers)

    def forward(self, x):
        # print(f'ann.forward(). x.shape: {x.shape}')

        x = x.view(x.size(0), -1)   # Flatten image into a vector (this is now done outside the model)

        for layer in self.layers:
            x = layer(x)

        return x

    def group_params_by_layer(self):
        """ Create a param group (dictionary) per layer """
        param_groups = []
        for layer in self.layers:
            if type(layer) == nn.Linear:
                param_groups.append({"params": layer.parameters()})
        return param_groups
