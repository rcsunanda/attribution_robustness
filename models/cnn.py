import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from models.standardization import PerImageStandardization


class ConvolutionalNetwork(nn.Module):
    """ A fully connected network with architecture given by params.
    logits output (not softmaxed)
    """
    def __init__(self, params):
        super(ConvolutionalNetwork, self).__init__()

        conv_base_configs = params['conv_base']
        classifier_configs = params['classifier']
        input_shape = params['input_shape']
        num_classes = params['num_classes']
        freeze_conv_base = params['freeze_conv_base']

        if params['normalization'] == 'per_image':
            print('Small CNN: Per image normalization applied in the model forward call')
            self.per_img_norm = PerImageStandardization()
        else:
            self.per_img_norm = None

        self.conv_base = self._create_conv_base(conv_base_configs, freeze_conv_base)
        print(self.conv_base)

        conv_base_output_shape = self._get_module_output_shape(self.conv_base, input_shape)
        conv_base_output_features = conv_base_output_shape[1]

        self.classifier = self._create_classifier(classifier_configs, conv_base_output_features, num_classes)
        print(self.classifier)

        classifier_output_shape = self._get_module_output_shape(self.classifier, conv_base_output_shape)

        print(f'No of trainable parameters in model: {self.count_trainable_parameters():,}')
        print('conv_base output shape: ', conv_base_output_shape)
        print('classifier output shape: ', classifier_output_shape)

    # @autocast()
    def forward(self, x):
        if self.per_img_norm is not None:
            x = self.per_img_norm(x)

        x = self.conv_base(x)
        x = self.classifier(x)
        return x

    def _create_conv_base(self, conv_base_configs, freeze_conv_base):
        layer_modules = []
        # print(f'conv_base_configs. type: {type(conv_base_configs)}, val: {conv_base_configs}')

        for layer_config in conv_base_configs:
            # print(f'layer_config. type: {type(layer_config)}, val: {layer_config}')

            layer_config = layer_config.copy()  # Make a copy and work on it (we have to delete some keys
            layer_type = layer_config['type']
            del layer_config['type']

            if layer_type == 'Conv2d':
                out_features = layer_config['out_channels'] # For use in subsequent BatchNorm layer
                layers = [nn.Conv2d(**layer_config),
                          nn.ReLU(inplace=True),
                          ]
                torch.nn.init.kaiming_uniform_(layers[0].weight)
            elif layer_type == 'BatchNorm':
                layers = [nn.BatchNorm2d(num_features=out_features)]
            elif layer_type == 'MaxPool2d':
                layers = [nn.MaxPool2d(**layer_config)]
            elif layer_type == 'AdaptiveAvgPool2d':
                layers = [nn.AdaptiveAvgPool2d(**layer_config)]
            elif layer_type == 'Dropout':
                layers = [nn.Dropout(**layer_config)]
            else:
                assert False

            layer_modules.extend(layers)

        layer_modules.append(nn.Flatten())   # Output of conv_base is flattened
        conv_base = nn.Sequential(*layer_modules)

        if freeze_conv_base:
            for param in conv_base.parameters():
                param.requires_grad = False

        return conv_base

    def _create_classifier(self, classifier_configs, in_features, num_classes):
        layer_modules = []
        for layer_config in classifier_configs:
            layer_config = layer_config.copy()  # Make a copy and work on it (we have to delete some keys

            layer_type = layer_config['type']
            del layer_config['type']

            if layer_type == 'ReLU':
                out_features = layer_config['out_features']
                layers = [nn.Linear(in_features=in_features, out_features=out_features),
                          nn.ReLU(inplace=True),
                          ]
                torch.nn.init.kaiming_uniform_(layers[0].weight)
            elif layer_type == 'Dropout':
                layers = [nn.Dropout(**layer_config)]
            elif layer_type == 'BatchNorm':
                layers = [nn.BatchNorm1d(num_features=in_features)]
            else:
                assert False

            in_features = out_features  # For next layer

            layer_modules.extend(layers)

        # Output layer (logits with no activation)
        output_layer = nn.Linear(in_features=in_features, out_features=num_classes)
        torch.nn.init.kaiming_uniform_(output_layer.weight)
        layer_modules.append(output_layer)

        classifier = nn.Sequential(*layer_modules)
        return classifier

    def _get_module_output_shape(self, module, input_shape):
        x = torch.rand(1, *input_shape)   # Create 1 random input instance
        out = module(x)
        return out.shape

    def count_trainable_parameters(self):
        assert isinstance(self, nn.Module)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
