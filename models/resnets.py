import torch
import torch.nn as nn
import torchvision
from models.standardization import PerImageStandardization
from torch.cuda.amp import autocast


class ResNet18(nn.Module):
    """ A CNN with ResNet18 convolutional base + a given fully connected classifier
    """
    def __init__(self, params):
        super(ResNet18, self).__init__()

        classifier_configs = params['classifier']
        num_classes = params['num_classes']
        pretrained = params['pretrained']
        kaiming_init = params['kaiming_init']
        freeze_conv_base = params['freeze_conv_base']

        if params['normalization'] == 'per_image':
            print('ResNet18(): Per image normalization applied in the model forward call')
            self.per_img_norm = PerImageStandardization()
        else:
            self.per_img_norm = None

        self.conv_base, conv_base_output_features = self._create_conv_base(pretrained, kaiming_init, freeze_conv_base)
        print(self.conv_base)

        self.classifier = self._create_classifier(classifier_configs, conv_base_output_features, num_classes)
        print(self.classifier)

        self.per_img_norm = PerImageStandardization()

        # conv_base_output_shape = self._get_module_output_shape(self.conv_base, params['input_shape'])
        # classifier_output_shape = self._get_module_output_shape(self.classifier, conv_base_output_shape)
        #
        # print('conv_base output shape: ', conv_base_output_shape)
        # print('classifier output shape: ', classifier_output_shape)

        print(f'No of trainable parameters in model: {self.count_trainable_parameters():,}')

    # @autocast()
    def forward(self, x):
        if self.per_img_norm is not None:
            x = self.per_img_norm(x)

        x = self.conv_base(x)
        x = self.classifier(x)
        return x

    def _create_conv_base(self, pretrained, kaiming_init, freeze_conv_base):
        resnet = torchvision.models.resnet18(pretrained)
        conv_base_output_features = resnet.fc.in_features

        conv_base_layers = list(resnet.children())[:-1]   # ResNet excluding the last FC layer
        conv_base_layers.append(nn.Flatten())
        conv_base = torch.nn.Sequential(*conv_base_layers)

        if kaiming_init:
            assert pretrained is False  # Invalid to have both pretrained = 1 and kaiming_init = 1
            self._kaiming_init_conv_layers(conv_base)

        if freeze_conv_base:
            for param in conv_base.parameters():
                param.requires_grad = False

        return conv_base, conv_base_output_features

    def _create_classifier(self, classifier_configs, in_features, num_classes):
        layer_modules = []
        for layer_config in classifier_configs:
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

        layer_modules.append(nn.Linear(in_features=in_features, out_features=num_classes))   # Output layer (logits with no activation)

        classifier = nn.Sequential(*layer_modules)
        return classifier

    def _get_module_output_shape(self, module, input_shape):
        x = torch.rand(1, *input_shape)   # Create 1 random input instance
        out = module(x)
        return out.shape

    def _kaiming_init_conv_layers(self, module):
        level0_conv_layers = [conv_layer for conv_layer in module.children() if type(conv_layer) == nn.Conv2d]
        seq_modules = [seq_layer for seq_layer in module.children() if type(seq_layer) == nn.Sequential]

        basic_blocks = []
        for seq_mod in seq_modules:
            blocks_in_seq = [child for child in seq_mod.children() if type(child) == torchvision.models.resnet.BasicBlock]
            basic_blocks.extend(blocks_in_seq)

        level1_conv_layers = []
        for block in basic_blocks:
            conv_layers_in_block = [child for child in block.children() if type(child) == type(child) == nn.Conv2d]
            level1_conv_layers.extend(conv_layers_in_block)

        # print(level0_conv_layers)
        # print(level1_conv_layers)

        conv_layers = level0_conv_layers + level1_conv_layers

        for layer in conv_layers:
            # print('Kaiming Init')
            torch.nn.init.kaiming_uniform_(layer.weight)

    def count_trainable_parameters(self):
        assert isinstance(self, nn.Module)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
