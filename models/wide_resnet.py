import torch
import torch.nn as nn
import torch.nn.functional as F


class WideResNet(nn.Module):
    def __init__(self, params):
        super(WideResNet, self).__init__()

        num_classes = params['num_classes']

        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # Assuming the use of basic blocks as defined in ResNet architectures
        self.unit_1_0 = self._make_residual_block(16, 16, stride=1, activate_before_residual=True)
        self.unit_2_0 = self._make_residual_block(16, 32, stride=2)
        self.unit_3_0 = self._make_residual_block(32, 64, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_features=64, out_features=num_classes)

        print(self)

    def _make_residual_block(self, in_filters, out_filters, stride, activate_before_residual=False):
        layers = []

        if activate_before_residual:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))

        if in_filters != out_filters:
            downsample = nn.Sequential(
                                    # nn.AvgPool2d(stride, stride),
                                    nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=stride, bias=False)
                                    )
        else:
            downsample = None

        layers.append(ResidualBlock(downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('AAAAAAA x.shape: ', x.shape)
        x = self.init_conv(x)
        print('BBBBBBB x.shape: ', x.shape)

        x = self.unit_1_0(x)

        print('CCCCCCC x.shape: ', x.shape)

        x = self.unit_2_0(x)

        print('DDDDDDD x.shape: ', x.shape)

        x = self.unit_3_0(x)

        print('EEEEEEE x.shape: ', x.shape)

        x = F.relu(x)  # Apply ReLU activation after the last residual block

        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)  # Flatten the tensor before passing it to the fully connected layer

        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, downsample=None):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        return x
