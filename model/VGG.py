import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(3, 3, (1, 1))]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG('VGG19')
    print(net.features)