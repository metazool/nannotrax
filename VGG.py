# Lifted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

# This version ran but accuracy is <0.1 and the values for AvgPool size and the first linear input to the classifier are utterly arbitrary

import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)

class VGG(nn.Module):
    # Try not initialising weights
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # Try resizing the first linear layer
            nn.ReLU(True),
            nn.Linear(73728, 4096), #, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        logging.debug(x)
        logging.debug(x.dim())
        logging.debug(len(x))
        x = self.features(x)
        logging.debug(x.dim())
        x = x.view(x.size(0), -1)
        logging.debug(len(x))
        logging.debug(self.classifier)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        # The only significant change, use an adaptive maxpool
    #    if v == 'A':
    #        layers += [nn.AdaptiveAvgPool2d(1)]
        if v == 'A':
            layers += [nn.AdaptiveAvgPool2d((12,12))]
        elif v =='M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg_custom(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #cfg = [64, 'A', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    cfg = [64, 'A', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'A']
    model = VGG(make_layers(cfg),**kwargs)
    return model

