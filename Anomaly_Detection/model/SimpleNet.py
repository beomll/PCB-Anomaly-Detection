import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet
from model.ResNeXt import resnext50


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetTrunk, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        out = self.avgpool(layer4)

        return layer1, layer2, layer3, out


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix):]] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def load_model(pretrained=False, path=None, model_type: str = 'resnet50'):
    if model_type == 'resnet50':
        model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])
        if pretrained and path is not None:
            state_dict = remove_prefix(torch.load(path)['state_dict'], 'backbone.')
            model.load_state_dict(state_dict, strict=False)

    else:
        model = resnext50()
        if pretrained and path is not None:
            state_dict = remove_prefix(torch.load(path)['state_dict'], 'backbone.')
            model.load_state_dict(state_dict, strict=False)

    return model


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, dropout: float = 0.5):
        super(FeatureAdapter, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.Dropout(dropout)
                                  )
        self.linear = nn.Sequential(nn.Linear(dim, dim),
                                    nn.LayerNorm(dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout)
                                    )

    def forward(self, x, origin_feature):
        return self.conv(x), self.linear(origin_feature)


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: float = 0.5):
        super(Discriminator, self).__init__()

        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=7),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Dropout(dropout)
                                  )

        self.score = nn.Sequential(nn.Linear(2048, 2048),
                                   nn.LayerNorm(2048),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(2048, 8),
                                   nn.Softmax(dim=1)
                                   )

        self.linear = nn.Sequential(nn.Linear(out_channels, 1)
                                    )

    def forward(self, x, origin_feature):
        out = self.conv(x)
        out = out.view(out.size(0), -1, self.out_channels)

        score = self.score(origin_feature)
        out = torch.einsum('bi, bij -> bj', score, out)
        out = F.relu(out)
        out = self.linear(out)

        return out


class SimpleNet(nn.Module):
    def __init__(self, pretrained, path, dropout=0.5, **kwargs):
        super(SimpleNet, self).__init__()

        self.backbone = load_model(pretrained=pretrained, path=path, model_type=kwargs['model_type'])

        self.pad_layer2 = nn.ZeroPad2d((14, 14, 14, 14))  # 28x28 -> 56x56
        self.pad_layer3 = nn.ZeroPad2d((21, 21, 21, 21))  # 14x14 -> 56x56

        self.domain_adapter = FeatureAdapter(1792, 512, dim=2048, dropout=dropout)
        self.discriminator = Discriminator(512, 512, dropout=dropout)

    def forward(self, x):
        layer1, layer2, layer3, out = self.backbone(x)

        layer2_padded = self.pad_layer2(layer2)
        layer3_padded = self.pad_layer3(layer3)

        out = out.flatten(1)
        layer_out = torch.cat([layer1, layer2_padded, layer3_padded], dim=1)
        domain_feature, domain_vector = self.domain_adapter(layer_out, out)

        out = self.discriminator(domain_feature, domain_vector).squeeze()

        return out


if __name__ == '__main__':
    import torch

    model = SimpleNet(pretrained=False, path=None, model_type='resnext')
    data = torch.randn(1, 3, 224, 224)
    out = model(data)
