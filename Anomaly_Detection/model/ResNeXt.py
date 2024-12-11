import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, inner_channels, cardinality, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, 3, stride=1, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion),
        )

        self.relu = nn.ReLU()
        self.projection = projection

    def forward(self, x):
        residual = self.residual(x)
        if self.projection is not None:
            skip_connection = self.projection(x)
        else:
            skip_connection = x

        out = self.relu(residual + skip_connection)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, block_list, cardinality):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.in_channels = 64
        self.stage1 = self.make_stage(block, 128, block_list[0], stride=1, cardinality=cardinality)
        self.stage2 = self.make_stage(block, 256, block_list[1], stride=2, cardinality=cardinality)
        self.stage3 = self.make_stage(block, 512, block_list[2], stride=2, cardinality=cardinality)
        self.stage4 = self.make_stage(block, 1024, block_list[3], stride=2, cardinality=cardinality)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        layer_1 = self.stage1(x)
        layer_2 = self.stage2(layer_1)
        layer_3 = self.stage3(layer_2)
        x = self.stage4(layer_3)

        x = self.avgpool(x)

        return layer_1, layer_2, layer_3, x

    def make_stage(self, block, inner_channels, block_nums, stride, cardinality):
        if self.in_channels != inner_channels * block.expansion or stride != 1:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)
            )
        else:
            projection = None

        layers = []
        for idx in range(block_nums):
            if idx == 0:
                layers.append(block(self.in_channels, inner_channels, cardinality, stride, projection))
                self.in_channels = inner_channels * block.expansion
            else:
                layers.append(block(self.in_channels, inner_channels, cardinality))

        return nn.Sequential(*layers)


def resnext50():
    return ResNeXt(Bottleneck, [3, 4, 6, 3], cardinality=32)

def resnext101():
    return ResNeXt(Bottleneck, [3, 4, 23, 3], cardinality=32)

