import torch
import torch.nn as nn
from torch.nn import BatchNorm2d as BatchNorm2d


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation *
                               multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3 = BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.dilation = dilation
        self.downsample = downsample
        self.multi_grid = multi_grid

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample != None:
            residual = self.downsample(residual)
        return self.relu(x + residual)
        # return self.relu_inplace(x+residual)


class ResNet(nn.Module):
    def __init__(self, in_channel, layers, block):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv2d(in_channel, 64, 3, 2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpooling = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # self.maxpooling_ceil = nn.MaxPool2d(
        #     kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride, bias=False), BatchNorm2d(
                planes*block.expansion, affine=True))
        layers = []
        def generate_multi_grid(index, grids): return grids[index % len(
            grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation,
                            downsample, generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpooling(x)
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def main():
    resnet = ResNet(3, [3, 4, 23, 3], BottleNeck)
    x = torch.randn(1, 3, 128, 128)
    output = resnet(x)
    print(output.shape)

    from thop import profile
    params, flops = profile(resnet, inputs=(x,))
    print(params)
    print(flops)
    print(resnet(x).shape)


if __name__ == "__main__":
    main()
