import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F

import loralib as lora


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, r: float = 0.5) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return lora.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, r=r)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, r: float = 0.5) -> nn.Conv2d:
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return lora.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, r=r)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            r: float = 0.5
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, r=r)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, r=r)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, mode='all') -> Tensor:
        identity = x

        out = self.conv1(x, mode=mode)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, mode=mode)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample[0](x, mode=mode)
            identity = self.downsample[1](identity)
            # identity = self.downsample(x, mode=mode)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            layers: List[int],
            features: List[int] = [16, 64, 64, 16],
            num_labels: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            Conv_r: float = 0.5,
            Linear_r: int = 4
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = lora.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, r=Conv_r)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.convout = lora.Conv2d(64, 1, kernel_size=3, stride=1, padding=1,
                                 bias=False, r=1)
        # self.bn1 = norm_layer(self.inplanes)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = [self._make_layer(block, 64, layers[0], Conv_r=Conv_r)]
        for num in range(1, len(layers)):
            self.layers.append(self._make_layer(block, features[num], layers[num], stride=1,
                                                dilate=replace_stride_with_dilation[num - 1], Conv_r=Conv_r))
        self.layers = nn.Sequential(*self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_labels)
        self.fc = lora.Linear(features[len(layers) - 1] * block.expansion, num_labels, r=Linear_r)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, Conv_r: float = 0.5) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, r=Conv_r),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, r=Conv_r))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, r=Conv_r))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, mode='all') -> Tensor:
        x = self.conv1(x, mode=mode)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layers(x, mode=mode)
        for layer in self.layers:
            for block in layer:
                x = block(x, mode=mode)
                # print(x.shape)
            # x = layer(x, mode=mode)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.convout(x)
        x = self.tanh(x)
        # print(x.shape)
        # return F.log_softmax(x, dim=1)
        return x

    def forward(self, x: Tensor, mode='all') -> Tensor:
        return self._forward_impl(x, mode=mode)


def resnet18(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet10(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return ResNet(BasicBlock, [1, 1, 1], **kwargs)


def resnet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return ResNet(BasicBlock, [1, 1], **kwargs)


def resnet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return ResNet(BasicBlock, [1], **kwargs)

