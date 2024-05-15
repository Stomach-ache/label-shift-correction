# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

import random
import math
# __all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

from .Expert_ResNeXt_sade import Bottleneck

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, returns_feat=True, use_experts=None, s=30):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            s = 1

        if use_experts is None:
            self.use_experts = list(range(num_experts))
        elif use_experts == "rand":
            self.use_experts = None
        else:
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s
        self.returns_feat = returns_feat
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)

        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        outs = []
        self.feat = []
        # self.logits = outs
        self.feat_before_GAP = []
        
        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts
        
        for ind in use_experts:
            feat_one = self._separate_part(out, ind)
            outs.append(feat_one)
            self.feat.append(feat_one)
            self.feat_before_GAP.append(feat_one)
        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.returns_feat:
            return {
                "output": final_out, 
                "feat": self.feat,
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out


class ResNet32SADE(nn.Module):  # From LDAM_DRW
    def __init__(self, num_classes=100, reduce_dimension=True, layer2_output_dim=None, layer3_output_dim=None,
                 use_norm=True, num_experts=3, returns_feat=True, **kwargs):
        super().__init__()
        self.backbone = ResNet_s(BasicBlock, [5, 5, 5],
                                 num_classes=num_classes, reduce_dimension=reduce_dimension,
                                 layer2_output_dim=layer2_output_dim,
                                 layer3_output_dim=layer3_output_dim, use_norm=use_norm,
                                 num_experts=num_experts,
                                 returns_feat=returns_feat,
                                 **kwargs)
        self.experts = [SubNet(self.backbone, 0)] + [SubNet(self.backbone, i) for i in range(1, num_experts)]

    def __getitem__(self, item):
        return self.experts[item]

    def forward(self, x, mode=None):
        x = self.backbone(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, num_classes=1000, use_norm=False,
                 reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None, share_layer3=False,
                 returns_feat=False, s=30):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList(
                [self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList(
            [self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList(
                [NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList(
                [nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            s = 1

        self.s = s

        self.returns_feat = returns_feat

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)

        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        outs = []
        self.feat = []
        # self.logits = outs
        self.feat_before_GAP = []

        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts

        for ind in use_experts:
            feat_one = self._separate_part(out, ind)
            outs.append(feat_one)
            self.feat.append(feat_one)
            self.feat_before_GAP.append(feat_one)
        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.returns_feat:
            return {
                "output": final_out,
                "feat": self.feat,
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out


class ResNext(nn.Module):

    def __init__(self, block, layers, num_experts, groups=1, width_per_group=64, dropout=None, num_classes=1000,
                 use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 returns_feat=False, s=30):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNext, self).__init__()

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        self.layer3s = nn.ModuleList(
            [self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList(
            [self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if use_norm:
            self.linears = nn.ModuleList(
                [NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList(
                [nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            s = 1

        self.s = s

        self.returns_feat = returns_feat

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks - 1)))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        outs = []
        self.feat = []
        for ind in range(self.num_experts):
            feat_one = self._separate_part(x, ind)
            outs.append(feat_one)
            self.feat.append(feat_one)
            self.feat_before_GAP.append(feat_one)
        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if self.returns_feat:
            return {
                "output": final_out,
                "feat": self.feat,
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out


class ResNeXt50SADE(nn.Module):
    def __init__(self, num_classes=1000, reduce_dimension=False, layer3_output_dim=None, layer4_output_dim=None,
                 num_experts=1, **kwargs):
        super().__init__()
        self.backbone = ResNext(Bottleneck, [3, 4, 6, 3], groups=32,
                                width_per_group=4, dropout=None, num_classes=num_classes,
                                reduce_dimension=reduce_dimension,
                                layer3_output_dim=layer3_output_dim,
                                layer4_output_dim=layer4_output_dim, num_experts=num_experts,
                                **kwargs)
        self.experts = [SubNet(self.backbone, 0)] + [SubNet(self.backbone, i) for i in range(1, num_experts)]

    def __getitem__(self, item):
        return self.experts[item]

    def forward(self, x, mode=None):
        x = self.backbone(x)
        return x

class SubNet(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.main_model = model
        self.idx = idx

    def forward(self, x):
        if self.idx == 0:
            out = self.main_model(x)
        feat = self.main_model.feat[:, self.idx]
        return feat



