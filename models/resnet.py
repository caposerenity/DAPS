from collections import OrderedDict
import torch.nn.functional as F
import torchvision
import torch
from torch import nn
from spcl.models.dsbn import DSBN2d, DSBN1d

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])
        #for i in range(4):
        #    x = super(Backbone, self).__getitem__(i)(x)
        #res2 = super(Backbone, self).__getitem__(4)(x)
        #res3 = super(Backbone, self).__getitem__(5)(res2)
        #res4 = super(Backbone, self).__getitem__(6)(res3)
        #return OrderedDict([["feat_res4", res4]])

class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])
    
    def bottleneck_forward(self, bottleneck, x, is_source):
        identity = x

        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, is_source)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                if not isinstance(module, DSBN2d):
                    identity = module(x)
                else:
                    identity = module(identity, is_source)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x, is_source=True):
        #对于reid head的dsbn特殊处理
        #需要取出没有child的module组成list一次执行，可以避免递归中重新实现所有带is_source的forward
        #Bottleneck的forward步骤有缺失
        module_seq=[]
        is_reid_head = False
        for _, (child_name, child) in enumerate(self.named_modules()):
            if isinstance(child, DSBN2d) or isinstance(child, DSBN1d):
                is_reid_head = True
            if isinstance(child, torchvision.models.resnet.Bottleneck):
                module_seq.append(child)
        if is_reid_head:
            feat = x.clone()
            for module in module_seq:
                feat = self.bottleneck_forward(module, feat, is_source)
        else:
            feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)
