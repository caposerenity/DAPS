import numpy as np
import math
import sys
import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
from collections import OrderedDict

class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, source_classes, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes = source_classes
        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, is_source=True):

        inputs = F.normalize(inputs, dim=1)
        # inputs: B*2048, features: L*2048
        indexes = torch.cat(indexes)
        indexes = indexes -1 # background label = -1, unlabeled in source domain = 5554
        inds = (indexes>=0)
        indexes = indexes[inds]
        for i in range(len(indexes)):
            if (indexes[i] == 5554) and is_source:
                indexes[i] = self.source_classes-1
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        inputs = hm(inputs, indexes, self.features, self.momentum).float().cuda()

        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        sim = torch.zeros(labels.max()+1, B).float().cuda()
        inputs = inputs.float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        #print("sim")
        #print(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        loss = F.nll_loss(torch.log(masked_sim+1e-6), targets, ignore_index=self.source_classes-1)
        return loss
