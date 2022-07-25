# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from models.da_loss import DALossComputation


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, lw_da_ins):
        super(DomainAdaptationModule, self).__init__()

        # self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = 256 #cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = res2_out_channels * stage2_relative_factor
        
        # self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = 1.0 #cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = 1.0 #cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = 0.1 #cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*0.1)
        self.grl_ins = GradientScalarLayer(-1.0*0.1)
        self.grl_ins_before = GradientScalarLayer(-1.0*0.1)
        self.grl_img_consist = GradientScalarLayer(1.0*0.1)
        self.grl_ins_consist = GradientScalarLayer(1.0*0.1)
        self.grl_ins_consist_before = GradientScalarLayer(1.0*0.1)
        
        in_channels = 256 * 4 #cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.lw_da_ins = lw_da_ins

        self.imghead = DAImgHead(1024)
        self.inshead = DAInsHead(256)
        self.inshead_before = DAInsHead(2048)
        self.loss_evaluator = DALossComputation()

    def forward(self, img_features, da_ins_feature, da_ins_labels, da_ins_feature_before, da_ins_labels_before, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        da_ins_feature_before = da_ins_feature_before.view(da_ins_feature_before.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        ins_grl_fea_before = self.grl_ins_before(da_ins_feature_before)

        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        ins_grl_consist_fea_before = self.grl_ins_consist_before(da_ins_feature_before)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_ins_features_before = self.inshead_before(ins_grl_fea_before)

        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_ins_consist_features_before = self.inshead_before(ins_grl_consist_fea_before)

        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()
        da_ins_consist_features_before = da_ins_consist_features_before.sigmoid()
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )
            da_img_loss, da_ins_loss_before, da_consistency_loss_before = self.loss_evaluator(
                da_img_features, da_ins_features_before, da_img_consist_features, da_ins_consist_features_before, da_ins_labels_before, targets
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * (self.lw_da_ins*da_ins_loss+(1.0-self.lw_da_ins)*da_ins_loss_before)
                #losses["loss_da_instance"] = self.ins_weight * da_ins_loss_before
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * (self.lw_da_ins*da_consistency_loss+(1.0-self.lw_da_ins)*da_consistency_loss_before)
                #losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss_before
            return losses
        return {}

