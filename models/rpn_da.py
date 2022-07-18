from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.ops import boxes as box_ops

class RegionProposalNetworkDA(RegionProposalNetwork):
    # def assign_targets_to_anchors(self, anchors, targets):
    #     # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
    #     labels = []
    #     matched_gt_boxes = []
    #     mask = []
    #     for anchors_per_image, targets_per_image in zip(anchors, targets):
    #         gt_boxes = targets_per_image["boxes"]
    #         is_source = targets_per_image["domain_labels"]
    #         # print(is_source)

    #         mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
    #         mask.append(mask_per_image)
    #         if not is_source.any():
    #             continue

    #         if gt_boxes.numel() == 0:
    #             # Background image (negative example)
    #             device = anchors_per_image.device
    #             matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
    #             labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
    #         else:
    #             match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
    #             matched_idxs = self.proposal_matcher(match_quality_matrix)
    #             # get the targets corresponding GT for each proposal
    #             # NB: need to clamp the indices because we can have a single
    #             # GT in the image, and matched_idxs can be -2, which goes
    #             # out of bounds
    #             matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

    #             labels_per_image = matched_idxs >= 0
    #             labels_per_image = labels_per_image.to(dtype=torch.float32)

    #             # Background (negative examples)
    #             bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
    #             labels_per_image[bg_indices] = 0.0

    #             # discard indices that are between thresholds
    #             inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
    #             labels_per_image[inds_to_discard] = -1.0

    #         labels.append(labels_per_image)
    #         matched_gt_boxes.append(matched_gt_boxes_per_image)
    #     # print(labels.shape)
    #     return labels, matched_gt_boxes
    
    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (List[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        print('prp', proposals.shape, bboxes.shape, scores.shape)
        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            print('lab', labels.shape, matched_gt_bboxes.shape)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses
