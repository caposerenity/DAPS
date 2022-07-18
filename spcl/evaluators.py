from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
from eval_func import _compute_iou
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs

def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets

# This is only used for testing the upper limit
def extract_gt_features(model, data_loader, device, is_source):
    model.eval()
    sour_fea_dict = collections.defaultdict(list)
    tgt_fea = OrderedDict()
    img_proposal_boxes = collections.defaultdict(list)
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images, targets = to_device(images, targets, device)
            
            gt_labels = targets[0]["labels"]
            gt_boxes = [t["boxes"] for t in targets]
            img_name = targets[0]["img_name"]

            
            images, targets = model.transform(images, targets)
            features = model.backbone(images.tensors)
            box_features = model.roi_heads.box_roi_pool(features, gt_boxes, images.image_sizes)
            box_features = model.roi_heads.reid_head(box_features, is_source)
            embeddings, _ = model.roi_heads.embedding_head(box_features)
            embeddings = embeddings.data.cpu()
            for j in range(len(gt_boxes[0])):
                if is_source:
                    sour_fea_dict[gt_labels[j].item()].append(embeddings[j].unsqueeze(0))
                else:
                    tgt_fea[img_name+"_"+str(j)] = embeddings[j].unsqueeze(0)
    if is_source:
        return sour_fea_dict
    else:
        return tgt_fea

def extract_dy_features(model, data_loader, device, is_source, memory_proposal_boxes=None, memory_target_features=None, momentum=0.2):
    model.eval()
    sour_fea_dict = collections.defaultdict(list)
    tgt_fea = OrderedDict()
    negative_fea = OrderedDict()
    positive_fea = OrderedDict()
    img_proposal_boxes = collections.defaultdict(list)
    img_store_idx = unqualified_nums  = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images, targets = to_device(images, targets, device)
            
            #batch size是1
            gt_labels = targets[0]["labels"]
            gt_boxes = [t["boxes"] for t in targets]
            img_name = targets[0]["img_name"]

            if is_source:
                images, targets = model.transform(images, targets)
                features = model.backbone(images.tensors)
                box_features = model.roi_heads.box_roi_pool(features, gt_boxes, images.image_sizes)
                box_features = model.roi_heads.reid_head(box_features, is_source)
                embeddings, _ = model.roi_heads.embedding_head(box_features)
                embeddings = embeddings.data.cpu()
                for j in range(len(gt_boxes[0])):
                    sour_fea_dict[gt_labels[j].item()].append(embeddings[j].unsqueeze(0))

            else:
                
                detections = model(images,is_source=is_source)
                
                boxes = detections[0]["boxes"].data.cpu()
                embeddings = detections[0]["embeddings"].data.cpu()
                scores = detections[0]["scores"]
                if len(boxes)==0:
                    #print("Here is an image without qualified proposal")
                    orig_thresh = model.roi_heads.score_thresh
                    model.roi_heads.score_thresh = 0
                    detections = model(images,is_source=is_source)
                    boxes = detections[0]["boxes"].data.cpu()
                    embeddings = detections[0]["embeddings"].data.cpu()
                    img_proposal_boxes[img_name].append(boxes[0].numpy().tolist())
                    img_proposal_boxes[img_name] = torch.Tensor(img_proposal_boxes[img_name]).data.cpu()
                    tgt_fea[img_name+"_"+str(0)] = embeddings[0].unsqueeze(0)
                    model.roi_heads.score_thresh = orig_thresh
                    unqualified_nums+=1
                    continue

                inds = (scores>=0.92)
                hard_inds = (scores>=0.8) ^ inds
                all_embeddings = embeddings.clone()
                hard_embeddings = embeddings[hard_inds]
                embeddings = embeddings[inds]
                all_boxes = boxes.clone()
                hard_boxes = boxes[hard_inds]
                boxes = boxes[inds]
                ious = []
                for j in range(len(embeddings)):
                    if memory_proposal_boxes is None:
                        tgt_fea[img_name+"_"+str(j)] = embeddings[j].unsqueeze(0)
                        img_proposal_boxes[img_name].append(boxes[j].numpy().tolist())
                    else:
                        #iou mapping
                        ious.append([])
                        for k in range(len(memory_proposal_boxes[img_name])):
                            iou = _compute_iou(memory_proposal_boxes[img_name][k],boxes[j])
                            ious[-1].append(iou)
                        if max(ious[-1])>0.7:
                            ious[-1] = ious[-1].index(max(ious[-1]))
                            memory_target_features[img_store_idx+ious[-1]] = momentum * memory_target_features[img_store_idx+ious[-1]] + (1. - momentum) * embeddings[j]
                            try:
                                memory_proposal_boxes[img_name][ious[-1]] =  momentum * memory_proposal_boxes[img_name][ious[-1]] + (1. - momentum) * boxes[j]
                            except TypeError as e:
                                print(e)
                                print(boxes[j])
                                print(memory_proposal_boxes[img_name])
                                print(ious)
                        else:
                            ious[-1] = -1
                
                # delete unmapped bboxes in memory
                if memory_proposal_boxes is not None:
                    instance_id = 0
                    for idx in range(len(memory_proposal_boxes[img_name])):
                        if idx in ious:
                            img_proposal_boxes[img_name].append(memory_proposal_boxes[img_name][idx].numpy().tolist())
                            tgt_fea[img_name+"_"+str(instance_id)] = memory_target_features[img_store_idx+idx].unsqueeze(0)
                            instance_id+=1
                    # add unmapped high-confidence proposal
                    for idx in range(len(ious)):
                        if ious[idx]==-1:
                            img_proposal_boxes[img_name].append(boxes[idx].numpy().tolist())
                            tgt_fea[img_name+"_"+str(instance_id)] = embeddings[idx].unsqueeze(0)
                            instance_id+=1
                    img_store_idx+=len(memory_proposal_boxes[img_name])

                if len(img_proposal_boxes[img_name])==0:
                        img_proposal_boxes[img_name].append(all_boxes[0].numpy().tolist())
                        tgt_fea[img_name+"_"+str(0)] = all_embeddings[0].unsqueeze(0)
                        unqualified_nums += 1
                
                img_proposal_boxes[img_name] = torch.Tensor(img_proposal_boxes[img_name]).data.cpu()

                hard_negative = torch.ones(hard_embeddings.shape[0])
                for j in range(hard_embeddings.shape[0]):
                    for k in range(len(img_proposal_boxes[img_name])):
                        if _compute_iou(hard_boxes[j],img_proposal_boxes[img_name][k])>0.3:
                            hard_negative[j] = 0
                            break
                hard_negative = (hard_negative>0)
                if hard_negative.sum()>0:
                    hard_embeddings = hard_embeddings[hard_negative]
                    negative_fea[img_name] = hard_embeddings

    if is_source:
        return sour_fea_dict
    else:
        print("unqualified_nums")
        print(unqualified_nums)
        return tgt_fea, img_proposal_boxes, negative_fea, positive_fea
            

def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
