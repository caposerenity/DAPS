import math
import sys
import os
from copy import deepcopy
from PIL import Image
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from apex import amp

from eval_func import eval_detection, eval_search_cuhk, eval_search_prw, _compute_iou
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


def train_one_epoch(cfg, model, optimizer, data_loader_s, data_loader_t, device, epoch, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", dataset='CUHK-SYSU')
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = min(len(data_loader_t) - 1, len(data_loader_s) - 1)
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, ((images1, targets1), (images2, targets2)) in enumerate(
        metric_logger.log_every(zip(data_loader_s, data_loader_t), cfg.DISP_PERIOD, header)
    ):
        images1, targets1 = to_device(images1, targets1, device)
        images2, targets2 = to_device(images2, targets2, device)

        loss_dict = model(images1, targets1, images2, targets2)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        # modify for using amp
        with amp.scale_loss(losses, optimizer) as scaled_loss:
            scaled_loss.backward()
        #losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader_s) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)

def train_one_epoch_da(cfg, model, optimizer, data_loader_s, data_loader_t, device, epoch, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", dataset='CUHK-SYSU')
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = min(len(data_loader_t) - 1, len(data_loader_s) - 1)
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, ((images1, targets1), (images2, targets2)) in enumerate(
        metric_logger.log_every(zip(data_loader_s, data_loader_t), cfg.DISP_PERIOD, header)
    ):

        images1, targets1 = to_device(images1, targets1, device)
        images2, targets2 = to_device(images2, targets2, device)

        loss_dict = model(images1, targets1, images2, targets2, epoch = epoch)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        with amp.scale_loss(losses, optimizer) as scaled_loss:
            scaled_loss.backward()
        #losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader_s) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

@torch.no_grad()
def crop_image(
    model, data_loader, device
):
    model.eval()
    place = "/data/PersonSearch/PRW/frames/"
    store_place = "/data/Re-ID/prw-crop/gt_train/"
    count = 0
    for images, targets in tqdm(data_loader, ncols=0):
        images, targets = to_device(images, targets, device)
        assert len(targets)==1
        gt_boxes = targets[0]["boxes"]
        gt_labels = targets[0]["labels"]
        count+=gt_boxes.shape[0]
        if not len(gt_boxes)==gt_boxes.shape[0]:
            print("gt_boxes")
            print(gt_boxes)
        name = targets[0]["img_name"]
        source_image = Image.open(os.path.join(place, name))
        name = name.split(".")[0]
        for i  in range(gt_boxes.shape[0]):
            det_box, gt_label = gt_boxes[i], gt_labels[i]
            img = source_image.crop(det_box.tolist())
            img.save(store_place+name+"_"+str(i)+"_"+str(gt_label.item())+".jpg")
        print(count)

@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" or gallery_loader.dataset.name == "CUHK-SYSU-COCO" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
