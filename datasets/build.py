import torch

from utils.transforms import build_transforms
from utils.utils import create_small_table
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from .cuhk_sysu import CUHKSYSU
from .prw import PRW

def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" and dataset.name != "CUHK-SYSU-COCO" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
            # for train set, we need its num_train_pids for cluster init
            # unlabeled pid 5555
            dataset.num_train_pids = num_pids + 1
            # for train set(specifically target domain),we need num of boxes for init cluster
            dataset.num_boxes = num_boxes
    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))
    return dataset

def build_dataset(dataset_name, root, transforms, split, verbose=True, is_source=True):
    if dataset_name == "CUHK-SYSU":
        dataset = CUHKSYSU(root, transforms, split, is_source=is_source, build_tiny=False)
    elif dataset_name == "PRW":
        dataset = PRW(root, transforms, split, is_source=is_source, build_tiny=False)
    elif dataset_name == "PRWCOCO":
        dataset = PRWCOCO(root, transforms, split, is_source=is_source)
    elif dataset_name == "CUHK-SYSU-COCO":
        dataset = CUHKSYSUCOCO(root, transforms, split, is_source=is_source)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")
    if verbose:
        dataset = print_statistics(dataset)
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_train_loader(cfg):
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    #datasampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=cfg.LOCAL_RANK)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        #sampler=datasampler
    )

def build_train_loader_da(cfg):
    transforms = build_transforms(is_train=True)
    dataset_s = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)
    #datasampler_s = DistributedSampler(dataset_s, num_replicas=dist.get_world_size(), rank=cfg.LOCAL_RANK)
    dataset_t = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
    #datasampler_t = DistributedSampler(dataset_t, num_replicas=dist.get_world_size(), rank=cfg.LOCAL_RANK)
    return torch.utils.data.DataLoader(
        dataset_s,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    #    sampler=datasampler_s
    ), torch.utils.data.DataLoader(
        dataset_t,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        #sampler=datasampler_t
    )

def build_train_loader_da_dy_cluster(cfg, dataset_s, dataset_t):
    return torch.utils.data.DataLoader(
        dataset_s,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    ), torch.utils.data.DataLoader(
        dataset_t,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

def build_test_loader(cfg, double_test=False):
    transforms = build_transforms(is_train=False)
    gallery_set = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "gallery")
    query_set = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "query")
    if double_test:
        gallery_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "gallery")
        query_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "query")
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return gallery_loader, query_loader

def build_cluster_loader(cfg,dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )
