import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from datasets import build_test_loader, build_train_loader_da, build_dataset,build_train_loader_da_dy_cluster,build_cluster_loader
from utils.transforms import build_transforms
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch_da, crop_image
from models.seqnet_da import SeqNetDa
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from apex import amp
from spcl.models.dsbn import convert_dsbn
from spcl.models.hm import HybridMemory
from spcl.utils.faiss_rerank import compute_jaccard_distance,update_target_memory
import torch.nn.functional as F
from spcl.evaluators import Evaluator, extract_features,extract_dy_features
from sklearn.cluster import DBSCAN
import collections


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model and convert dsbn")
    model = SeqNetDa(cfg)
    convert_dsbn(model.roi_heads.reid_head)

    model.to(device)

    print("Building dataset")
    transforms = build_transforms(is_train=False)
    dataset_source_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)
    source_classes  = dataset_source_train.num_train_pids
    print("source_classes :"+str(source_classes))

    print("Loading test data")
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
        tgt_cluster_loader = build_cluster_loader(cfg, dataset_target_train)
        model.eval()
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)
    # Create hybrid memory
    memory = HybridMemory(256, source_classes, source_classes,
                            temp=0.05, momentum=0.2).to(device)
    
    # init source domian identity level centroid
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = build_cluster_loader(cfg,dataset_source_train)
    sour_fea_dict = extract_dy_features(model, sour_cluster_loader, device, is_source=True)
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)
    print("source_centers length")
    print(len(source_centers))
    print(source_centers.shape)
    print("the last one is the feature of 5555, remember don't use it")

    memory.features = source_centers.cuda()
    del source_centers, sour_fea_dict, sour_cluster_loader


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    model.roi_heads.memory = memory
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    target_start_epoch = cfg.TARGET_REID_START
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    del dataset_source_train
    transforms = build_transforms(is_train=True)
    dataset_source_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)
    dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        
        if (epoch==target_start_epoch):
            
            # DBSCAN cluster
            eps = 0.5
            eps_tight = eps-0.02
            eps_loose = eps+0.02
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

        if (epoch>=target_start_epoch):
            # init target domain instance level features
            # we can't use target domain GT detection box feature to init, this is only for measuring the upper bound of cluster performance
            #for dynamic clustering method, we use the proposal after several epoches for first init, moreover, we'll update the memory with proposal before each epoch
            print("==> Initialize target-domain instance features in the hybrid memory")
            transforms = build_transforms(is_train=False)
            dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
            tgt_cluster_loader = build_cluster_loader(cfg,dataset_target_train)
            if epoch==target_start_epoch:
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(model, tgt_cluster_loader, device, is_source=False)
            else:
                target_features = memory.features[source_classes:].data.cpu().clone()
                #target_features = memory.features[source_classes:source_classes+len(sorted_keys)].data.cpu().clone()
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(model, tgt_cluster_loader, device, is_source=False, memory_proposal_boxes=img_proposal_boxes, memory_target_features=target_features)
            sorted_keys = sorted(target_features.keys())
            print("target_features instances :"+str(len(sorted_keys)))
            target_features = torch.cat([target_features[name] for name in sorted_keys], 0)
            target_features = F.normalize(target_features, dim=1).cuda()
            
            negative_fea = torch.cat([negative_fea[name] for name in sorted(negative_fea.keys())], 0)
            print(negative_fea.shape)
            negative_fea = F.normalize(negative_fea, dim=1).cuda()
            print("hard negative instances :"+str(len(negative_fea)))

            source_centers = memory.features[0:source_classes].clone()
            memory.features = torch.cat((source_centers, target_features), dim=0).cuda()
            del source_centers,target_features, tgt_cluster_loader
            transforms = build_transforms(is_train=True)
            dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
            
            # Calculate distance
            print('==> Create pseudo labels for unlabeled target domain with self-paced policy')
            target_features = memory.features[source_classes:].clone()

            rerank_dist = compute_jaccard_distance(target_features, k1=30, k2=6, search_option=3, use_float16=True)
            del target_features
            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
            pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print("pseudo_labels length :")
            print(len(pseudo_labels))
            print(pseudo_labels)
            num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
            num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)
            
            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num):
                labels = []
                outliers = 0
                for i, id in enumerate(cluster_id):
                    if id!=-1:
                        labels.append(source_classes+id)
                    else:
                        labels.append(source_classes+num+outliers)
                        outliers += 1
                return torch.Tensor(labels).long()

            pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
            pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
            pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

            # compute R_indep and R_comp
            N = pseudo_labels.size(0)
            label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
            label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
            label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

            R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
            R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
            assert((R_comp.min()>=0) and (R_comp.max()<=1))
            assert((R_indep.min()>=0) and (R_indep.max()<=1))

            cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
            cluster_img_num = collections.defaultdict(int)
            for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
                cluster_R_comp[label.item()-source_classes].append(comp.item())
                cluster_R_indep[label.item()-source_classes].append(indep.item())
                cluster_img_num[label.item()-source_classes]+=1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
            cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
            if (epoch==target_start_epoch):
                indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

            outliers = 0
            # use sorted_keys for searching pseudo_labels
            print('==> Modifying labels in target domain to build new training set')
            index_count = 0
            for i, anno in enumerate(dataset_target_train.annotations):
                boxes_nums = len(img_proposal_boxes[anno["img_name"]])
                anno["pids"]=torch.zeros(boxes_nums)
                anno["boxes"]=img_proposal_boxes[anno["img_name"]]
                for j in range(boxes_nums):
                    index = sorted_keys.index(anno["img_name"]+"_"+str(j))
                    label = pseudo_labels[index]
                    indep_score = cluster_R_indep[label.item()-source_classes]
                    comp_score = R_comp[index]
                    if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-source_classes])):
                        anno["pids"][j] = index_count+source_classes+1
                    else:
                        anno["pids"][j] = index_count+source_classes+1
                        pseudo_labels[index] = source_classes+len(cluster_R_indep)+outliers
                        outliers+=1
                    index_count += 1
                dataset_target_train.annotations[i] = anno
            print(index_count)
            # statistics of clusters and un-clustered instances
            '''index2label = collections.defaultdict(int)
            for label in pseudo_labels:
                index2label[label.item()]+=1
            print(sorted(index2label.items(), key=lambda d: d[1], reverse=True))
            index2label = np.fromiter(index2label.values(), dtype=float)
            print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                        .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))'''

            memory.features = torch.cat((memory.features, negative_fea), dim=0).cuda()
            # hard_negative cases are assigned with unused labels
            memory.labels = (torch.cat((torch.arange(source_classes), pseudo_labels , torch.arange(len(negative_fea))+pseudo_labels.max()+1))).to(device)
            memory.num_samples = memory.features.shape[0]
            print(len(memory.labels))
        else:
            memory.labels = (torch.arange(source_classes)).to(device)
            memory.num_samples = source_classes
        train_loader_s, train_loader_t = build_train_loader_da_dy_cluster(cfg, dataset_source_train, dataset_target_train)

        train_one_epoch_da(cfg, model, optimizer, train_loader_s, train_loader_t, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    'amp': amp.state_dict()
                },
                osp.join(output_dir, f"gt_epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    main(args)
