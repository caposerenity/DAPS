#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# python train.py --cfg exp_cuhk/config.yaml --eval --ckpt exp_cuhk/epoch_19.pth
# python train.py --cfg exp_cuhk/config_test.yaml --eval --ckpt exp_cuhk/epoch_19.pth
# python train.py --cfg exp_prw/config_test.yaml --eval --ckpt exp_prw/epoch_17.pth

# python train.py --cfg exp_cuhk_da/config_test.yaml --eval --ckpt exp_cuhk_da/epoch_19.pth
# python train.py --cfg exp_cuhk_da_consist/config_test.yaml --eval --ckpt exp_cuhk_da_consist/epoch_19.pth
# python train.py --cfg exp_prw_da/config_test.yaml --eval --ckpt exp_prw_da/epoch_17.pth
# python train.py --cfg exp_prw_da_consist/config_test.yaml --eval --ckpt exp_prw_da_consist/epoch_17.pth

# python train.py --cfg exp/exp_cuhk_coco_da/config_test.yaml --eval --ckpt exp/exp_cuhk_coco_da/epoch_27.pth
#python train.py --cfg exp/exp_prwcoco_da/config_test.yaml --eval --ckpt exp/exp_prwcoco_da/epoch_17.pth
CUDA_VISIBLE_DEVICES=2 nohup python -u train_da_dy_cluster.py --cfg configs/cuhk_sysu_da.yaml >testres.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_da_dy_cluster.py --cfg /data/Projects_Data/SeqNet-DA/exp/exp_cuhk_da/config_test.yaml --eval --ckpt /data/Projects_Data/SeqNet-DA/exp/exp_cuhk_da/gt_epoch_6.pth >testres.out 2>&1 &