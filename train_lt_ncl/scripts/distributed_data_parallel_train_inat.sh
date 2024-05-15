#!/bin/bash
#BSUB -J ncl_imagenet_x50_sade_config
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q zhangml
#BSUB  -gpu "num=2:mode=exclusive_process:aff=yes"
#module load anaconda3
#source activate LT

export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo


PYTHON=${PYTHON:-"python"}
CONFIG=config/iNat18/inat18_NCL_wo_hcm.yaml
NUM_GPUS=2
GPUS=2,3
PORT=${PORT:-27110}

echo CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        main/multi_network_train_final.py --cfg $CONFIG ${@:4}

CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
        main/multi_network_train_final.py --cfg $CONFIG ${@:4}
