#!/bin/bash

RANK_SIZE=2

export CUDA_VISIBLE_DEVICES=0,1

. /home/marcel/Elasticity/Repository/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Elasticity/Repository/kungfu-mindspore/mindspore)

/home/marcel/KungFu/kungfu/bin/kungfu-run \
    -np $RANK_SIZE \
    -logfile kungfu-run.log \
    -logdir ./log \
    python kf_cluster_size_op.py
