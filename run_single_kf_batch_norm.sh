#!/bin/bash

RANK_SIZE=1

export CUDA_VISIBLE_DEVICES=2,3

. /home/marcel/Elasticity/Repository/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Elasticity/Repository/kungfu-mindspore/mindspore)

/home/marcel/KungFu/kungfu/bin/kungfu-run \
    -np $RANK_SIZE \
    -logfile kungfu-run.log \
    -logdir ./log \
    python kf_batch_norm.py
