#!/bin/bash

. /home/marcel/Elasticity/Repository/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Elasticity/Repository/kungfu-mindspore/mindspore)

python kf_cluster_size_op.py
