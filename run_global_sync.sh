RANK_SIZE=2

. /home/marcel/Elasticity/Repository/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Elasticity/Repository/kungfu-mindspore/mindspore)


/usr/local/bin/mpirun \
    -n $RANK_SIZE \
    --output-filename mpirun_log \
    python global.py
