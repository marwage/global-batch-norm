import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np


def main():
    ms.common.set_seed(42)
    np.random.seed(42)
    #  device = "GPU"
    device = "CPU"
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=device)
    kfops.init(device)

    kf_cluster_size_op = kfops.KungFuClusterSize()
    size = kf_cluster_size_op()
    print("size {}".format(size))

    kfops.finalize(device)


if __name__ == "__main__":
    main()
