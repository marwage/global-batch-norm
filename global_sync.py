import mindspore as ms
import mindspore.communication as C
import numpy as np


def main():
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    C.init()
    ms.context.reset_auto_parallel_context()
    ms.context.set_auto_parallel_context(parallel_mode=ms.context.ParallelMode.DATA_PARALLEL)
    global_bn_op = ms.nn.SyncBatchNorm(num_features=3)
    x = ms.Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
    output = global_bn_op(x)
    print(output)

if __name__ == "__main__":
    main()
