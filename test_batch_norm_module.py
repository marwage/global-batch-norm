import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer
from mindspore.nn.layer.kungfu import KungfuBatchNorm


def test_kungfu():
    ms.common.set_seed(42)
    np.random.seed(42)
    training = True
    device = "GPU"
    if True:
        ms.context.set_context(mode=ms.context.GRAPH_MODE,
                               device_target=device)
    else:
        ms.context.set_context(mode=ms.context.PYNATIVE_MODE,
                               device_target=device)
    kfops.init(device)

    num_features = 3
    kf_bn_op = KungfuBatchNorm(num_features)
    kf_bn_op.set_train(training)

    x_np = (np.random.rand(4, 3, 12, 24) * 10).astype(np.float32)

    cluster_size = kfops.kungfu_current_cluster_size()
    if cluster_size > 1:
        # only works for cluster_size == 2
        rank = kfops.kungfu_current_rank()
        if rank == 0:
            x_np = x_np[:2]
        else:
            x_np = x_np[2:]

    if False:
        print("input shape")
        print(x_np.shape)
        print("input")
        print(x_np)

    x = ms.Tensor(x_np)
    kf_out = kf_bn_op(x)

    if False:
        print("output shape")
        print(kf_out.asnumpy().shape)
        print("output")
        print(kf_out)

    ms_bn_op = ms.nn.BatchNorm2d(num_features)
    ms_bn_op.set_train(training)
    x = ms.Tensor(x_np)
    ms_out = ms_bn_op(x)

    if False:
        print("ms output")
        print(ms_out)

    if True:
        diff = kf_out.asnumpy() - ms_out.asnumpy()
        diff = np.abs(diff)
        print("max diff {}".format(diff.max()))

    # COMPARE
    if False:
        kf_params = kf_bn_op.get_parameters()
        ms_params = ms_bn_op.get_parameters()
        for ms_param, kf_param in zip(ms_params, kf_params):
            a = ms_param.asnumpy()
            b = kf_param.asnumpy()
            if np.array_equal(a, b):
                print("{} Equal".format(ms_param.name))
            else:
                print("{} Unequal".format(ms_param.name))
                print("val ms {}".format(a))
                print("val kf {}".format(b))
                max_diff = np.max(np.abs(a - b))
                print("max diff {}".format(max_diff))
    if False:
        for attr in dir(ms_bn_op):
            if hasattr(ms_bn_op, attr) and hasattr(kf_bn_op, attr):
                if getattr(ms_bn_op, attr) != getattr(kf_bn_op, attr):
                    print("{} Unequal".format(attr))
                else:
                    print("{} Equal".format(attr))
    if False:
        if kf_bn_op.momentum == ms_bn_op.momentum:
            print("Momentum Equal")
    if True:
        print("kf moving mean {}".format(kf_bn_op.moving_mean.asnumpy()))
        print("ms moving mean {}".format(ms_bn_op.moving_mean.asnumpy()))
        print("kf moving variance {}".format(kf_bn_op.moving_variance.asnumpy()))
        print("ms moving variance {}".format(ms_bn_op.moving_variance.asnumpy()))


    kfops.finalize(device)

def main():
    test_kungfu()


if __name__ == "__main__":
    main()
