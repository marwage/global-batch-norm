import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer


class KungfuBatchNorm(ms.nn.Cell):
    @cell_attr_register
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init="ones",
                 beta_init="zeros",
                 moving_mean_init="zeros",
                 moving_var_init="ones",
                 input_dims="2d",
                 data_format="NCHW"):
        super().__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError("num_features must be at least 1")
        self.num_features = num_features
        if momentum < 0 or momentum > 1:
            error_msg = "momentum should be a number in range [0, 1], but got {}".format(momentum)
            raise ValueError(error_msg)
        self.momentum = 1.0 - momentum
        self.input_dims = input_dims
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if ms.context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.eps = eps
        self.moving_mean = ms.Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = ms.Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = ms.Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = ms.Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)

        self._cluster_size_op = kfops.KungFuClusterSize()
        self._all_reduce_op = kfops.KungFuAllReduce()
        self._square_op = ms.ops.Square()
        self._sqrt_op = ms.ops.Sqrt()

        # DEBUG
        self._print_op = ms.ops.Print()

    def _construct(self, x):
        # Assume x of shape (N, C, H, W)
        batch_size = x.shape[0]
        if self.training:
            # calculate global batch size
            cluster_size = self._cluster_size_op()
            global_batch_size = batch_size * cluster_size

            # mean along N
            sum_local_batch = x.sum(axis=0)
            sum_global_batch = self._all_reduce_op(sum_local_batch)
            mean_global_batch = sum_global_batch / global_batch_size

            # calculate expected value
            expected_value = self.moving_mean.copy()
            for i in range(self.num_features):
                expected_value[i] = mean_global_batch[i].mean()

            # calculate variance
            variance = self.moving_variance.copy()
            local_squared = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    local_squared[i, j] = self._square_op(x[i, j] - expected_value[j])
            sum_local_var = local_squared.sum(axis=0)
            sum_global_var = self._all_reduce_op(sum_local_var)
            mean_variance = sum_global_var / global_batch_size
            for j in range(self.num_features):
                variance[j] = mean_variance[j].mean()

            # normalise input
            x_norm = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    zero_mean = (x[i, j] - expected_value[j])
                    one_var = self._sqrt_op(variance[j] + self.eps)
                    x_norm[i, j] = zero_mean / one_var

            for i in range(batch_size):
                for j in range(self.num_features):
                    x_norm[i][j] = self.gamma[j] * x_norm[i][j] + self.beta[j]

            for j in range(self.num_features):
                # DEBUG
                #  self._print_op("mean ", j, " ", expected_value[j])
                #  self._print_op("variance ", j, " ", variance[j])

                self.moving_mean[j] = ((1 - self.momentum) * self.moving_mean[j]
                                   + self.momentum * expected_value[j])
                self.moving_variance[j] = ((1 - self.momentum) * self.moving_variance[j]
                                       + self.momentum * variance[j])

                # DEBUG
                #  self._print_op("moving mean ", j, " ", self.moving_mean[j])
                #  self._print_op("moving variance ", j, " ", self.moving_variance[j])

            # DEBUG
            for j in range(self.num_features):
                self._print_op("moving mean ", j, " ", self.moving_mean[j])
                self._print_op("moving variance ", j, " ", self.moving_variance[j])

            return x_norm

        # inference
        x_norm = x.copy()
        for i in range(batch_size):
            for j in range(self.num_features):
                x_norm[i][j] = ((x[i][j] - self.moving_mean[j]) /
                               self._sqrt_op(self.moving_variance[j] + self.eps))
                x_norm[i][j] = self.beta[j] * x_norm[i][j] + self.gamma[j]

        return x_norm


    def construct(self, x):
        # Assume x of shape (N, C, H, W)
        batch_size = x.shape[0]
        if self.training:
            # calculate global batch size
            cluster_size = self._cluster_size_op()
            global_batch_size = batch_size * cluster_size

            # mean along N
            sum_local_batch = x.sum(axis=0)
            sum_global_batch = self._all_reduce_op(sum_local_batch)
            mean_global_batch = sum_global_batch / global_batch_size

            # calculate expected value
            expected_value = mean_global_batch.mean(axis=[1, 2])

            # calculate variance
            local_squared = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    local_squared[i, j] = self._square_op(x[i, j] - expected_value[j])
            sum_local_var = local_squared.sum(axis=0)
            sum_global_var = self._all_reduce_op(sum_local_var)
            mean_variance = sum_global_var / global_batch_size
            variance = mean_variance.mean(axis=[1, 2])

            # normalise input
            x_norm = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    zero_mean = (x[i, j] - expected_value[j])
                    one_var = self._sqrt_op(variance[j] + self.eps)
                    x_norm[i, j] = zero_mean / one_var

            for i in range(batch_size):
                for j in range(self.num_features):
                    x_norm[i][j] = self.gamma[j] * x_norm[i][j] + self.beta[j]

            self.moving_mean = ((1 - self.momentum) * self.moving_mean
                               + self.momentum * expected_value)
            self.moving_variance = ((1 - self.momentum) * self.moving_variance
                                   + self.momentum * variance)

            return x_norm

        # inference
        x_norm = x.copy()
        for i in range(batch_size):
            for j in range(self.num_features):
                x_norm[i][j] = ((x[i][j] - self.moving_mean[j]) /
                               self._sqrt_op(self.moving_variance[j] + self.eps))
                x_norm[i][j] = self.beta[j] * x_norm[i][j] + self.gamma[j]

        return x_norm


def test_kungfu():
    ms.common.set_seed(42)
    np.random.seed(42)
    device = "GPU"
    kfops.init(device)

    num_features = 3
    global_bn_op = KungfuBatchNorm(num_features)
    global_bn_op.set_train()

    x_np = (np.random.rand(4, 3, 12, 24) * 10).astype(np.float32)
    rank = kfops.kungfu_current_rank()
    if rank == 0:
        x_shard = x_np[:2]
    else:
        x_shard = x_np[2:]

    # DEBUG
    #  print("input shape")
    #  print(x_shard.shape)

    x = ms.Tensor(x_shard)
    kf_out = global_bn_op(x)

    # DEBUG
    #  print("output shape")
    #  print(kf_out.asnumpy().shape)

    ms_bn_op = ms.nn.BatchNorm2d(num_features)
    ms_bn_op.set_train()
    x = ms.Tensor(x_np)
    ms_out = ms_bn_op(x)

    if rank == 0:
        diff = kf_out - ms_out[:2]
    else:
        diff = kf_out - ms_out[2:]
    print("max diff {}".format(diff.max()))

    kfops.finalize(device)

def test_kungfu_single():
    ms.common.set_seed(42)
    np.random.seed(42)
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
    kf_bn_op.set_train()
    #  kf_bn_op.set_train(False)

    x_np = (np.random.rand(4, 3, 12, 24) * 10).astype(np.float32)

    # DEBUG
    #  print("input shape")
    #  print(x_np.shape)
    #  print("input")
    #  print(x_np)

    x = ms.Tensor(x_np)
    kf_out = kf_bn_op(x)

    # DEBUG
    #  print("output shape")
    #  print(kf_out.asnumpy().shape)
    #  print("output")
    #  print(kf_out)

    ms_bn_op = ms.nn.BatchNorm2d(num_features)
    ms_bn_op.set_train()
    #  ms_bn_op.set_train(False)
    x = ms.Tensor(x_np)
    ms_out = ms_bn_op(x)

    # DEBUG
    #  print("ms output")
    #  print(ms_out)

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
    #  test_kungfu()
    test_kungfu_single()


if __name__ == "__main__":
    main()
