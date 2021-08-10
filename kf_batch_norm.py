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

        self._cluster_size_op = kfops.kungfu_current_cluster_size() # FIXME: not a Mindspore op
        self._all_reduce_op = kfops.KungFuAllReduce()
        self._square_op = ms.ops.Square()
        self._sqrt_op = ms.ops.Sqrt()

    def construct(self, x):
        if self.training:
            batch_size = x.shape[0]
            sum_local_batch = x.sum(axis=0)
            sum_global_batch = self._all_reduce_op(sum_local_batch)
            cluster_size = self._cluster_size_op # FIXME: not a Mindspore op
            global_batch_size = batch_size * cluster_size
            expected_value = sum_global_batch / global_batch_size

            sum_local_squ = self._square_op(x - expected_value)
            sum_local_var = sum_local_squ.sum(axis=0)
            sum_global_var = self._all_reduce_op(sum_local_var)
            variance = sum_global_var / global_batch_size

            x_norm = (x - expected_value) / self._sqrt_op(variance + self.eps)
            for i in range(batch_size):
                for j in range(self.num_features):
                    x_norm[i][j] = self.beta[j] * x_norm[i][j] + self.gamma[j]

            self._print_op("mov mean shape ", self.moving_mean.shape)
            self._print_op("exp val shape ", expected_value.shape)
            self._print_op("mov var shape ", self.moving_variance.shape)
            self._print_op("var shape", variance.shape)
            for i in range(self.num_features):
                self.moving_mean[i] = (self.momentum * self.moving_mean[i]
                                   + (1 - self.momentum) * expected_value[i].mean())
                self.moving_variance[i] = (self.momentum * self.moving_variance[i]
                                       + (1 - self.momentum) * variance[i].mean())

            return x_norm
        else: # inference
            batch_size = x.shape[0]
            x_norm = (x - self.moving_mean) / self._sqrt_op(self.moving_variance + self.eps)
            for i in range(batch_size):
                for j in range(self.num_features):
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

    x_np = (np.random.rand(4, 3, 12, 12) * 10).astype(np.float32)
    rank = kfops.kungfu_current_rank()
    if rank == 0:
        x_shard = x_np[:2]
    else:
        x_shard = x_np[2:]
    x = ms.Tensor(x_shard)
    kf_out = global_bn_op(x)

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

def main():
    test_kungfu()


if __name__ == "__main__":
    main()