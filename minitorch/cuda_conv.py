from numba import cuda
import numba
from torch.autograd import grad
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    MAX_DIMS,
)
from .tensor_functions import Function


to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32
MAX_KERNAL_SIZE = 32


@cuda.jit()
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    BLOCK_DIM = 32

    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    si = cuda.shared.array(BLOCK_DIM, dtype=numba.float64)

    x, y, z = cuda.grid(3)
    tx = cuda.threadIdx.x

    out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    out_index[0] = z  # current batch
    out_index[1] = y  # current output channel
    out_index[2] = x  # current pos in width

    in_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    in_index[0] = z

    wt_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    wt_index[0] = y

    tmp = 0.
    for i in range(in_channels):
        wt_index[1] = i
        in_index[1] = i
        in_index[2] = x

        if x < width:
            si[tx] = input[index_to_position(in_index, s1)]

        cuda.syncthreads()

        for j in range(kw):
            if not reverse:
                # this input cell has preloaded into shared input array -- si
                if tx + j < BLOCK_DIM:
                    val = si[tx + j]
                elif x + j < width:
                    in_index[2] = x + j
                    val = input[index_to_position(in_index, s1)]
                else:
                    val = 0.
                wt_index[2] = j
            else:
                # this input cell has preloaded into shared input array -- si
                if tx - j >= 0:
                    val = si[tx - j]
                elif x - j >= 0:
                    in_index[2] = x - j
                    val = input[index_to_position(in_index, s1)]
                else:
                    val = 0.
                wt_index[2] = kw - j - 1
            tmp += weight[index_to_position(wt_index, s2)] * val

        cuda.syncthreads()

    if x < out_width:
        out[index_to_position(out_index, out_strides)] = tmp
    

class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x w
            weight (:class:`Tensor`) : out_channel x in_channel x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))

        blockspergrid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out_channels,
            batch
        )
        threadsperblock = (THREADS_PER_BLOCK, 1, 1)

        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        
        blockspergrid = (
            (kw + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out_channels,
            in_channels
        )
        threadsperblock = (THREADS_PER_BLOCK, 1, 1)

        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)

        blockspergrid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            in_channels,
            batch
        )
        threadsperblock = (THREADS_PER_BLOCK, 1, 1)

        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_input.tuple(),
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d_cuda = Conv1dFun.apply


@cuda.jit()
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    BLOCK_DIM = 32

    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    si = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float64)

    x, y, z = cuda.grid(3)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    out_index[0] = int(z / out_channels)  # current batch
    out_index[1] = int(z % out_channels)  # current output channel
    out_index[2] = x                      # current pos in height
    out_index[3] = y                      # current pos in width

    in_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    in_index[0] = out_index[0]

    wt_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    wt_index[0] = out_index[1]

    tmp = 0.
    for i in range(in_channels):
        wt_index[1] = i
        in_index[1] = i
        in_index[2] = x
        in_index[3] = y

        if x < height and y < width:
            si[tx, ty] = input[index_to_position(in_index, s1)]

        cuda.syncthreads()

        for j in range(kh):
            for k in range(kw):
                if not reverse:
                    # this input cell has preloaded into shared input array -- si
                    if tx + j < BLOCK_DIM and ty + k < BLOCK_DIM:
                        val = si[tx + j, ty + k]
                    elif x + j < height and y + k < width:
                        in_index[2] = x + j
                        in_index[3] = y + k
                        val = input[index_to_position(in_index, s1)]
                    else:
                        val = 0.
                    wt_index[2] = j
                    wt_index[3] = k
                else:
                    # this input cell has preloaded into shared input array -- si
                    if tx - j >= 0 and ty - k >= 0:
                        val = si[tx - j, ty - k]
                    elif x - j >= 0 and y - k >= 0:
                        in_index[2] = x - j
                        in_index[3] = y - k
                        val = input[index_to_position(in_index, s1)]
                    else:
                        val = 0.
                    wt_index[2] = kh - j - 1
                    wt_index[3] = kw - k - 1
                tmp += weight[index_to_position(wt_index, s2)] * val

        cuda.syncthreads()

    if x < out_height and y < out_width:
        out[index_to_position(out_index, out_strides)] = tmp


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))

        blockspergrid = (
            (h + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            int(batch * out_channels)
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv2d[blockspergrid, threadsperblock](
            *output.tuple(), *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)

        blockspergrid = (
            (kh + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (kw + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            in_channels * out_channels
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)

        blockspergrid = (
            (h + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            int(batch * out_channels)
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_input.tuple(),
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d_cuda = Conv2dFun.apply
