from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    new_height = int(height / kh)
    new_width = int(width / kw)
    out = input.contiguous()
    out = out.view(batch, channel, height, new_width, kw)
    out = out.permute(0, 1, 3, 2, 4)
    out = out.contiguous()
    out = out.view(batch, channel, new_width, new_height, kh * kw)
    out = out.permute(0, 1, 3, 2, 4)
    return out


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError('Need to implement for Task 4.3')
    out = tile(input, kernel)
    out = out.mean(4)
    out = out.view(batch, channel, out.shape[2], out.shape[3])
    return out


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError('Need to implement for Task 4.4')
        input, dim = ctx.saved_values
        return argmax(input, dim) * grad_output


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    m = max(input, dim)
    return input - (input - m).exp().sum(dim).log() - m


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    out = tile(input, kernel)
    out = max(out, 4)
    out = out.view(batch, channel, out.shape[2], out.shape[3])
    return out


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError('Need to implement for Task 4.4')
    if ignore:
        return input

    rand_tensor = rand(input.shape, backend=input.backend)
    mask = rand_tensor > rate
    return mask * input
