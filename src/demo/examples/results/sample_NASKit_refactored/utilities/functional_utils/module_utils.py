


# CONVOLUTION-RELATED UTILITIES

def __init_kernel_based_op_params(kernel_size, stride, padding, dilation):

    if (type(padding) is str and padding != 'same'):
        raise ValueError('Invalid padding value')

    k = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
    s = (stride, stride) if type(stride) is int else stride
    d = (dilation, dilation) if type(dilation) is int else dilation

    p = padding
    if type(p) is int:
        p = (p, p)
    elif type(p) is tuple:
        pass  # tuple padding is already in the right format
    elif type(p) is str:
        if p == 'same':
            # calculate 'same' padding based on the kernel size and dilation
            p = (((k[0] - 1) * d[0] + 1) // 2, ((k[1] - 1) * d[1] + 1) // 2)

    return k, s, d, p


def kernel_based_outshape(in_shape, kernel_size,
                          filter_count, stride, padding, dilation, **kwargs):
    """
    Calculate the output shape of any kernel-based operation (convolution,
    pooling, etc.)

    Args:
        in_shape (:class:`torch.Tensor`): the input feature map's shape \
        (NxCxHxW)
        kernel_size (int / tuple): the operation's filter size \
        (accepts `int` (`kernel_size` x `kernel_size`) or tuple (HxW))
        filter_count (int): the number of filters in the operation
        stride (int / tuple): the operation's filter stride \
        (accepts `int` or tuple (HxW))
        padding (int / tuple / str): the output map's padding (accepts `int` \
        or `tuple` values (HxW), and `'same'`, which pads the output map to \
        maintain its input resolution) dilation (int / tuple): the operation's \
        filter dilation value (accepts `int` or tuple (HxW))
    """
    in_height, in_width = in_shape[2:]

    k, s, d, p = __init_kernel_based_op_params(kernel_size, stride,
                                               padding, dilation)

    dilated_kernel_height = k[0] + (k[0] - 1) * (d[0] - 1)
    dilated_kernel_width = k[1] + (k[1] - 1) * (d[1] - 1)

    # conv output dimensions
    out_height = ((in_height + 2 * p[0] - dilated_kernel_height) // s[0] + 1)
    out_width = ((in_width + 2 * p[1] - dilated_kernel_width) // s[1] + 1)

    return (in_shape[0], filter_count, out_height, out_width)


def kernel_based_validation(in_shape, out_shape, kernel_size,
                            filter_count, stride, padding, dilation, **kwargs):
    """
    Validates the hyperparameters of a kernel-based operation given the input
    dimensions and intrinsic constraints (`padding` <= `kernel_size` / 2)
    Args:
        in_shape (:class:`torch.Tensor`): the input feature map's shape \
        (NxCxHxW)
        kernel_size (int / tuple): the operation's filter size \
        (accepts `int` (`kernel_size` x `kernel_size`) or tuple (HxW))
        filter_count (int): the number of filters in the operation
        stride (int / tuple): the operation's filter stride \
        (accepts `int` or tuple (HxW))
        padding (int / tuple / str): the output map's padding (accepts `int` \
        or `tuple` values (HxW), and `'same'`, which pads the output map to \
        maintain its input resolution) dilation (int / tuple): the operation's \
        filter dilation value (accepts `int` or tuple (HxW))
    """
    assert len(in_shape) == 4 and len(out_shape) == 4, (
        'Invalid input/output shape provided'
    )

    k, s, d, p = __init_kernel_based_op_params(kernel_size, stride,
                                               padding, dilation)

    # kernel size smaller than input shape
    if k[0] > in_shape[2] or k[1] > in_shape[3]:
        return False

    # padding is at most half the kernel size
    if p[0] > k[0] // 2 or p[1] > k[1] // 2:
        return False

    # stride and filter count > 0
    if s[0] <= 0 or s[1] <= 0 or filter_count <= 0:
        return False

    # valid
    return True

