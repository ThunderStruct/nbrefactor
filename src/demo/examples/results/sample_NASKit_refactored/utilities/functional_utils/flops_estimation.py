

def calculate_conv_flops(in_shape, out_shape,
                         kernel_size,
                         separable=False):
    """
    General formula to calculate MACs/FLOPs for convolutional operations

    Ref:
        Abstracted from http://bit.ly/411Uscw, but modified to account for \
        padding, dilation, etc., through inference from the output size

        Standard Conv:
        (2 * input_channels * (kernel_size^2) * output_height * output_width)
        * output_channels

        Separable Conv:
        (2 x input_channels * (kernel_size^2) * output_height * output_width)
        +
        (2 * input_channels * output_channels)

    Returns:
        :obj:`float`: calculated FLOPs for the given conv. layer
    """

    mac = in_shape[1] * (kernel_size**2) * out_shape[2] * out_shape[3]
    flops = 2 * mac

    if separable:
        # add point-wise terms, as opposed to multiplying by out_channels
        # (much less FLOPs)
        mac_depthwise = in_shape[1] * (kernel_size**2) \
        * out_shape[2] * out_shape[3]
        flops_depthwise = 2 * mac_depthwise

        mac_pointwise = in_shape[1] * out_shape[1] * out_shape[2] * out_shape[3]
        flops_pointwise = 2 * mac_pointwise

        flops = flops_depthwise + flops_pointwise
    else:
        # multiply by out_channels --> standard conv
        flops = flops * out_shape[1]

    return flops


def calculate_linear_flops(in_shape, out_shape):
    """
    General formula to calculate MACs/FLOPs for linear operations

    Ref:
        http://bit.ly/411Uscw

    Args:
        in_shape (:obj:`tuple`): input shape for the linear operation
        out_shape (:obj:`tuple`): output shape for the linear operation

    Returns:
        :obj:`float`: calculated FLOPs for the given linear layer
    """
    mac = in_shape[1] * out_shape[1]

    return 2 * mac


def calculate_pool_flops(out_shape, flops_per_element):
    """
    General formula to calculate MACs/FLOPs for pooling operations.

    Args:
        out_shape (:obj:`tuple`): output shape for the pooling operation
        flops_per_element (:obj:`float`): number of FLOPs for each sliding \
        window iteration (differs for each type of pooling operation). \
        i.e. for MaxPool, each iteration comprises of 1 comparison operation, \
        while AveragePool requires multiple additions + divisions.

    Returns:
        :obj:`float`: calculated FLOPs for the given pooling layer
    """

    return flops_per_element * out_shape[1] * out_shape[2] * out_shape[3]


def calculate_norm_flops(in_shape):
    """
    General formula to calculate MACs/FLOPs for normalization operations.
    This estimation assumes 8 FLOPs per element (mean, variance, scale, shift).

    Args:
        in_shape (:obj:`tuple`): input shape for the pooling operation

    Returns:
        :obj:`float`: calculated FLOPs for the given normalization layer
    """

    return 8 * in_shape[1] * in_shape[2] * in_shape[3]


