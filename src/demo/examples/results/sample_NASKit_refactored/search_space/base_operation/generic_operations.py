from ...utilities.functional_utils.module_utils import kernel_based_validation
from ...utilities.functional_utils.flops_estimation import calculate_conv_flops
from .base_operation import BaseOperation
from copy import deepcopy
from ...utilities.logger import Logger
from ...utilities.functional_utils.module_utils import kernel_based_outshape
from ...utilities.functional_utils.flops_estimation import calculate_norm_flops
import torch.nn.functional as F
from ...utilities.functional_utils.flops_estimation import calculate_pool_flops
import torch
from .base_operation import OperationType
import torch.nn as nn

class AbstractGenericOperation(BaseOperation):
    """
    Purely virtual class used to distinguish between the generic, sample-able
    operations and role-specific operations (primarily input/output stems).
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        super(AbstractGenericOperation, self).__init__(in_shape,
                                                       out_shape,
                                                       **kwargs)




#=================================================================
# SPATIAL OPS
#=================================================================


class Conv2d(BaseOperation):

    OPERATION_TYPE = OperationType.SPATIAL_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9, 11]
    }

    def __init__(self, in_shape, out_shape, filter_count,
                 kernel_size, **kwargs):
        super(Conv2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.padding = (kernel_size - 1) // 2   # zero-padded to maintain input
                                                # resolution
        self.conv = nn.Conv2d(in_shape[1],
                              filter_count,
                              kernel_size,
                              stride=1,
                              padding=self.padding)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.conv(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(Conv2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=1,
                                              padding='same',
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['stride'] = 1
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 'same'

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class SepConv2d(BaseOperation):

    OPERATION_TYPE = OperationType.SPATIAL_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9, 11]
    }

    def __init__(self, in_shape, out_shape, filter_count, kernel_size,
                 **kwargs):
        super(SepConv2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.padding = (kernel_size - 1) // 2   # zero-padded to maintain input
                                                # resolution
        self.depthwise = nn.Conv2d(in_shape[1],
                                   in_shape[1],
                                   kernel_size,
                                   stride=1,
                                   padding=self.padding,
                                   groups=in_shape[1])
        self.pointwise = nn.Conv2d(in_shape[1], filter_count, 1)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size,
                                    separable=True)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over
        :func:`~BaseOperation.get_all_partials` for faster performance (the
        cache uses minimal memory as these are uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(SepConv2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=1,
                                              padding='same',
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['stride'] = 1
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 'same'

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class DilatedConv2d(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.SPATIAL_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9, 11],
        'dilation': [2, 3]
    }

    def __init__(self, in_shape, out_shape, filter_count, kernel_size,
                 dilation, **kwargs):
        super(DilatedConv2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.padding = (kernel_size - 1) // 2  # zero-padded to maintain input
                                               # resolution
        self.dilation = dilation
        self.conv = nn.Conv2d(in_shape[1],
                              filter_count,
                              kernel_size,
                              stride=1,
                              padding=self.padding,
                              dilation=dilation)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.conv(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(DilatedConv2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            dilation = p_op.keywords['dilation']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=1,
                                              padding='same',
                                              dilation=dilation)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['stride'] = 1
            partials[idx].keywords['padding'] = 'same'

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class Identity(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.SPATIAL_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(Identity, self).__init__(in_shape, out_shape, **kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return 0.0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(Identity.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials

#=================================================================
# REDUCTION OPS
#=================================================================


class MaxPool2d(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'kernel_size': [2, 3],
        'stride': [1, 2],
        'padding': [0, 1, 2]
    }

    def __init__(self, in_shape, out_shape, kernel_size, stride, padding,
                 **kwargs):
        super(MaxPool2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.maxpool(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_pool_flops(out_shape=self.out_shape,
                                    flops_per_element=1)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(MaxPool2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=in_shape[1],
                                              stride=p_op.keywords['stride'],
                                              padding=p_op.keywords['padding'],
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['filter_count'] = in_shape[1]

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class AvgPool2d(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'kernel_size': [2, 3],
        'stride': [1, 2],
        'padding': [0, 1, 2]
    }

    def __init__(self, in_shape, out_shape, kernel_size, stride, padding,
                 **kwargs):
        super(AvgPool2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.avgpool(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # additions
        fpe = (self.kernel_size**2 - 1) * self.in_shape[1] * \
        self.out_shape[2] * self.out_shape[3]

        # divisions are accounted for below
        return calculate_pool_flops(out_shape=self.out_shape,
                                    flops_per_element=fpe)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(AvgPool2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=in_shape[1],
                                              stride=p_op.keywords['stride'],
                                              padding=p_op.keywords['padding'],
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['filter_count'] = in_shape[1]
            partials[idx].keywords['dilation'] = 1

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class GlobalAvgPool2d(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(GlobalAvgPool2d, self).__init__(in_shape, out_shape, **kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return torch.mean(x, dim=(2, 3), keepdim=True)


    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # we use the input shape below instead of the output shape since this
        # is not a regular sliding window operation (padding-inference is not
        # required). We add +1 to account for the single div. FLOP
        return calculate_pool_flops(out_shape=self.in_shape,
                                    flope_per_element=1) + 1

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(GlobalAvgPool2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            out_shape = (*in_shape[:2], 1, 1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape

        return partials


class TransformChannels(AbstractGenericOperation):
    """
    1x1 Conv used to transform the channel dimension
    """

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64]
    }

    def __init__(self, in_shape, out_shape, filter_count, **kwargs):
        super(TransformChannels, self).__init__(in_shape, out_shape,
                                                **kwargs)

        self.filter_count = filter_count
        self.conv = nn.Conv2d(in_shape[1],
                              filter_count,
                              1,
                              stride=1,
                              padding=0)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.conv(x)


    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # this is a 1x1 convolution operation
        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=1)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(TransformChannels.get_all_partials())

        for idx, p_op in enumerate(partials):
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=1,
                                              filter_count=fc,
                                              stride=1,
                                              padding=0,
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['kernel_size'] = 1
            partials[idx].keywords['stride'] = 1
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 0

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class ReduceResolution(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'scale_factor': [2, 3]
    }

    def __init__(self, in_shape, out_shape, scale_factor, **kwargs):
        super(ReduceResolution, self).__init__(in_shape, out_shape,
                                               **kwargs)

        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return nn.functional.interpolate(x, scale_factor=1 / self.scale_factor,
                                         mode='bilinear', align_corners=False)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # as per torchprofile's implementation for bilinear interpolation
        # the flops_per_element is ~= 4
        return calculate_pool_flops(out_shape=self.in_shape,
                                    flops_per_element=4)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(ReduceResolution.get_all_partials())

        for idx, p_op in enumerate(partials):

            out_shape = (
                in_shape[0],
                in_shape[1],
                in_shape[2] // p_op.keywords['scale_factor'],
                in_shape[3] // p_op.keywords['scale_factor']
            )
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape

        return partials


class StridedConv2d(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9],
        'stride': [2, 3]
    }

    def __init__(self, in_shape, out_shape, filter_count, kernel_size,
                 stride, **kwargs):
        super(StridedConv2d, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.stride = stride
        self.conv = nn.Conv2d(in_shape[1],
                              filter_count,
                              kernel_size,
                              stride=stride,
                              padding=0)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.conv(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(StridedConv2d.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=p_op.keywords['stride'],
                                              padding=0,
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 0

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class StridedSepConv(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.REDUCTION_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9],
        'stride': [2, 3]
    }

    def __init__(self, in_shape, out_shape, filter_count, kernel_size,
                 stride, **kwargs):
        super(StridedSepConv, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.stride = stride
        self.depthwise = nn.Conv2d(in_shape[1],
                                   in_shape[1],
                                   kernel_size,
                                   stride=stride,
                                   padding=0,
                                   groups=in_shape[1])
        self.pointwise = nn.Conv2d(in_shape[1], filter_count, 1)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size,
                                    separable=True)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(StridedSepConv.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=p_op.keywords['stride'],
                                              padding=0,
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 0

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


#=================================================================
# NORMALIZATION OPS
#=================================================================


class BatchNormalization(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.NORMALIZATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(BatchNormalization, self).__init__(in_shape,
                                                 out_shape, **kwargs)

        self.batch_norm = nn.BatchNorm2d(in_shape[1])  # channel-wise batch norm

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.batch_norm(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_norm_flops(in_shape=self.in_shape)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(BatchNormalization.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class LayerNormalization(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.NORMALIZATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(LayerNormalization, self).__init__(in_shape,
                                                 out_shape, **kwargs)

        self.layer_norm = nn.LayerNorm(in_shape[1:])

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.layer_norm(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_norm_flops(in_shape=self.in_shape)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(LayerNormalization.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class GroupNormalization(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.NORMALIZATION_OP
    HYPERPARAMS = {
        'num_groups': [1, 4, 8, 16]
    }

    def __init__(self, in_shape, out_shape, num_groups, **kwargs):
        super(GroupNormalization, self).__init__(in_shape, out_shape,
                                                 **kwargs)

        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(num_groups, in_shape[1])

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.group_norm(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_norm_flops(in_shape=self.in_shape)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(GroupNormalization.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        # validate operation (in_shape[1] must be divisible by num_groups)
        partials = list(filter(lambda p: in_shape[1] % \
                               p.keywords['num_groups'] == 0, partials))


        return partials



class InstanceNormalization(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.NORMALIZATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(InstanceNormalization, self).__init__(in_shape,
                                                    out_shape, **kwargs)

        self.instance_norm = nn.InstanceNorm2d(in_shape[1])

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.instance_norm(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_norm_flops(in_shape=self.in_shape)

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(InstanceNormalization.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class Dropout(AbstractGenericOperation):
    OPERATION_TYPE = OperationType.NORMALIZATION_OP
    HYPERPARAMS = {
        'p': [0.25, 0.5]
    }

    def __init__(self, in_shape, out_shape, p, **kwargs):
        super(Dropout, self).__init__(in_shape, out_shape, **kwargs)

        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        # if self.training:
        #     mask = torch.bernoulli(torch.full(x.size(), 1 - self.p))
        #     x = x * mask / (1 - self.p)

        return self.dropout(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # dropout is typically negligible when calculating FLOPS (according to
        # torchprofile implementation)
        return 0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(Dropout.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


#=================================================================
# ACTIVATION OPS
#=================================================================


class ReLU(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.ACTIVATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(ReLU, self).__init__(in_shape, out_shape, **kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return nn.functional.relu(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """
        # activation functions are typically negligible when calculating FLOPS
        # (according to torchprofile implementation)
        return 0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(ReLU.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class Swish(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.ACTIVATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(Swish, self).__init__(in_shape, out_shape, **kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return x * torch.sigmoid(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """
        # activation functions are typically negligible when calculating FLOPS
        # (according to torchprofile implementation)
        return 0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(Swish.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class HSwish(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.ACTIVATION_OP

    def __init__(self, in_shape, out_shape, **kwargs):
        super(HSwish, self).__init__(in_shape, out_shape, **kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return x * F.relu6(x + 3) / 6

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """
        # activation functions are typically negligible when calculating FLOPS
        # (according to torchprofile implementation)
        return 0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(HSwish.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


class LeakyReLU(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.ACTIVATION_OP
    # inplace updates seem to be bugged
    # (https://github.com/pytorch/pytorch/issues/104943)
    # HYPERPARAMS = {
    #     'learnable': [True, False]
    # }

    def __init__(self, in_shape, out_shape, learnable=False,
                 negative_slope=0.01, **kwargs):
        super(LeakyReLU, self).__init__(in_shape, out_shape, **kwargs)

        self.learnable = learnable
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
                                       #, inplace=learnable)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        return self.leaky_relu(x)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """
        # activation functions are typically negligible when calculating FLOPS
        # (according to torchprofile implementation)
        return 0

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(LeakyReLU.get_all_partials())

        for idx, p_op in enumerate(partials):
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = in_shape

        return partials


# class ParametricReLU(AbstractGenericOperation):
#     def __init__(self, num_parameters=1, **kwargs):
#         super(ParametricReLU, self).__init__()
#         self.prelu = nn.PReLU(num_parameters=num_parameters)

#     def forward(self, x):
#         return self.prelu(x)


#=================================================================
# COMPOSITE OPS
#=================================================================


class ActivatedConv(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.COMPOSITE_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5, 7, 9],
        'stride': [1, 2, 3],
        'dilation': [1, 2, 3],
        'activation': [ReLU, Swish, HSwish, LeakyReLU],
        'separable': [True, False]
    }

    def __init__(self, in_shape, out_shape, filter_count,
                 kernel_size, activation, stride, dilation, separable,
                 **kwargs):
        super(ActivatedConv, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.padding = (kernel_size - 1) // 2
        self.separable = separable
        self.stride = stride
        self.dilation = dilation

        if separable:
            self.depthwise = nn.Conv2d(in_shape[1],
                                       in_shape[1],
                                       kernel_size,
                                       padding=self.padding,
                                       groups=in_shape[1])
            self.pointwise = nn.Conv2d(in_shape[1], filter_count, 1)
        else:
            self.conv = nn.Conv2d(in_shape[1],
                                  filter_count,
                                  kernel_size,
                                  stride=stride,
                                  padding=self.padding,
                                  dilation=dilation)

        self.activation = activation(id=-1,
                                     in_shape=out_shape, out_shape=out_shape)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        if separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv()

        x = self.activation(x)

        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        return calculate_conv_flops(in_shape=self.in_shape,
                                    out_shape=self.out_shape,
                                    kernel_size=self.kernel_size,
                                    separable=self.separable)



    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(ActivatedConv.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=1,
                                              padding='same',
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['padding'] = 'same'

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class ConvBnReLUBlock(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.COMPOSITE_OP
    HYPERPARAMS = {
        'filter_count': [8, 16, 32, 64, 128],
        'kernel_size': [1, 3, 5]
    }

    def __init__(self, in_shape, out_shape, filter_count,
                 kernel_size, **kwargs):
        super(ConvBnReLUBlock, self).__init__(in_shape, out_shape, **kwargs)

        self.kernel_size = kernel_size
        self.filter_count = filter_count
        self.padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_shape[1],
                              filter_count,
                              kernel_size,
                              stride=1,
                              padding=self.padding)
        self.bn = nn.BatchNorm2d(filter_count)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        c_flops = calculate_conv_flops(in_shape=self.in_shape,
                                       out_shape=self.out_shape,
                                       kernel_size=self.kernel_size)
        bn_flops = calculate_norm_flops(in_shape=self.in_shape)

        return c_flops + bn_flops

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(ConvBnReLUBlock.get_all_partials())

        for idx, p_op in enumerate(partials):
            k_size = p_op.keywords['kernel_size']
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=k_size,
                                              filter_count=fc,
                                              stride=1,
                                              padding='same',
                                              dilation=1)

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape
            partials[idx].keywords['stride'] = 1
            partials[idx].keywords['dilation'] = 1
            partials[idx].keywords['padding'] = 'same'

        # validate operation
        partials = list(filter(lambda p: kernel_based_validation(**p.keywords),
                               partials))

        return partials


class ResidualBlock(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.COMPOSITE_OP
    HYPERPARAMS = {
        'filter_count': [ 16, 32, 64 ]
    }

    def __init__(self, in_shape, out_shape, filter_count,
                 stride=1, **kwargs):
        super(ResidualBlock, self).__init__(in_shape, out_shape, **kwargs)

        self.stride = stride
        self.filter_count = filter_count

        self.conv1 = nn.Conv2d(in_shape[1], filter_count, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filter_count, filter_count, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_count)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_shape[1], filter_count, kernel_size=1, stride=stride),
            nn.BatchNorm2d(filter_count)
        )

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        c1_flops = calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.in_shape, # padding:same
                                        kernel_size=3)
        bn_flops = calculate_norm_flops(in_shape=self.in_shape)

        c2_flops = calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.out_shape,
                                        kernel_size=3)


        ds_flops = calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.out_shape,
                                        kernel_size=1)

        return c1_flops + bn_flops * 3 + c2_flops + ds_flops

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(ResidualBlock.get_all_partials())

        for idx, p_op in enumerate(partials):
            fc = p_op.keywords['filter_count']
            out_shape = kernel_based_outshape(in_shape=in_shape,
                                              kernel_size=3,
                                              filter_count=fc,
                                              stride=1,
                                              padding=1,
                                              dilation=1)
            out_shape = kernel_based_outshape(in_shape=out_shape,
                                              kernel_size=3,
                                              filter_count=fc,
                                              stride=1,
                                              padding=1,
                                              dilation=1)
            out_shape = (
                out_shape[0],
                max(out_shape[1], in_shape[1]),
                out_shape[2],
                out_shape[3]
            )       # all dimensions remain the same,
                    # the larger of the 2 channels (residual block / identity)
                    # is used

            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape

        return partials


class InceptionBlock(AbstractGenericOperation):

    OPERATION_TYPE = OperationType.COMPOSITE_OP
    HYPERPARAMS = {
        'filter_count_1': [ 16, 32 ],
        'filter_count_3': [ 16, 32, 64 ],
        'filter_count_5': [ 32, 64 ],
        'filter_count': [ 32, 64 ]
    }

    def __init__(self, in_shape, out_shape, filter_count_1, filter_count_3,
                 filter_count_5, filter_count, **kwargs):
        super(InceptionBlock, self).__init__(in_shape, out_shape, **kwargs)

        self.filter_count = filter_count
        self.filter_count_1 = filter_count_1
        self.filter_count_3 = filter_count_3
        self.filter_count_5 = filter_count_5

        self.branch1 = nn.Conv2d(in_shape[1], filter_count_1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_shape[1], filter_count_3, kernel_size=1),
            nn.Conv2d(filter_count_3, filter_count_3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_shape[1], filter_count_5, kernel_size=1),
            nn.Conv2d(filter_count_5, filter_count_5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_shape[1], filter_count, kernel_size=1)
        )

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)

        return out

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        c_flops = calculate_conv_flops(in_shape=self.in_shape,
                                       out_shape=self.in_shape,
                                       kernel_size=1)
        c_flops += calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.in_shape,
                                        kernel_size=1)
        c_flops += calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.in_shape,
                                        kernel_size=3)
        c_flops += calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.in_shape,
                                        kernel_size=1)
        c_flops += calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.in_shape,
                                        kernel_size=5)
        c_flops += calculate_conv_flops(in_shape=self.in_shape,
                                        out_shape=self.out_shape,
                                        kernel_size=1)
        m_flops = calculate_pool_flops(out_shape=self.out_shape,
                                       flops_per_element=1)

        return c_flops + m_flops

    @staticmethod
    def get_partials(in_shape):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over :func:`~BaseOperation.get_all_partials`
        for faster performance (the cache uses minimal memory as these are
        uninitialized partials)

        Args:
            in_shape (:class:`torch.Tensor`) the 4-dim shape of the input \
            feature map (NxCxHxW)
        Returns:
            list: a list of partial functions with initialized valid \
            hyperparameters
        """
        # deepcopy the memoized partials list
        partials = deepcopy(InceptionBlock.get_all_partials())

        for idx, p_op in enumerate(partials):

            countof = lambda kw: p_op.keywords[kw]
            out_shape = (
                in_shape[0],
                sum([countof('filter_count_1'),  # channel-wise concat
                    countof('filter_count_3'),
                    countof('filter_count_5'),
                    countof('filter_count')]),
                in_shape[2],
                in_shape[3]
            )
            partials[idx].keywords['in_shape'] = in_shape
            partials[idx].keywords['out_shape'] = out_shape

        return partials


# ------------------------------------------------------------------------------
# Print the diversity achieved by the hyperparameters' combinations
all_operations = AbstractGenericOperation.__subclasses__()
all_partials = [op.get_partials((64, 16, 128, 128)) for op in all_operations]
all_partials = [op for ops in all_partials for op in ops]

Logger.debug((
    f'{len(all_operations)} primitives; diversified into '
    f'{len(all_partials)} parameterized operations'
), line=False, caller=False)

