from .base_operation import BaseOperation
import json
from functools import partial
from ...utilities.metadata.operation_metadata import OperationMetadata
import torch.nn.functional as F
from ...utilities.functional_utils.flops_estimation import calculate_pool_flops
import torch
from ...utilities.functional_utils.flops_estimation import calculate_linear_flops
from .base_operation import OperationType
import torch.nn as nn
from ...utilities.partial_wrapper import PartialWrapper

class AbstractRoleSpecificOperation(BaseOperation):
    """
    A virtual class encapsulating Role-Specific Operations (primarily input and
    output stems).

    The class is built to enhance readability, simply to distinguish between
    role-specific and generic operations.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        super(AbstractRoleSpecificOperation, self).__init__(in_shape,
                                                            out_shape,
                                                            **kwargs)




class InputStem(AbstractRoleSpecificOperation):
    """
    Sequence of operations used to process all inputs
    """

    OPERATION_TYPE = OperationType.ROLE_SPECIFIC_OP
    FILTER_COUNT = 64
    KERNEL_SIZE = 3
    STRIDE = 1
    PADDING = (KERNEL_SIZE - 1) // 2

    def __init__(self, in_shape, out_shape, **kwargs):
        """
        Args:
            in_sape (:class:`torch.Tensor`): the raw data input shape
            **kwargs (dict): Additional keyword arguments provided as a \
            dictionary (used to unify intialization calls in case additional \
            args are provided)
        """
        super(InputStem, self).__init__(in_shape, out_shape, **kwargs)

        # self.kernel_size = InputStem.KERNEL_SIZE

        # Stem layers
        # self.conv = nn.Conv2d(in_shape[1],
        #                       InputStem.FILTER_COUNT,
        #                       kernel_size=self.kernel_size,
        #                       stride=InputStem.STRIDE,
        #                       padding=InputStem.PADDING)

        # self.bn = nn.BatchNorm2d(InputStem.FILTER_COUNT)

        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.relu(out)
        # out = self.maxpool(out)

        return x

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # c_flops = calculate_conv_flops(in_shape=self.in_shape,
        #                             out_shape=self.out_shape,
        #                             kernel_size=self.kernel_size)

        # bn_flops = calculate_norm_flops(in_shape=self.in_shape)

        return 0 # bn_flops # + c_flops

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
            list: a 1-element list of partial functions with initialized valid \
            hyperparameters (in the form of a list to conform with primitives)
        """

        # out_shape = kernel_based_outshape(in_shape=in_shape,
        #                                   kernel_size=InputStem.KERNEL_SIZE,
        #                                   filter_count=InputStem.FILTER_COUNT,
        #                                   stride=InputStem.STRIDE,
        #                                   padding=InputStem.PADDING,
        #                                   dilation=1)

        return [PartialWrapper(InputStem,
                               in_shape=in_shape,
                               out_shape=in_shape)]

    @property
    def metadata(self):
        return OperationMetadata(op=type(self).__name__,
                                 id=self.id)


class OutputStem(AbstractRoleSpecificOperation):
    """
    Sequence of operations used as an output stem for image datasets
    """

    OPERATION_TYPE = OperationType.ROLE_SPECIFIC_OP
    INTERMEDIATE_LINEAR_SIZE = 128

    def __init__(self, in_shape, out_shape, num_classes, **kwargs):
        """
        Args:
            num_classes (int): the number of output classes
            **kwargs (dict): Additional keyword arguments provided as a \
            dictionary (used to unify intialization calls in case additional \
            args are provided)
        """
        super(OutputStem, self).__init__(in_shape,
                                         out_shape,
                                         **kwargs)

        self.num_classes = num_classes

        self.fc = nn.Linear(in_shape[1], OutputStem.INTERMEDIATE_LINEAR_SIZE)
        self.classifier = nn.Linear(OutputStem.INTERMEDIATE_LINEAR_SIZE,
                                    num_classes)


    def forward(self, x):
        """
        Perform a forward pass through the operation

        Args:
            x (:class:`torch.Tensor`): the operation's input data

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        out = torch.mean(x, dim=(2, 3), keepdim=True)   # Global Average Pooling
        out_cropped = out.view(out.size(0), -1)
        out1 = self.fc(out_cropped)
        out2 = self.classifier(out1)
        scout = F.log_softmax(out2, dim=-1)

        return scout


    def reshape(self, in_shape=None, num_classes=None):
        """
        Adapt the output stem to new class count (class-incremental learning)
        or reshapes the entire operation if the network completely changed.

        *This operation preserves weights if applicable.*

        Args:

        """

        in_shape_changed = in_shape is not None and in_shape != self.in_shape
        num_classes_changed = num_classes is not None and \
        num_classes != self.num_classes

        if in_shape is not None and in_shape != self.in_shape:
            # adjust the fully connected layer (if in_shape changed)
            old_fc_weights = self.fc.weight
            old_fc_bias = self.fc.bias
            new_fc = nn.Linear(in_shape[1],
                               OutputStem.INTERMEDIATE_LINEAR_SIZE)
            min_fs = min(in_shape[1], self.in_shape[1])

            with torch.no_grad():  # not tracked in the computation graph
                new_fc.weight[:, :min_fs].copy_(old_fc_weights[:, :min_fs])
                new_fc.bias[:min_fs].copy_(old_fc_bias[:min_fs])

            self.fc = new_fc
            self.in_shape = in_shape

        if num_classes_changed:
            # temp store input shape
            def_nc = num_classes or self.num_classes
            in_size = self.classifier.in_features

            classifier = nn.Linear(in_size, def_nc)

            # handle both increasing and decreasing number of classes
            n_cls = min(self.classifier.out_features, def_nc)

            # move portion of pretrained weights to new classifier
            with torch.no_grad():  # not tracked in the computation graph
                classifier.weight[:n_cls].copy_(self.classifier.weight[:n_cls])
                classifier.bias[:n_cls].copy_(self.classifier.bias[:n_cls])

            # update the attributes
            self.classifier = classifier
            self.num_classes = def_nc
            self.out_shape = (self.in_shape[0], def_nc)

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        intermediate_shape = (0, OutputStem.INTERMEDIATE_LINEAR_SIZE)
        l1_flops = calculate_linear_flops(in_shape=self.in_shape,
                                          out_shape=intermediate_shape)
        l2_flops = calculate_linear_flops(in_shape=intermediate_shape,
                                          out_shape=(0, self.num_classes))
        return l1_flops + l2_flops

    @staticmethod
    def get_partials(in_shape, num_classes):
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
            list: a 1-element list of partial functions with initialized valid \
            hyperparameters (in the form of a list to conform with primitives)
        """

        return [PartialWrapper(OutputStem,
                        in_shape=in_shape,
                        out_shape=(in_shape[0], num_classes),  # for uniformity,
                                                               # discarded in
                                                               # init
                        num_classes=num_classes)]

    @property
    def metadata(self):
        return OperationMetadata(op=type(self).__name__,
                                 id=self.id)


class AlignConcat1x1(nn.Module):
    """
    [DEPRECATED] - opted for a functional approach instead since this op should
    not have any trainable params.

    ACC1x1 is a custom multiplexing module used to aggregate multiple inputs.
    Aligns the inputs' resolutions by upsampling all inputs to the largest
    resolution (aligning through downsampling results in a loss of information),
    followed by channel-wise concatenation and a 1x1 convolution to
    set the given target channel dimension
    """

    def __init__(self, in_shapes, intermediary_shape, out_shape,
                 prepend_to, concat_channels_size=None, **kwargs):
        """
        Args:
            in_shapes (list): list of 4D :class:`torch.Tensor` input shapes
            prepend_to (:class:`functools.partial`): a partial \
            :class:`~BaseOperation` object that succeeds the \
            :class:`~AlignConcat1x1`.
            intermediary_shape (:class:`torch.Tensor`): the output shape of \
            the ACC module (input shape to the `prepend_to` operation)
            out_shape (:class:`torch.Tensor`): the operation's calculated \
            output shape (after the `prepend_to` operation)
            concat_channels_size (optional, int): the size of the concatenated \
            inputs. Must be given to initialize the conv1x1 in case it channel-\
            shape adjustment is needed. (align -> concat -> conv1x1 transforms \
            channel dimension from `concat_channels_size` to \
            `prepend_to.in_channels` -> `prepend_to` operation). If this \
            argument is not provided, the conv1x1 operation is discarded and \
            `prepend_to.in_channels` = the concatenated channel dimension.
            **kwargs (dict): Additional keyword arguments provided as a \
            dictionary (used to unify intialization calls in case additional \
            args are provided)
        """

        super(AlignConcat1x1, self).__init__()

        self.in_shapes = in_shapes
        self.intermediary_shape = intermediary_shape
        self.out_shape = out_shape

        if isinstance(prepend_to, partial):
            self.prepend_to = prepend_to()
        else:
            # pre-instantiated; serialized
            self.prepend_to = prepend_to

        self.id = self.prepend_to.id

        self.upsample = PartialWrapper(nn.Upsample,
                                mode='bilinear',
                                align_corners=False)
        if concat_channels_size:
            self.conv = nn.Conv2d(concat_channels_size,
                                  self.prepend_to.in_channels,
                                  kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]):
        """
        Perform a forward pass through the operation

        Args:
            inputs (list): the operation's input data vector of
            :class:`torch.Tensor` objects

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        # infer spatial dimensions
        max_height = max([in_tensor.shape[2] for in_tensor in inputs])
        max_width = max([in_tensor.shape[3] for in_tensor in inputs])

        # if not hasattr(self, 'out_channels'):
        #     self.out_channels = max([in_tensor[1] for in_tensor in inputs])

        # align resolutions
        aligned_inputs = [self.upsample(size=(max_height,
                                              max_width))(input_tensor) \
                          for input_tensor in inputs]
        self.inputs_count = len(inputs)
        # stack inputs w.r.t. the channels
        out = torch.cat(aligned_inputs, dim=1)

        if hasattr(self, 'conv'):
            out = self.conv(out)

        out = self.prepend_to(out)

        return out

    @property
    def op_name(self):
        """
        Operation name getter

        Returns:
            str: the operations name
        """

        return f'{self.prepend_to.op_name}\n({type(self).__name__})'

    @property
    def op_color(self):
        """
        Operation color getter

        Returns:
            str: the operation's visualization color
        """

        return f'{self.prepend_to.op_color}'

    @property
    def flops(self):
        """
        Calculates the FLOPs used for this operation

        Returns:
            :obj:`float`: the operation's FLOPs
        """

        # upsampling/interpolation is estimated through the number of inputs
        num_of_interpolation = 2
        if hasattr(self, 'inputs_count'):
            num_of_interpolation = self.inputs_count

        # as per torchprofile's implementation for bilinear interpolation
        # the flops_per_element is ~= 4
        # upsampling flops:
        ups_flops = calculate_pool_flops(out_shape=self.intermediary_shape,
                                         flops_per_element=4)

        return self.prepend_to.flops + ups_flops * num_of_interpolation

    @staticmethod
    def get_partials(in_shapes):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over
        :func:`~BaseOperation.get_all_partials` for faster performance (the
        cache uses minimal memory as these are uninitialized partials)

        Args:
            in_shapes (list): list of :class:`torch.Tensor` objects, each \
            corressponding to a 4-dim shape of an input feature map (NxCxHxW)

        Returns:
            list: a 1-element list of partial functions with initialized valid \
            hyperparameters (in the form of a list to conform with primitives)
        """

        intermediary_shape = (
            max([o[0] for o in in_shapes]),
            sum([o[1] for o in in_shapes]),
            max([o[2] for o in in_shapes]),
            max([o[3] for o in in_shapes])
        )

        return [PartialWrapper(AlignConcat1x1, in_shapes=in_shapes,
                        intermediary_shape=intermediary_shape)]


    @staticmethod
    def deserialize(hyperparams):
        """
        """
        prepend_to = json.loads(hyperparams['prepend_to'])
        prepend_cls = globals()[prepend_to['op']]
        prepend_hps = prepend_to['hyperparams']

        del hyperparams['prepend_to']

        return AlignConcat1x1(prepend_to=prepend_cls(**prepend_hps),
                              **hyperparams)



    def serialize(self):
        """
        """

        return json.dumps(self.metadata.params)

    @property
    def metadata(self):
        return OperationMetadata(op=type(self).__name__,
                                 id=self.id,
                                 in_shape=self.in_shape,
                                 out_shape=self.out_shape)


    def __str__(self):
        return str(self.metadata)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        """
        Hashing the metadata of the operation as that is enforced to be
        unique.
        """

        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.metadata == other.metadata


