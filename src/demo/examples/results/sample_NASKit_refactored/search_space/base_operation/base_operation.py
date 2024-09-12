import json
from ...utilities.partial_wrapper import PartialWrapper
import abc
from ...utilities.metadata.operation_metadata import OperationMetadata
from enum import auto
import torch.nn as nn
from enum import Enum
import itertools

class BaseOperation(nn.Module, abc.ABC):
    """
    Abstract skeleton for operations on the edges of the \
    search space graph.

    Supported operations are primitive operations and compound primitives \
    (i.e. a combination of primitives as an individual building block).

    Hierarchical structures can be formed through recursive Network \
    objects, where the first stage is composed of `BaseOperation` objects.

    Each :class:`~BaseOperation` must have an `id` attribute
    """

    OPERATION_TYPE = None       # type of operation; spatial, reduction, etc.
    HYPERPARAMS = {}

    __ID_TRACKER = 0


    def __init__(self, in_shape, out_shape, **kwargs):
        # super().__init__()

        # explicit superclass init to ensure expected MRO
        nn.Module.__init__(self)
        abc.ABC.__init__(self)

        if 'id' in kwargs:
            assert type(kwargs['id']) == int, 'Invalid operation ID provided'
            self.id = kwargs['id']
            # update auto-ID
            BaseOperation.__ID_TRACKER = max(BaseOperation.__ID_TRACKER,
                                             self.id)
        else:
            self.id = BaseOperation.__ID_TRACKER
            BaseOperation.__ID_TRACKER += 1

        self.in_shape = in_shape
        self.out_shape = out_shape

        # The block below is now deprecated;
        # DFS is implemented to traverse the graph and find the nearest
        # previous `out_channels`
        # if kwargs and 'in_channels' in kwargs:
        #     # pass `out_channels` for operations that do not require
        #     # `in_channels`
        #     # e.g. ReLU, BatchNorm, etc.
        #     self.out_channels = kwargs['in_channels']


    @abc.abstractmethod
    def forward(self, x, edge_data):
        """
        The nn module forward operation

        Args:
            x (:class:`~torch.FloatTensor`): the input feature tensor
            edge_data (dict): addtional kwargs to be stored on the edge
        """
        raise NotImplementedError('Abstract method was not implemented')


    @property
    def op_name(self):
        """
        Operation hash function. This method embeds both the operation type
        and its hyperparameters into the name.

        This property is used for the hashing and equality of operations,
        should return a *unique* name.

        Since operations could repeat with the same exact hyperaparameters in an
        architecture, we use the index of the operation in the architecture to
        make it distinct

        Returns:
            str: the operation's inferred name
        """
        cls = self.__class__
        id = self.id if hasattr(self, 'id') else -1
        ret_name = f'(#{str(id)}) {cls.__name__}'

        if hasattr(cls, 'HYPERPARAMS'):
            for p, _ in cls.HYPERPARAMS.items():
                if hasattr(self, p):
                    hp_initials = ''.join([word[0].lower() \
                                           for word in p.split('_')])
                    ret_name += f'-{hp_initials}({getattr(self, p)})'

        return ret_name


    @property
    def op_color(self):
        """
        Operation color getter

        Returns:
            str: the operation's visualization color
        """
        if not hasattr(type(self), 'OPERATION_TYPE'):
            return '#34495e'    # asphalt (grey-blue)

        if type(self).OPERATION_TYPE == OperationType.SPATIAL_OP:
            return '#2ecc71'    # emerald (light green)
        elif type(self).OPERATION_TYPE == OperationType.REDUCTION_OP:
            return '#3498db'    # peter river (sky blue)
        elif type(self).OPERATION_TYPE == OperationType.NORMALIZATION_OP:
            return '#e74c3c'    # crayola red (bright red)
        elif type(self).OPERATION_TYPE == OperationType.ACTIVATION_OP:
            return '#f1c40f'    # sunflower (yellow)
        elif type(self).OPERATION_TYPE == OperationType.COMPOSITE_OP:
            return '#9b59b6'    # amethyst (light purple)
        elif type(self).OPERATION_TYPE == OperationType.ROLE_SPECIFIC_OP:
            return '#2c3e50'    # midnight blue (grey-dark blue)
        else:
            return '#34495e'    # asphalt (grey-blue)


    @classmethod
    # @lru_cache(maxsize=32)    # lru will not perform well given the random
                                # nature of the sampler
                                # we lazy-init/cache statically within the
                                # subclass instead
    def get_all_partials(cls):
        """
        Gets all combinations of hyperparameters for a given \
        :class:`~BaseOperation` subclass using its respective static \
        `HYPERPARAMETERS` attribute

        Returns:
            list: a list of PartialWrapper partial functions of the \
            :class:`~BaseOperation`'s subclass (cls) pre-initialized with \
            all combinations of hyperaparameters
        """

        if hasattr(cls, '__ALL_PARTIALS'):
            # cached
            return cls.__ALL_PARTIALS

        ret_list = []

        if hasattr(cls, 'HYPERPARAMS'):
            params_dict = cls.HYPERPARAMS
            param_keys = list(params_dict.keys())
            combs = itertools.product(*params_dict.values())

            for hyperparams in combs:
                # generate wrappers for every hyperparam comb
                hyperparam_dict = dict(zip(param_keys, hyperparams))
                ret_list.append(PartialWrapper(cls, **hyperparam_dict))
        else:
            # hyperparameter-less operation
            ret_list.append(PartialWrapper(cls))

        cls.__ALL_PARTIALS = ret_list

        return ret_list


    @classmethod
    def deserialize(cls, args):
        """
        """

        return cls(**args)


    def serialize(self):
        """
        """

        return json.dumps(self.metadata.params)


    @property
    def metadata(self):
        hyperparams = {}
        # operation-specific hyperparameters
        if hasattr(type(self), 'HYPERPARAMS'):
            params_dict = type(self).HYPERPARAMS
            for p, _ in params_dict.items():
                if hasattr(self, p):
                    hyperparams[p] = getattr(self, p)

        return OperationMetadata(op=type(self).__name__,
                                 id=self.id,
                                 in_shape=self.in_shape,
                                 out_shape=self.out_shape,
                                 **hyperparams)


    def decompile(self):
        """
        Decompiles the operation and returns a representative
        :class:`PartialWrapper`.
        """
        kwargs = {}
        # operation-specific hyperparameters
        if hasattr(type(self), 'HYPERPARAMS'):
            params_dict = type(self).HYPERPARAMS
            for p, _ in params_dict.items():
                if hasattr(self, p):
                    kwargs[p] = getattr(self, p)
        kwargs['id'] = self.id
        kwargs['in_shape'] = self.in_shape
        kwargs['out_shape'] = self.out_shape

        return PartialWrapper(type(self), **kwargs)


    @staticmethod
    def reset_auto_id():
        BaseOperation.__ID_TRACKER = 0


    @staticmethod
    def get_auto_id():
         return BaseOperation.__ID_TRACKER


    def __str__(self):
        md = self.metadata
        return f'{md.op}({str(md)})'


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


class OperationType(Enum):
    SPATIAL_OP          = auto()     # Feature extraction and dimension
                                     # preservation (conv, identity, etc.)
    REDUCTION_OP        = auto()     # Dimension reduction (max_pool, avg_pool,
                                     # strided conv, etc.)
    NORMALIZATION_OP    = auto()     # Transformations and regularization
                                     # (batch norm, group norm, etc.)
    ACTIVATION_OP       = auto()     # Non-linear functions
                                     # (ReLU, H-Swish, etc.)
    COMPOSITE_OP        = auto()     # Grouped building blocks (ReLUConvBN,
                                     # inception cell, hierarchical motif, etc.)
    ROLE_SPECIFIC_OP    = auto()     # Rule-based operations that cannot be
                                     # sampled (InputStem, OutputStem)


