from .metadata import Metadata

class OperationMetadata(Metadata):

    def __init__(self,
                 op, id,
                 in_shape=None,
                 out_shape=None,
                 is_partial=False,
                 **hyperparameters):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'op': op,
            'id': id
        }

        if in_shape is not None:
            params['in_shape'] = in_shape
        if out_shape is not None:
            params['out_shape'] = out_shape

        for k, v in hyperparameters.items():
            params[k] = v

        params['_is_partial'] = is_partial
        super(OperationMetadata, self).__init__(params=params)


    @staticmethod
    def init_from_partial(partial_wrapper):
        """
        Primarily used to init :class:`~SearchSpaceMetadata` as all Search
        Space operations are contained as :class:`~PartialWrapper` objects
        until sampled.
        """
        return OperationMetadata(op=partial_wrapper.func.__name__,
                                 id=None, in_shape=None, out_shape=None,
                                 is_partial=True,
                                 **(partial_wrapper.keywords))


    @property
    def signature(self):
        """
        Essentially the same as `__str__()`, but excluding the operation's UID.

        This is used to check if two operations have the same "signature", or
        properties and structure.

        Returns:
            :obj:`str`: the operation's signature
        """

        return ','.join([f'{k}={str(v)}' for k, v in self.params.items() \
                         if k != 'id'])


