from .metadata import Metadata

class SearchSpaceMetadata(Metadata):

    def __init__(self, type, num_vertices, encoding,
                 operations_metadata, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'type': type,
            'num_vertices': num_vertices,
            'encoding': encoding,
            'operations_metadata': operations_metadata
        }

        super(SearchSpaceMetadata, self).__init__(params=params, **kwargs)


