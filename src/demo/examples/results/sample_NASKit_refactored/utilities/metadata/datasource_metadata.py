from .metadata import Metadata

class DataSourceMetadata(Metadata):
    def __init__(self, path, segment_size, segment_idx,
                 num_workers, transforms, dataset, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'path': path,
            'segment_size': segment_size,
            'segment_idx': segment_idx,
            'num_workers': num_workers,
            'transforms': transforms,
            'dataset': dataset
        }

        super(DataSourceMetadata, self).__init__(params=params, **kwargs)


