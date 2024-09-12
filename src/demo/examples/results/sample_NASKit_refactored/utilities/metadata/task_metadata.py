from .metadata import Metadata

class TaskMetadata(Metadata):
    def __init__(self, t_type, id, version, name, modality, objectives_metadata,
                 search_space_metadata, datasource_metadata, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'type': t_type,
            'id': id,
            'version': version,
            'name': name,
            'modality': modality,
            'objectives_metadata': objectives_metadata,
            'search_space_metadata': search_space_metadata,
            'datasource_metadata': datasource_metadata
        }

        super(TaskMetadata, self).__init__(params=params, **kwargs)


class VisionTaskMetadata(TaskMetadata):
    def __init__(self, t_type, id, version, name, modality, objectives_metadata,
                 search_space_metadata, datasource_metadata, train_batch_size,
                 val_batch_size, learning_rate, nas_epochs, candidate_epochs,
                 in_shape, out_shape, classes, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            't_type': t_type,
            'id': id,
            'version': version,
            'name': name,
            'modality': modality,
            'objectives_metadata': objectives_metadata,
            'search_space_metadata': search_space_metadata,
            'datasource_metadata': datasource_metadata,
            'train_batch_size': train_batch_size,
            'val_batch_size': val_batch_size,
            'learning_rate': learning_rate,
            'nas_epochs': nas_epochs,
            'candidate_epochs': candidate_epochs,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'classes': classes
        }

        super(VisionTaskMetadata, self).__init__(**params, **kwargs)


