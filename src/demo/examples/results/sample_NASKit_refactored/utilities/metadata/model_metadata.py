from .metadata import Metadata

class ModelMetadata(Metadata):
    def __init__(self, id, version, wl_hash, model_hash, serialized_graph,
                 mflops, total_params, learnable_params, adj_matrix, nodes,
                 task_map, tasks_metadata, **kwargs):
        """
        Not simply getting `**kwargs` as the strongly-typed args can be more
        robustly handled in instantiation
        """
        params = {
            'id': id,
            'version': version,
            'wl_hash': wl_hash,
            'model_hash': model_hash,
            'serialized_graph': serialized_graph,
            'mflops': mflops,
            'total_params': total_params,
            'learnable_params': learnable_params,
            'adj_matrix': adj_matrix,
            'nodes': nodes,
            'task_map': task_map,
            'tasks_metadata': tasks_metadata
        }

        super(ModelMetadata, self).__init__(params=params, **kwargs)


