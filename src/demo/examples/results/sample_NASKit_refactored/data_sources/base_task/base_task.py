import abc
from ...utilities.metadata.task_metadata import TaskMetadata

class BaseTask(abc.ABC):

    def __init__(self, id, version, name, datasource, search_space,
                 scoring_func, importance_weight, objective, finetune_lr):
        self.id = id
        self.version = version
        self.name = name
        self.modality = self.__class__._MODALITY
        self.datasource = datasource
        self.search_space = search_space
        self.scoring_func = scoring_func
        self.importance_wight = importance_weight
        self.objective = objective
        self.finetune_lr = finetune_lr


    @abc.abstractmethod
    def loaders(self):
        raise NotImplementedError('Abstract method was not implemented')

    @property
    def metadata(self):
        return TaskMetadata(t_type=type(self).__name__,
                            id=self.id,
                            version=self.version,
                            name=self.name,
                            modality=self.modality,
                            objectives_metadata=self.objective.metadata_list,
                            search_space_metadata=self.search_space.metadata,
                            datasource_metadata=self.datasource.metadata)

    def save(self, dir='./tasks/', filename=None):

        def_filename = f'task_{self.id}v.{self.version}-{self.name}.json'

        self.metadata.save(dir=dir,
                           filename=filename if filename else def_filename)


    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.metadata)

    def __eq__(self, other):
        """
        Using __eq__ is much faster than comparing __hash__ values
        (dependent on __str__ values of datasource + search_space which
        potentially encapsulates a lot of operations)
        """
        if not isinstance(other, self.__class__):
            return False

        return self.metadata == other.metadata

    def __hash__(self):
        return hash(self.metadata)


