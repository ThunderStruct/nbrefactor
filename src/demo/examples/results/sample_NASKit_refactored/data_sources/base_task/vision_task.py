from ...scoring_functions import default_img_classification_scoring
from torch.utils.data import DataLoader
from ..base_optimization_objective.image_classification_objective import ICObjective
from ..vision_datasource import VisionDataSource
from .base_task import BaseTask
from ...utilities.metadata.task_metadata import VisionTaskMetadata

class VisionTask(BaseTask):
    """
    Data structure containing all task-related attributes, including the
    dataset and the search space. This allows for task-specific search spaces
    """

    _MODALITY = 'Image Classification'

    def __init__(self, id, version, name, datasource, search_space,
                 train_batch_size=128, val_batch_size=128,
                 nas_epochs=10, candidate_epochs=10, callbacks=[],
                 learning_rate=0.001, finetune_lr=0.0003, scoring_func=None,
                 importance_weight=1.0, objective=None):
        """
        Args:
            dataset (:class:`~VisionDataSource`): the dataset used to evaluate \
            the candidates
            root_path (:obj:`str`): the source path of data. \
            If this argument is not provided for other data sources, they are \
            downloaded through :class:`torchvision`
        """

        assert isinstance(datasource, VisionDataSource), (
            'Invalid dataset provided. Please ensure the data source is of '
            'type `VisionDataSource`'
        )

        assert datasource.train_data is not None, (
            'Dataset was not loaded. Please load the Data prior to '
            'initializing the task object'
        )

        # defaults
        sc_func = scoring_func or default_img_classification_scoring
        obj = objective
        if objective is None:
            obj = ICObjective()
            # default criterion
            obj.add_criterion(metric=ICObjective.Metric.VAL_ACC,
                              thresholds_enabled=False)

        super(VisionTask, self).__init__(id=id,
                                         version=version,
                                         name=name,
                                         datasource=datasource,
                                         search_space=search_space,
                                         scoring_func=sc_func,
                                         importance_weight=importance_weight,
                                         objective=obj,
                                         finetune_lr=finetune_lr)

        self.datasource = datasource
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.nas_epochs = nas_epochs
        self.candidate_epochs = candidate_epochs
        self.callbacks = callbacks
        self.learning_rate = learning_rate

        self.classes = self.datasource.val_data.classes
        self.num_classes = len(self.classes)



    # def __filter_classes(self, train_data, val_data, ratio):
    #     num_classes = len(set(train_data.targets))
    #     filter_count = int(num_classes * ratio)
    #     selected_indices = torch.randperm(num_classes)[:filter_count]

    #     # filter train/val subsets
    #     train_indices = []
    #     for i, label in enumerate(train_data.targets):
    #         if label in selected_indices:
    #             train_indices.append(i)
    #     train_data = Subset(train_data, train_indices)   # in-place

    #     val_indices = []
    #     for i, label in enumerate(val_data.targets):
    #         if label in selected_indices:
    #             val_indices.append(i)
    #     val_data = Subset(val_data, val_indices)         # in-place


    # def normalize_data(self):
    #     """

    #     TODO: I'm not a big fan of re-initializing the entire dataset.
    #     Computing the mean/std requires the intialization of a `DataLoader`
    #     with the given transforms (as some of which may be resizing the inputs
    #     which affects the mean/std), looping through the dataset to calculate
    #     the expectation/variance -> std, then using the computed values to
    #     re-init the dataset with the normalization transform.
    #     Look into ways to compute the mean and std with less overhead and
    #     without knowing the values a priori (for domain-agnostic purposes).

    #     """
    #     assert hasattr(self, 'mean') and hasattr(self, 'std'), (
    #         '`mean` and `std` were not computed! '
    #         'Ensure that `__compute_data_attributes` is called'
    #     )

    #     self.transforms = transforms.Compose([self.train_data.transform,
    #                                           transforms.Normalize(
    #                                               mean=self.mean,
    #                                               std=self.std
    #                                           )])

    #     Logger.info('Reinitializing data with normalization transforms...')

    #     # re-init dataset with new transforms
    #     self.__init_dataset()


    def loaders(self):
        train_loader = DataLoader(self.datasource.train_data,
                                  batch_size=self.train_batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.datasource.val_data,
                                batch_size=self.val_batch_size,
                                shuffle=False)

        return train_loader, val_loader


    ## @staticmethod
    # def class_segmentation_factory(task, segment_size):
    #     """
    #     Factory method to segment a given task into multiple class-wise
    #     tasks. Although the resulting segments are distinct `Task` objects,
    #     they all share the same name/id.
    #     """

    #     if task.num_classes % segment_size != 0:
    #         Logger.warning((
    #             'Number of classes is not divisible by the provided ',
    #             'segmentation count. Flooring the value... ',
    #             f'{task.num_classes % segment_size} classes will not be ',
    #             'considered!\n'
    #         ))

    #     seg_range = range(task.num_classes // segment_size)

    #     train_data, val_data = task.train_data, task.val_data

    #     # temp delete loaded dataset to avoid deep-copying them
    #     del task.train_data
    #     del task.val_data

    #     task_list = []
    #     inc_id = task.id + 1
    #     for segment_idx in seg_range:
    #         # train_classes only include the segment's classes
    #         train_classes = list(range(segment_idx * segment_size,
    #                                    (segment_idx + 1) * segment_size))
    #         # val_classes include all segments' classes up to segment_idx
    #         val_classes = list(range((segment_idx + 1) * segment_size))

    #         seg_task = deepcopy(task)

    #         seg_task.num_classes = val_classes
    #         seg_task.id = inc_id
    #         inc_id += 1

    #         seg_task.train_data = Subset(train_data,
    #                 [idx for idx in range(len(train_data)) \
    #                 if train_data.targets[idx] in train_classes]
    #         )

    #         seg_task.val_data = Subset(val_data,
    #                 [idx for idx in range(len(val_data)) \
    #                 if val_data.targets[idx] in val_classes]
    #         )

    #         task_list.append(seg_task)

    #     del train_data
    #     del val_data
    #     gc.collect()

    #     return task_list

    @property
    def shapes(self):
        """
        Gets the input shape (`tuple(batch_size, channels, height, width)`) and
        the output shape (`tuple(batch_size, num_classes)`)

        Returns:
            tuple: ( `(int:batch_size, int:channels, int:height, int:width),
                      (int:batch_size, int:num_classes)` )
        """

        def get_spatial_dimensions():
            # [Deprecated]: transforms are now applied during initialization
            # if hasattr(self, 'transforms') and self.transforms is not None:
            #     for t in self.transforms.transforms:
            #         if isinstance(t, transforms.Resize):
            #             return (self.channels, *t.size)

            # self.train_data is a list of tuples (tensor, label int)
            return self.datasource.train_data[0][0].size()

        return (
            (self.train_batch_size, *(get_spatial_dimensions())),
            (self.train_batch_size, self.num_classes)
         )

    ## @property
    # def distributions(self):
    #     """
    #     """
    #     assert hasattr(self, 'train_dist') and hasattr(self, 'val_dist'), (
    #         '`__compute_data_attributes()` needs to be called to initialize '
    #         '`self.train_dist` and `self.val_dist`'
    #     )

    #     return (self.train_dist, self.val_dist)

    @property
    def metadata(self):
        ss_metadata = self.search_space.metadata
        ds_metadata = self.datasource.metadata
        obj_metadata = self.objective.metadata_list
        shapes = self.shapes
        return VisionTaskMetadata(t_type=type(self).__name__,
                                  id=self.id,
                                  version=self.version,
                                  name=self.name,
                                  modality=self.modality,
                                  objectives_metadata=obj_metadata,
                                  search_space_metadata=ss_metadata,
                                  datasource_metadata=ds_metadata,
                                  train_batch_size=self.train_batch_size,
                                  val_batch_size=self.val_batch_size,
                                  learning_rate=self.learning_rate,
                                  nas_epochs=self.nas_epochs,
                                  candidate_epochs=self.candidate_epochs,
                                  in_shape=shapes[0],
                                  out_shape=shapes[1],
                                  classes=self.classes)

    def __str__(self):
        return str(self.metadata)

    def __eq__(self, other):
        """
        Using __eq__ is much faster than comparing __hash__ values
        (dependent on __str__ values of datasource + search_space which
        potentially encapsulates a lot of operations)
        """
        if not isinstance(other, VisionTask):
            return False

        return self.metadata == other.metadata



