from ..search_space.base_operation.generic_operations import Conv2d
from ..search_space.base_operation.generic_operations import Identity
from ..search_space.base_operation.generic_operations import StridedSepConv
from ..utilities.logger import Logger
import torch
from ..search_space.base_operation.generic_operations import Swish
from ..data_sources.pre_processing import cifar100_transforms
from ..search_space.base_operation.generic_operations import StridedConv2d
from ..data_sources.task_manager import TaskManager
from ..utilities.functional_utils.search_space_utils import spatial_ops_boost_hook
from ..data_sources.vision_datasource import VisionDataSource
from ..utilities.argument_parser import Params
import numpy as np
from ..neural_architecture_search.cnas import ContinualNAS
from ..search_space.base_operation.generic_operations import ReduceResolution
from ..evaluation_strategy.base_evaluator.image_classification_evaluator import ImageClassificationEvaluator
from ..evaluation_strategy.base_xai_interpreter.deep_taylor_decomposition import DeepTaylorDecomposition
from ..data_sources.base_optimization_objective.image_classification_objective import ICObjective
from ..search_space.base_operation.generic_operations import HSwish
from ..search_space.base_operation.generic_operations import TransformChannels
from ..data_sources.base_task.vision_task import VisionTask
from ..search_space.base_operation.generic_operations import AvgPool2d
from ..search_space.base_operation.generic_operations import DilatedConv2d
from ..search_space.base_operation.generic_operations import ReLU
from ..search_space.base_operation.generic_operations import LayerNormalization
from ..search_space.base_operation.generic_operations import LeakyReLU
from ..search_space.base_search_space.layer_wise_search_space import LWSearchSpace
from ..search_space.base_operation.generic_operations import Dropout
from ..search_algorithm.base_optimizer.random_search import RandomSearch
from ..data_sources.pre_processing import mnist_transforms
from ..search_space.base_operation.generic_operations import SepConv2d
from ..data_sources.pre_processing import cifar10_transforms
from ..search_space.base_operation.generic_operations import InstanceNormalization
from ..utilities.functional_utils.graph_utils import predecessor_successor_lists
from ..search_space.base_operation.generic_operations import MaxPool2d
from ..search_space.base_operation.generic_operations import GroupNormalization
from ..search_space.base_operation.generic_operations import BatchNormalization


OPERATIONS = [
    Conv2d,
    SepConv2d,
    DilatedConv2d,
    Identity,

    MaxPool2d,
    AvgPool2d,
    # GlobalAvgPool2d, # flattens the spatial dimensions (keeps the channel
                       # dim), hence invalidating most operations it precedes
    TransformChannels,
    ReduceResolution,
    StridedConv2d,
    StridedSepConv,

    BatchNormalization,
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization,
    Dropout,

    ReLU,
    Swish,
    HSwish,
    LeakyReLU,

    # ConvBnReluBlock,
    # ResidualBlock,
    # InceptionBlock
]

config = Params.get_args()

# set_reproducible(random_seed=42)

# Datasets
mnist_dataset = VisionDataSource(path='./mnist/',
                                 dataset=VisionDataSource.Dataset.MNIST,
                                 transform=mnist_transforms(),
                                 autoload=True)
mnist_obj = ICObjective()
mnist_obj.add_criterion(metric=ICObjective.Metric.VAL_ACC,
                        min_threshold=0.6,
                        target_threshold=0.9,
                        thresholds_enabled=False,
                        score_weight=0.7)   # score weights will be normalized;
                                            # they are just relative to each
                                            # other
mnist_obj.add_criterion(metric=ICObjective.Metric.MODEL_SIZE,
                        score_weight=0.1)
mnist_obj.add_criterion(metric=ICObjective.Metric.TRAIN_ACC_CONV,
                        score_weight=0.2)

cifar10_dataset = VisionDataSource(path='./cifar10/',
                                   dataset=VisionDataSource.Dataset.CIFAR10,
                                   transform=cifar10_transforms(),
                                   autoload=True)
cifar10_obj = ICObjective()
cifar10_obj.add_criterion(metric=ICObjective.Metric.VAL_ACC,
                          min_threshold=0.6,
                          target_threshold=0.9,
                          thresholds_enabled=False,
                          score_weight=0.8)
cifar10_obj.add_criterion(metric=ICObjective.Metric.TRAIN_ACC_CONV,
                          score_weight=0.2,
                          thresholds_enabled=False)

VisionDataSource.download_cifar100(path='./cifar100/',
                                   force_overwrite=False,
                                   allow_segmentation=True)

# Search Space
ss = LWSearchSpace(
    num_vertices=6,
    operations=OPERATIONS,
    encoding='multi-branch',
)

ss.register_sampling_hook('spatial_ops_boost', spatial_ops_boost_hook)

# Tasks
def visiontaskfactory(id, version, name, datasource, obj=None):
    task = VisionTask(id=id, version=version, name=name, datasource=datasource,
                      search_space=ss,
                      nas_epochs=3, candidate_epochs=2, objective=obj,
                      callbacks=[])
    input_shape, output_shape = task.shapes

    Logger.info(f'Task ({id}; {name}) - Shapes: {input_shape} | {output_shape}')

    return task


task_manager = TaskManager()
task_manager.add_task(visiontaskfactory(0, 0, 'cifar10', cifar10_dataset,
                                        cifar10_obj))
task_manager.add_task(visiontaskfactory(1, 0, 'mnist', mnist_dataset,
                                        mnist_obj))


c100_ds = VisionDataSource.Dataset.CIFAR100
for seg_idx in range(10):
    # segment_size = 10
    tns = cifar100_transforms()
    seg_task = VisionDataSource.class_segmentation_factory('./cifar100',
                                                           dataset=c100_ds,
                                                           segment_size=10,
                                                           segment_idx=seg_idx,
                                                           transform=tns)

    cifar100_obj = ICObjective()
    cifar100_obj.add_criterion(metric=ICObjective.Metric.VAL_ACC,
                               min_threshold=0.15,
                               target_threshold=0.9,
                               thresholds_enabled=True,
                               score_weight=0.8)
    cifar100_obj.add_criterion(metric=ICObjective.Metric.TRAIN_ACC_CONV,
                               score_weight=0.1)
    cifar100_obj.add_criterion(metric=ICObjective.Metric.COMP_PERF,
                               score_weight=0.1)
    task_manager.add_task(visiontaskfactory(2, seg_idx,
                                            f'cifar100_{seg_idx}', seg_task,
                                            cifar100_obj))


# Optimizer
search_algorithm = RandomSearch()


# XAI Interpreter
xai_interpreter = DeepTaylorDecomposition(false_pred_count=32,
                                          true_pred_count=32)


# Evaluation Strategy
evaluation_strategy = ImageClassificationEvaluator(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_training_logs=True,
    verbose=True,
    xai_interpreter=xai_interpreter
)


# Neural Architecture Search
nas = ContinualNAS(search_algorithm=search_algorithm,
                   evaluation_strategy=evaluation_strategy)

try:
    nas.run(task_manager=task_manager)
except Exception as e:
    print(e)
    raise e
#finally:
    # runtime.unassign()



nas.optimizer._best_candidate


adj_matrix = np.array([[0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0]]);

pred, succ = predecessor_successor_lists(adj_matrix)

succ

