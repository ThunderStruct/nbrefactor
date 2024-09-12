from ..search_space.base_operation.generic_operations import Conv2d
from ..search_space.base_operation.generic_operations import Identity
from ..search_space.base_operation.generic_operations import StridedSepConv
from ..utilities.logger import Logger
import torch
from ..search_space.base_operation.generic_operations import Swish
from ..search_space.base_operation.generic_operations import StridedConv2d
from ..data_sources.vision_datasource import VisionDataSource
from google.colab import runtime
from ..utilities.argument_parser import Params
from ..evaluation_strategy.evaluation_callbacks.adaptive_cutoff_threshold import AdaptiveCutoffThreshold
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
from ..search_space.base_operation.generic_operations import MaxPool2d
from ..search_space.base_operation.generic_operations import GroupNormalization
from ..neural_architecture_search.nas import NAS
from ..utilities.reproducibility import set_reproducible
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

set_reproducible(random_seed=42)


# Dataset
cifar10_dataset = VisionDataSource(
    path='./cifar10',
    dataset=VisionDataSource.Dataset.CIFAR10,
    transform=cifar10_transforms()
)
cifar10_dataset.load()

mnist_dataset = VisionDataSource(
    path='./mnist',
    dataset=VisionDataSource.Dataset.MNIST,
    transform=mnist_transforms()
)
mnist_dataset.load()


# Search Space
ss = LWSearchSpace(
    num_vertices=7,
    operations=OPERATIONS,
    encoding='multi-branch'
)


# Tasks
cifar10_task = VisionTask(
    id=0,
    version=0,
    name='cifar10',
    datasource=cifar10_dataset,
    candidate_epochs=5,
    nas_epochs=3,
    search_space=ss
)

mnist_obj = ICObjective()
mnist_obj.add_criterion(metric=ICObjective.Metric.VAL_ACC,
                        min_threshold=0.6,
                        target_threshold=0.9,
                        thresholds_enabled=True)
mnist_task = VisionTask(
    id=1,
    version=0,
    name='mnist',
    datasource=mnist_dataset,
    candidate_epochs=3,
    nas_epochs=10,
    search_space=ss,
    objective=mnist_obj,
    callbacks=[
        AdaptiveCutoffThreshold()
    ]
)
input_shape, output_shape = cifar10_task.shapes
Logger.info(f'Shapes: {input_shape} | {output_shape}')

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
nas = NAS(search_algorithm=search_algorithm,
          evaluation_strategy=evaluation_strategy)

try:
    nas.run(task=mnist_task)
except Exception as e:
    print(e)
    raise e
finally:
    runtime.unassign()


