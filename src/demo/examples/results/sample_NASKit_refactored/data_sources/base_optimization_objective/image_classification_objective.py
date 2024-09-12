from .base_optimization_objective import BaseObjective
from enum import auto
from enum import Enum
import numpy as np

class ICObjective(BaseObjective):
    """
    Image Classification Thresholds
    """

    class Metric(Enum):
        # Validation Accuracy
        VAL_ACC                 = auto()
        # Parameter Count
        MODEL_SIZE              = auto()
        # MFLOPs
        COMP_PERF               = auto()
        # Training Accuracy Convergence Rate
        # (second derivative of train. acc.)
        TRAIN_ACC_CONV          = auto()
        # Training Loss Convergence Rate
        # (second derivative of train. loss)
        TRAIN_LOSS_CONV          = auto()


    _DEFAULTS = {
        Metric.VAL_ACC: {
            'key': 'val_avg_acc',
            'polarity': 1,
            'weight': 0.8,
            # disable thresholds by default
            'thresholds_enabled': False,
            'min_threshold': 0.0,
            'target_threshold': 1.0
        },
        Metric.MODEL_SIZE: {
            'key': 'total_params',
            'polarity': -1,
            'weight': 0.1,
            # disable thresholds by default
            'thresholds_enabled': False,
            'min_threshold': np.inf,
            'target_threshold': 0.0
        },
        Metric.COMP_PERF: {
            'key': 'mflops',
            'polarity': -1,
            'weight': 0.1,
            # disable thresholds by default
            'thresholds_enabled': False,
            'min_threshold': np.inf,
            'target_threshold': 0.0
        },
        Metric.TRAIN_ACC_CONV: {
            # this value is typically negative
            # (training accuracy typically decelerates)
            'key': 'train_acc_conv_rate',
            'polarity': 1,
            'weight': 0.2,
            # disable thresholds by default
            'thresholds_enabled': False,
            'min_threshold': -1.0,
            'target_threshold': 1.0
        },
        Metric.TRAIN_LOSS_CONV: {
            # this value is also typically negative
            # (training loss typically decelerates)
            'key': 'train_loss_conv_rate',
            'polarity': -1,
            'weight': 0.1,
            # disable thresholds by default
            'thresholds_enabled': False,
            'min_threshold': -1.0,
            'target_threshold': 1.0
        }
    }


    def __init__(self):
        super(ICObjective, self).__init__()


