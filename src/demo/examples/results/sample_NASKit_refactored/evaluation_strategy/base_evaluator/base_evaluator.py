from collections import defaultdict
import abc

class BaseEvaluator(abc.ABC):

    def __init__(self,
                 device,
                 save_training_logs,
                 verbose,
                 xai_interpreter=None):
        super().__init__()

        self.device = device
        self.save_training_logs = save_training_logs
        self.verbose = verbose
        self.xai = xai_interpreter
        self.running_metrics = defaultdict(dict)


    # --------------------------------------------------------------------------
    #   Abstract Methods (implement to conform to this base class)
    # --------------------------------------------------------------------------


    @abc.abstractmethod
    def optimize(self, model, task):
        raise NotImplementedError('Abstract method was not implemented')


