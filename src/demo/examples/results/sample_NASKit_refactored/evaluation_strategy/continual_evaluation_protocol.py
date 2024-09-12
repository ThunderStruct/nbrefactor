import abc

class CLEvaluatorProtocol(abc.ABC):


    @abc.abstractmethod
    def evaluate(self, model, task):
        raise NotImplementedError('Abstract method was not implemented')

    @abc.abstractmethod
    def fine_tune(self, model, task,
                  output_layer_only=False, dir='./model_metrics/'):
        """
        """
        raise NotImplementedError('Abstract method was not implemented')

