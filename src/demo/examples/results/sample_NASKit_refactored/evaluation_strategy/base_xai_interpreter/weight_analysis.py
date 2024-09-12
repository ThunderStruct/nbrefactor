from .base_xai_interpreter import BaseXAIInterpreter

class WeightAnalysis(BaseXAIInterpreter):

    NAME = 'WA'

    def __init__(self):
        super().__init__()

    def register_hooks(self, model):
        pass

    def interpret(self):
        pass


