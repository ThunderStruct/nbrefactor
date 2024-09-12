from ...search_space.base_operation.role_specific_operations import AlignConcat1x1
import torch
from ...search_space.base_operation.role_specific_operations import OutputStem
from ...search_space.base_operation.role_specific_operations import InputStem
from .base_xai_interpreter import BaseXAIInterpreter

class FisherInformation(BaseXAIInterpreter):

    NAME = 'FI'


    def __init__(self, false_pred_count, true_pred_count):
        super().__init__()

        self.set_caching(false_pred_count, true_pred_count)

        self.gradients = {}
        self.hooks = []


    def register_hooks(self, model):
        # hooks registered in the interpret() method for custom passes
        # running DTD within the training process is too expensive
        pass


    def interpret(self, model, x):
        model.eval()

        relevance_scores = {}
        intermediate_outputs = {}
        back_hooks = []
        forward_hooks = []

        # layer hooks
        def _backward_hook(module, grad_input, grad_output):
            relevance_scores[module.id] = grad_output[0]

        def _forward_hook(module, args, output):
            intermediate_outputs[module.id] = output[0]

        for op in model.nodes():
            if op is None:
                # omitted isolates
                continue

            hook_op = op

            if isinstance(op, AlignConcat1x1):
                # apply the hook to the prepended operation rather than the
                # multiplexer
                hook_op = op.prepend_to

            if isinstance(hook_op, InputStem) \
            or isinstance(hook_op, OutputStem):
                # Input/Output Stem are ignored, the relevance of the

                # OutputStem is inherently 0 using DTD since we're calculating
                # the gradients w.r.t the output, which throws off the
                # normalization
                continue

            back_hook = hook_op.register_full_backward_hook(_backward_hook,
                                                            prepend=False)
            back_hooks.append(back_hook)
            forward_hook = hook_op.register_forward_hook(_forward_hook,
                                                         prepend=False)
            forward_hooks.append(forward_hook)

        # forward pass the passed data
        logits = model(x)
        pred_class = torch.argmax(logits, dim=1)

        # init gradient of predicted class score w.r.t. the input
        out_grads = torch.zeros_like(logits)
        out_grads[0, pred_class] = 1.0

        # backward pass (triggers previously registered hooks)
        x.requires_grad_(True)
        model.zero_grad()
        logits.backward(out_grads)

        # calculate the DTD for each layer
        for layer_id, grad in relevance_scores.items():
            scores = grad * intermediate_outputs[layer_id]
            # convert the scores tensor into a scalar score value
            scalar = torch.mean(grad * intermediate_outputs[layer_id]).item()
                     # scores.sum(dim=tuple(range(scores.dim())), keepdim=True)
            relevance_scores[layer_id] = scalar

        # normalize scores [omitted; no need, the scores are relative]
        # self.normalize_scores(relevance_scores)

        # cleanup
        for hook in back_hooks + forward_hooks:
            hook.remove()

        del intermediate_outputs
        del back_hooks
        del forward_hooks

        return relevance_scores



