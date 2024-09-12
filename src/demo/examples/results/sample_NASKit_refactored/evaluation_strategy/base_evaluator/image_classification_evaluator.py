import torch.optim as optim
import time
from copy import deepcopy
from ...performance_metrics.evaluation_metrics import EvaluationMetrics
from ...utilities.logger import Logger
from ...utilities.functional_utils.misc_utils import flatten_dict
from .base_evaluator import BaseEvaluator
from collections import defaultdict
import torch
from ..continual_evaluation_protocol import CLEvaluatorProtocol
from ...utilities.logger import TrainingLogger
import torch.nn as nn

class ImageClassificationEvaluator(BaseEvaluator, CLEvaluatorProtocol):

    def __init__(self,
                 device,
                 save_training_logs,
                 verbose,
                 xai_interpreter=None):
        super().__init__(device=device,
                         save_training_logs=save_training_logs,
                         verbose=verbose,
                         xai_interpreter=xai_interpreter)


    def _calculate_batch_metrics(self, split, loss,
                                 outputs, labels, n_batches,
                                 top_k=[]):
        """
        TODO: add hook for batch/epoch metrics calculation to allow additional
        metrics' calculations per task
        """

        if split not in self.running_metrics:
            # init running metrics
            self.running_metrics[split]['running_loss'] = 0.0
            self.running_metrics[split]['correct'] = 0
            self.running_metrics[split]['samples'] = 0
            self.running_metrics[split]['avg_acc'] = 0
            self.running_metrics[split]['avg_loss'] = 0
            self.running_metrics[split]['acc_conv_rates'] = []
            self.running_metrics[split]['loss_conv_rates'] = []
            if len(top_k) > 0:
                # `avg_acc` == `top_1_acc`
                self.running_metrics[split]['top_k'] = {
                    k: (0.0, 0.0) for k in top_k
                }

        metrics = self.running_metrics[split]
        metrics['running_loss'] += loss.item()

        _, y_pred = torch.max(outputs, 1)
        prev_acc = metrics['avg_acc']
        prev_loss = metrics['avg_loss']

        metrics['correct'] += (y_pred == labels).sum().item()
        metrics['samples'] += labels.size(0)

        metrics['avg_loss'] = metrics['running_loss'] / n_batches
        metrics['avg_acc'] = metrics['correct'] / metrics['samples']

        metrics['acc_conv_rates'].append(metrics['avg_acc'] - prev_acc)
        metrics['loss_conv_rates'].append(metrics['avg_loss'] - prev_loss)

        self.running_metrics[split] = metrics

        if len(top_k) > 0:
            for k in top_k:
                # calculate top_k accuracy
                _, pred = outputs.topk(k, 1, True, True)
                crct, tot = metrics['top_k'][k]
                crct += torch.eq(pred, labels.view(-1, 1)).sum().item()
                tot += labels.size(0)
                self.running_metrics[split]['top_k'][k] = (crct, tot)

        return self.running_metrics[split]


    def _calculate_epoch_metrics(self, reset_running_metrics):
        res_dict = deepcopy(self.running_metrics)
        for split in self.running_metrics:
            top_k = {}
            if 'top_k' in self.running_metrics[split]:
                temp_top_k = self.running_metrics[split]['top_k']
                for k in temp_top_k:
                    top_k[k] = temp_top_k[k][0] / temp_top_k[k][1]

                res_dict[split]['top_k'] = top_k

            # acc/loss second derivatives (our custom convergence rates)
            res_dict[split]['acc_conv_rate'] = self.__convergence_derivative(
                res_dict[split]['acc_conv_rates'])
            res_dict[split]['loss_conv_rate'] = self.__convergence_derivative(
                res_dict[split]['loss_conv_rates'])

            # delete recorded rates (too many unnecessary values; were only
            # used to calculated the convergence derivative)
            del res_dict[split]['acc_conv_rates']
            del res_dict[split]['loss_conv_rates']

        if reset_running_metrics:
            self.running_metrics = defaultdict(dict)

        return res_dict


    def __convergence_derivative(self, convergence_rates):
        sec_deriv = []      # the difference of means of adjacent pairs
        for i in range(1, len(convergence_rates) - 1):
            # (i-1 and i+1 to get the central difference for numerical
            # differentiation)
            rate_change = (convergence_rates[i+1] - convergence_rates[i-1]) / 2
            sec_deriv.append(rate_change)

        # we take the mean of the second derivative values to get a scalar value
        if len(sec_deriv) > 0:
            # this inequality will only fail if we have < 4 batches
            # (3 batches => 2 approx. numerical 1st derivates => cannot get
            # central difference for 2nd deriv.)
            return sum(sec_deriv) / len(sec_deriv)

        return 0


    def _train(self, model, epoch, loader, optimizer, criterion):
        model.train()

        # training loop
        n_batches = len(loader)
        for batch_idx, (inputs, labels) in enumerate(loader):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            outputs = None
            try:
                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward pass
                loss.backward()
                optimizer.step()

                # calculate/update metrics
                batch_res = self._calculate_batch_metrics(split='train',
                                                          loss=loss,
                                                          outputs=outputs,
                                                          labels=labels,
                                                          n_batches=n_batches,
                                                          top_k=[1, 5])

                if self.verbose:
                    TrainingLogger.log_training(epoch,
                                                batch_idx,
                                                batch_res['avg_loss'],
                                                batch_res['avg_acc'])

            except Exception as e:
                Logger.debug(model.adj_matrix)
                Logger.debug(str(model))
                model.visualize()
                raise e


    def _validate(self, model, epoch, loader, criterion):
        model.eval()

        with torch.no_grad():
            # validation loop
            n_batches = len(loader)
            for batch_idx, (inputs, labels) in enumerate(loader):
                # set tensor device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # calculate/update metrics
                batch_res = self._calculate_batch_metrics(split='val',
                                                          loss=loss,
                                                          outputs=outputs,
                                                          labels=labels,
                                                          n_batches=n_batches,
                                                          top_k=[1, 5])

                # pass samples to the interpreter for caching (if applicable)
                if self.xai is not None:
                    _, y_pred = torch.max(outputs, 1)
                    self.xai.cache_predictions(inputs, (y_pred == labels))

                if self.verbose:
                    TrainingLogger.log_validation(epoch,
                                                  batch_idx,
                                                  batch_res['avg_loss'],
                                                  batch_res['avg_acc'])


    def optimize(self, model, task, fine_tune=False, dir='./model_metrics/'):
        """
        Candidate optimization; this is the "evaluation" phase in a traditional
        NAS cycle.

        Args:
            model (:class:`~Network`): the candidate model to be optimized (\
            in-place)
            task (:class:`~BaseTask`): any :class:`~BaseTask` sub-\
            class, defining the optimization task/objectives
            fine_tune (:obj:`bool`): defines whether the optimization \
            should fine-tune the model or fully optimize it.
        """

        if self.device.type != 'cpu':
            # move model to GPU
            model.cuda()

        model.activate_task(task.id)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # candidate optimzer
        # optimizer = optim.SGD(model.parameters(),
        #                       lr=task.lr,
        #                       momentum=0.9)
        lr = task.learning_rate if not fine_tune else task.finetune_lr
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr)

        # init loaders
        train_loader, val_loader = task.loaders()

        # training metrics
        eval_metrics = EvaluationMetrics(task.metadata)

        if self.verbose:
            TrainingLogger.reset(task.id, task.version, task.name,
                                 task.candidate_epochs,
                                 len(train_loader), len(val_loader),
                                 log_to_file=self.save_training_logs)

        for epoch in range(task.candidate_epochs):
            start_time = time.time()
            early_stop = False

            self._train(model, epoch, train_loader, optimizer, self.criterion)
            self._validate(model, epoch, val_loader, self.criterion)

            metrics = self._calculate_epoch_metrics(reset_running_metrics=True)
            flat_metrics = flatten_dict(metrics, sep='_')

            eval_metrics.add_metrics(metrics=flat_metrics,
                                     epoch=epoch,
                                     start_time=start_time)

            if task.objective.target_threshold_met(eval_metrics):
                break

            for cb in task.callbacks:
                cb(flat_metrics, task.num_classes, model)
                if cb.early_stop:
                    early_stop = True

            if early_stop:
                Logger.critical(f'Early stopping candidate...')
                break

        # TODO: move the XAI calculation and recording to a hook for flexibility
        # init XAI data
        xai_name, xai_scores = 'None', []
        if self.xai is not None:
            xai_scores = self.xai.interpret(model,
                                            self.xai.get_cached_predictions()\
                                            .to(self.device))
            xai_name = self.xai.name

        Logger.progress(f'XAI: {xai_name} -- Scores {xai_scores}')

        # commit candidate data
        TrainingLogger.commit_file(filename=str(hash(model)) + '.txt')

        model.metrics.add_eval_metrics(eval_metrics)
        model.metrics.save(filename=str(hash(model)) + '.csv',
                           dir=dir)


    def evaluate(self, model, task, insert_metrics=True, epochs=1,
                 dir='./model_metrics/'):
        """


        Args:
            insert_metrics (:obj:`bool`): whether or not to insert \
            the evaluation metrics into the :obj:`~model.metrics`
            epochs (:obj:`int`): number of validation epochs. This defaults \
            to `1` as the network is not being optimized, hence the \
            validation metrics are going to be static. Argument is added in \
            case a special case needs multiple epochs
        """

        if self.device.type != 'cpu':
            # move model to GPU
            model.cuda()

        model.activate_task(task.id)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # init validation loader
        _, val_loader = task.loaders()

        # reset logger
        if self.verbose:
            TrainingLogger.reset(task.id, task.version, task.name,
                                 epochs,
                                 0, len(val_loader),
                                 log_to_file=self.save_training_logs)

        # training metrics
        eval_metrics = EvaluationMetrics(task.metadata)

        Logger.info(
            f'Evaluating model ({str(model.wl_hash)}) '
            f'for task {task.id} v.{task.version} | {task.name}...'
        )

        for epoch in range(epochs):
            start_time = time.time()

            self._validate(model, epoch, val_loader, self.criterion)

            metrics = self._calculate_epoch_metrics(reset_running_metrics=True)
            flat_metrics = flatten_dict(metrics, sep='_')

            eval_metrics.add_metrics(metrics=flat_metrics,
                                     epoch=epoch,
                                     start_time=start_time)

        # commit candidate data if applicable
        if insert_metrics:
            model.metrics.add_eval_metrics(eval_metrics)
            model.metrics.save(filename=str(hash(model)) + '.csv',
                               dir=dir)

        return eval_metrics


    def fine_tune(self, model, task, output_layer_only=False,
                  dir='./model_metrics/'):
        """
        """

        if output_layer_only:
            # freeze all layers except the output layer; class-incremental
            for param in model.parameters():
                param.requires_grad = False
            # unfreeze output layer
            for param in model.get_output_layer(task.id).parameters():
                param.requires_grad = True

        return self.optimize(model, task, fine_tune=True, dir=dir)


