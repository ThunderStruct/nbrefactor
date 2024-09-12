import gc
import torch
from ..data_sources.base_task import BaseTask
from ..utilities.functional_utils.environment_utils import get_system_usage
from ..search_space.model.network import Network
from ..utilities.logger import Logger
from ..performance_metrics.system_metrics import SystemMetrics


class NAS:

    def __init__(self,
                 search_algorithm,
                 evaluation_strategy,
                 visualize_candidates=True,
                 **kwargs):

        self.sys_metrics = SystemMetrics()
        self.visualize_candidates = visualize_candidates

        # SEARCH ALGORITHM INIT
        self.optimizer = search_algorithm

        # EVALUATION STRATEGY INIT
        self.evaluator = evaluation_strategy


    def optimize_candidates(self, models, nas_epoch, task):
        for model in models:
            # WL hash
            hash = model.wl_hash

            # visualize model pre-training (computationally expensive,
            # consider using only when debugging)
            if self.visualize_candidates:
                model.visualize(dir='./plots',
                                show_plot=True)

            # log architecture and parameters' info pre-training
            Logger.info(f'\nEvaluating Arch {nas_epoch+1}/'
                        f'{task.nas_epochs} "{hash}"')
            Logger.info(f'\nModel: {model.metadata.pretty_print()}')
            Logger.info(f'{model.learnable_params} learnable parameters out of '
                        f'{model.total_params} total parameters')

            # evaluate candidate (model training/validation)
            self.evaluator.optimize(model, task)

            # candidate evaluation complete
            Logger.success(f'Completed candidate "{hash}" evaluation')
            Logger.separator()

            # pass evaluation feedback to optimizer
            self.optimizer.add_results(model.metrics.aggregate())


    def free_mem(self):
        """
        This should be the only point of garbage collection as
        :func:`gc.collect` can be computationally expensive. `del` calls
        throughout the NAS runs will be freed here
        """
        gc.collect()

        if self.evaluator.device.type != 'cpu':
            torch.cuda.empty_cache()


    def reproduce(self, serialized_graph):
        """
        """

        model = Network.deserialize(serialized_graph)
        self.evaluate(model, 0)


    def run(self, task, dir='./nas_results/'):
        """
        """

        assert isinstance(task, BaseTask), (
            'Assigned task must be of type `BaseTask`. '
            f'Type {type(task)} was given instead.'
        )

        # save task details prior to runnning NAS in case of abortion
        # intermediary model data and training logs are saved during evaluation
        task.save()

        # assign given task to the optimizer
        self.optimizer.assign_task(task)

        # ----------------------------------------------------------------------
        # NAS Loop
        # if a minimum threshold is set for a task, the loop is
        # infinite until the minimum is met, otherwise stop at `nas_epochs`
        nas_epoch = 0
        while True:
            # sample candidate(s) from the search space
            models = self.optimizer.sample()
            if models is None or not len(models):
                # could not find a valid architecture after a number of
                # attempts
                continue

            # train sampled network
            self.optimize_candidates(models, nas_epoch, task)

            # top candidate selection
            self.optimizer.candidate_selection(models=models,
                                               tasks=[task])

            # housekeeping; gc triggered at the end of every NAS epoch only
            # as it is inefficient if initiated more frequently
            self.free_mem()

            # perform NAS break checks
            nas_epoch += 1

            metrics = self.optimizer.top_metrics.records[-1]    # last eval

            if nas_epoch >= task.nas_epochs \
            and task.objective.min_threshold_met(metrics):
                break

            if task.objective.target_threshold_met(metrics):
                Logger.success('Target threshold reached!')
                break

            if nas_epoch >= task.nas_epochs:
                # NAS epochs reached but minimum not yet met; issue warning
                Logger.warning(
                    f'Minimum threshold for task {task.id} v.{task.version} is '
                    'not met! Overriding the given NAS epochs\' count'
                )

        # NAS loop complete, save results
        self.optimizer.metrics.save(filename='results.csv',
                                    dir=dir)

        self.sys_metrics.add_record(task_id=task.id,
                                    task_version=task.version,
                                    task_name=task.name,
                                    nas_epoch=nas_epoch,
                                    sys_usage=get_system_usage())

        self.sys_metrics.save(filename='sys_usage.csv')

        # NAS optimization complete
        top_train_acc = max(self.optimizer.top_metrics['train_avg_acc'])
        top_val_acc = max(self.optimizer.top_metrics['val_avg_acc'])
        Logger.success(f'NAS optimization for task: {str(task.id)} '
                    f'v.{task.version} | '
                    f'{str(task.name)} is complete! Top1 train_acc: '
                    f'{top_train_acc}, Top1 val_acc: {top_val_acc}')
        Logger.separator()


