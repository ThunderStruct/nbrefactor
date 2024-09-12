from ..data_sources.task_manager import TaskManager
from ..search_algorithm.continual_optimization_protocol import CLOptimizerProtocol
from ..evaluation_strategy.continual_evaluation_protocol import CLEvaluatorProtocol
from ..utilities.functional_utils.environment_utils import get_system_usage
from ..utilities.logger import Logger
from .nas import NAS


class ContinualNAS(NAS):

    def __init__(self,
                 search_algorithm,
                 evaluation_strategy,
                 visualize_candidates=True):

        self.__encountered_tasks = []

        super(ContinualNAS, self).__init__(search_algorithm,
                                           evaluation_strategy,
                                           visualize_candidates)

    def run(self, task_manager, dir='./nas_results/'):
        """
        """

        assert isinstance(task_manager, TaskManager), (
            'Assigned `task_manager` must be of type `TaskManager`. '
            f'Type {type(task)} was given instead.'
        )
        assert isinstance(self.optimizer, CLOptimizerProtocol), (
            'The assigned Search Algorithm must conform to '
            '`CLOptimizerProtocol` in order to run NAS with Continual '
            'Learning capabilities'
        )
        assert isinstance(self.evaluator, CLEvaluatorProtocol), (
            'The assigned Search Algorithm must conform to '
            '`CLEvaluatorProtocol` in order to run NAS with Continual '
            'Learning capabilities'
        )

        for task in task_manager.fixed_scheduler():

            # save task details prior to runnning NAS in case of abortion
            # intermediary model data and training logs are saved during
            # evaluation
            task.save()
            self.__encountered_tasks.append(task)

            Logger.info((
                f'Initiating Task {task.id} v.{task.version} - {task.name} '
                f'({str(task.metadata)})'
            ))

            # ------------------------------------------------------------------
            # ASSIGN AND FIT NEW TASK

            # Assign new task & check if fine-tuning is required (class/domain-
            # incremental scenarios; if new classes are added or task-boundary
            # has shifted)
            self.optimizer.assign_task(task)
            top_candidate, should_finetune = self.optimizer.fit()

            if should_finetune:
                Logger.progress('Task fitting successful, fine-tuning...')
                # state_dict = top_candidate.state_dict()     # preserve weights

                # fine-tune model for the current task
                self.evaluator.fine_tune(top_candidate, task)

                # assess the fine-tuning to see if NAS changes are needed
                nas_needed = False
                for e_task in self.__encountered_tasks:
                    metrics = self.evaluator.evaluate(top_candidate, e_task)
                    if not e_task.objective.min_threshold_met(metrics):
                        nas_needed = True
                        break

                if not nas_needed:
                    # move on to the next task
                    # del state_dict    # [no longer instantiated]
                    Logger.success(f'{task.id} v.{task.version} - "{task.name}"'
                                   f' is complete (fine-tuned)!')
                    Logger.separator()

                    f_name = f'{task.id}-{task.version}_results.csv'
                    self.optimizer.metrics.save(filename=f_name,
                                                dir=dir)

                    self.sys_metrics.add_record(task_id=task.id,
                                                task_version=task.version,
                                                task_name=task.name,
                                                nas_epoch=nas_epoch,
                                                sys_usage=get_system_usage())
                    self.sys_metrics.save(filename='sys_usage.csv')

                    continue

            # ------------------------------------------------------------------
            # NAS LOOP

            # if a minimum threshold is set for a task, the loop is infinite
            # until the minimum is met, otherwise stop at `nas_epochs`
            nas_epoch = 0
            while True:

                # --------------------------------------------------------------
                # SAMPLE CANDIDATE(S)

                models = []
                if top_candidate is not None:
                    # augment candidate(s) to fit a new task

                    # extensions are for class-/domain-incremental scenarios
                    # where fine-tuning was not sufficient
                    models = self.optimizer.augment(base_model=top_candidate)
                else:
                    # initial task optimization (sample from scratch)
                    models = self.optimizer.sample()

                if models is None or not len(models):
                    # could not find a valid architecture after a number of
                    # attempts
                    continue

                # --------------------------------------------------------------
                # EVALUATE CANDIDATE(S)

                # train sampled network
                self.optimize_candidates(models, nas_epoch, task)

                # top candidate selection
                self.optimizer.candidate_selection(models=models,
                                                   tasks=self.\
                                                   __encountered_tasks)

                # housekeeping; gc triggered at the end of every NAS epoch only
                # as it is inefficient if initiated more frequently
                self.free_mem()


                # --------------------------------------------------------------
                # SAVE RESULTS

                f_name = f'{task.id}-{task.version}_results.csv'
                self.optimizer.metrics.save(filename=f_name,
                                            dir=dir)

                self.sys_metrics.add_record(task_id=task.id,
                                            task_version=task.version,
                                            task_name=task.name,
                                            nas_epoch=nas_epoch,
                                            sys_usage=get_system_usage())
                self.sys_metrics.save(filename='sys_usage.csv')


                # --------------------------------------------------------------
                # ASSESS THRESHOLDS

                # perform NAS break checks
                nas_epoch += 1

                metrics = self.optimizer.top_metrics.records[-1]    # last eval

                if task.objective.target_threshold_met(metrics):
                    Logger.success('Target threshold reached!')
                    break

                if nas_epoch >= task.nas_epochs:
                    minimum_is_met = True
                    # check if the minimum threshold is met across all tasks
                    for t in self.__encountered_tasks:
                        # Logger.debug(str(t))
                        # Logger.debug(metrics)
                        if not t.objective.min_threshold_met(metrics):
                            minimum_is_met = False
                            # NAS epochs reached but minimum thresholds not yet
                            # met; issue warning
                            Logger.warning(
                                f'Minimum threshold for task {t.id} is not '
                                'met! Overriding the given NAS epochs\' count'
                            )
                            # break     # skip breaking -> warn about all tasks

                    if minimum_is_met:
                        # minimum threshold is met and NAS epochs reached
                        break


            # ------------------------------------------------------------------
            # END OF NAS

            # Reset XAI parameters
            if self.evaluator.xai is not None:
                self.evaluator.xai.reset()

            # NAS optimization complete
            top_train_acc = max(self.optimizer.top_metrics['train_avg_acc'])
            top_val_acc = max(self.optimizer.top_metrics['val_avg_acc'])
            Logger.success(f'NAS optimization for task: {str(task.id)} '
                        f'v.{task.version} '
                        f'| {str(task.name)} is complete! Top1 train_acc: '
                        f'{top_train_acc}, Top1 val_acc: {top_val_acc}')
            Logger.separator()


