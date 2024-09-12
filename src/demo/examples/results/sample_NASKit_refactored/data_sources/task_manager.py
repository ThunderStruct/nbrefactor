from .base_task import BaseTask

class TaskManager:

    def __init__(self, task_list=[]):
        self.tasks = task_list


    def add_task(self, task):
        assert isinstance(task, BaseTask), (
            'Invalid task provided. Please ensure that the argument is of type'
            ' `BaseTask`'
        )

        # [Deprecated] - duplicate IDs are now valid for class-/domain-increment
        # ids = [t.id for t in self.tasks]
        # if task.id in ids:
        #     Logger.warning(f'Task ID "{task.id}" was already used. Ignoring.')
        #     return

        self.tasks.append(task)


    def fixed_scheduler(self):
        for task in self.tasks:
            yield task

