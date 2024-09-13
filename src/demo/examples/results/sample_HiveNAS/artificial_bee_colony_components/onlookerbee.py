from ..helper_tools.logger import Logger
from .artificialbee_base_class import ArtificialBee
import pandas as pd
from .foodsource import FoodSource
import time
"""Onlooker Bees' class responsible for the exploitation phase of the
Artificial Bee Colony optimization
"""

class OnlookerBee(ArtificialBee):
    '''Onlooker Bees, primarily responsible for the exploitation of \
    the solution space. Onlookers search for neighboring food sources.

    Attributes:
        employee (:class:`~core.abc.employee_bee.EmployeeBee`): the assigned \
        Employee Bee; stored to access greedy-selection
        food_source (:class:`~core.abc.food_source.FoodSource`): the current \
        food source (i.e neighbor)
        id_tracker (int): the bee's ID for logging and tracking purposes
    '''

    def __init__(self):
        '''Initializes an onlooker bee
        '''

        if not hasattr(OnlookerBee, 'id_tracker'):
            OnlookerBee.id_tracker = 0

        super(OnlookerBee, self).__init__(None, OnlookerBee.id_tracker)

        OnlookerBee.id_tracker += 1
        self.employee = None


    def search(self, obj_interface):
        '''
        Exploit position (near a random :class:`~core.abc.employee_bee.EmployeeBee` \
        chosen according to computed probability)

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the \
            objective interface used to sample/evaluate candidates
        '''

        self.food_source = self.get_random_neighbor(obj_interface)


    def assign_employee(self, employee):
        '''Assigns an EmployeeBee to the Onlooker for neighbor-search

        Args:
            employee (:class:`~core.abc.employee_bee.EmployeeBee`): the \
            assigned employee bee
        '''

        self.employee = employee
        self.food_source = FoodSource(self.employee.food_source.position)


    def evaluate(self, obj_interface, itr):
        '''
        Evaluates sampled position and increments employee's trial counter

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface \
            used to sample/evaluate candidates
            itr (int): current ABC iteration (for logging and result-saving purposes)

        Returns:
            :class:`pandas.Series`: a Pandas Series containing the evaluation's results (represents a \
            row in the main results CSV file)
        '''

        Logger.evaluation_log('OnlookerBee', self.id, self.food_source.position)
        t = time.time()

        res = obj_interface.evaluate(self.food_source.position)
        # unpack
        self.food_source.fitness = res['fitness']
        epochs = res['epochs']
        weights_filename = res['filename']
        params = res['params']
        momentum = res['momentum']

        self.food_source.time = time.time() - t
        self.employee.trials += 1
        self.employee.greedy_select(self.food_source, obj_interface.is_minimize)

        series = pd.Series({
            'bee_type': type(self).__name__,
            'bee_id': self.id,
            'bee_parent': self.employee.id,
            'itr': itr,
            'candidate': self.food_source.position,
            'fitness': self.food_source.fitness,
            'center_fitness': self.get_center_fs().fitness,
            'momentum': sum([x[1] for _,x in momentum.items()]) / len(momentum),
            'epochs': epochs,
            'momentum_epochs': 0,
            'params': params,
            'weights_filename': weights_filename,
            'time': self.food_source.time
        })
        return series


    def get_center_fs(self):
        '''Returns the parent's center food_source

        Returns:
            :class:`~core.abc.food_source.FoodSource`: associated employee bee's \
            center food source
        '''

        return self.employee.center_fs


    def __str__(self):
        '''For logging/debugging purposes

        Returns:
            str: the pretty-print contents of the OnlookerBee
        '''

        return 'OnlookerBee {} -> Parent Employee ({}) -- FS: {}'\
        .format(self.id, self.employee.id, self.food_source)


