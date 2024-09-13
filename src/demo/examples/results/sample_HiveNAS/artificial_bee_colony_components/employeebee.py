from ..helper_tools.logger import Logger
import numpy as np
from .artificialbee_base_class import ArtificialBee
import pandas as pd
from copy import deepcopy
from ..helper_tools.operational_parameters import Params
import time
"""Employee Bees' class responsible for the exploration phase of the
Artificial Bee Colony optimization
"""

class EmployeeBee(ArtificialBee):
    '''Employee Bees, responsible for explorations and partial exploitation of the
    solution space. Employees search for food sources in the neighborhood, evaluate
    candidates, and compute the probabilities needed for the stochastic onlooker assignment

    Attributes:
        center_fs (:class:`~core.abc.food_source.FoodSource`): the central food souorce \
        which can be greedy-selected by associated onlookers during exploitation. This \
        food source holds the best fitness in the evaluated vicinity (neighbors)
        food_source (:class:`~core.abc.food_source.FoodSource`): the employee's current \
        food source (i.e during non-initial iterations when employees are exploiting)
        id_tracker (int): the bee's ID for logging and tracking purposes
        trials (int): the number of trials/evaluations done around a given center \
        food source. Used to abandon an area once the abandonment limit is reached
    '''

    def __init__(self, food_source):
        '''Initializes an employee bee with a center food source

        Args:
            food_source (:class:`~core.abc.food_source.FoodSource`): the initial FoodSource to \
            be assigned as the :code:`center_fs`
        '''

        if not hasattr(EmployeeBee, 'id_tracker'):
            EmployeeBee.id_tracker = 0

        super(EmployeeBee, self).__init__(food_source, EmployeeBee.id_tracker)

        EmployeeBee.id_tracker += 1
        self.trials = 0
        self.center_fs = food_source   # center_fs can be greedy-selected by
                                       # child onlookers (i.e the new
                                       # optimization center)


    def search(self, obj_interface):
        '''
        Explore new random position (near previously-sampled position) and assigns it to the \
        current food source

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the given \
            objective interface used to sample/evaluate candidates
        '''

        if self.food_source.fitness is None:
            # unevaluated (first iteration after abandonment reset)
            return

        # find neighbor near food_source in the bee's memory
        self.food_source = self.get_random_neighbor(obj_interface)


    def reset(self, new_fs):
        '''Resets EmployeeBee once abandonment limit is reached

        Args:
            new_fs (:code:`core.abc.food_source.FoodSource`): a reintialization food source. \
            Assigned as the :code:`center_fs`
        '''

        self.trials = 0
        self.food_source = new_fs
        self.center_fs = new_fs


    def calculate_fitness(self):
        '''Calculate fitness of an :class:`~core.abc.employee_bee.EmployeeBee`, given by:

                         ⎧ 1 / (1 + Fm(Xm→))       if  Fm(Xm→)≥0
            Fit_m(Xm→)=  ⎨
                         ⎩ 1 + abs(Fm(Xm→))        if  Fm(Xm→)<0

        Returns:
            float: adjusted fitness value for the stochastic assignment operator
        '''

        fitm = 0
        if self.center_fs.fitness >= 0:
            fitm = 1 / (1 + self.center_fs.fitness)
        else:
            fitm = 1 + np.abs(self.center_fs.fitness)

        return fitm


    def compute_probability(self, sum_fitness):
        '''
        Calculate probability of an EmployeeBee being chosen by
        an OnlookerBee based on Fitess values; given by:

        Pn = Fit_n(Xn→) / [∑ (Fit_m(Xm→)) for all m]

        Args:
            sum_fitness (float): sum of all fitness values in the population, used for the roulette wheel selector

        Returns:
            float: calculated probability that the current employee should be selected by an onlooker
        '''

        return self.calculate_fitness() / sum_fitness


    def evaluate(self, obj_interface, itr):
        '''Evaluates sampled position and increments trial counter

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface \
            used to sample/evaluate candidates
            itr (int): current ABC iteration (for logging and result-saving purposes)

        Returns:
            :class:`pandas.Series`: a Pandas Series containing the evaluation's results (represents a \
            row in the main results CSV file)
        '''

        Logger.evaluation_log('EmployeeBee', self.id, self.food_source.position)
        t = time.time()

        res = obj_interface.evaluate(self.food_source.position)
        # unpack
        self.food_source.fitness = res['fitness']
        epochs = res['epochs']
        weights_filename = res['filename']
        params = res['params']
        momentum = res['momentum']

        self.food_source.time = time.time() - t
        # ACT early bandonment (ACT enabled and network could not pass epoch 1)
        abandon_early = Params['TERMINATION_THRESHOLD_FACTOR'] > 0.0 and epochs <= 1
        self.trials += 1 if not abandon_early else Params['ABANDONMENT_LIMIT']

        if self.center_fs.fitness is None:
            # Check if this evaluation is the first in its area
            # (Iteration 1 after reset; i.e no need to greedy-select)
            self.center_fs = deepcopy(self.food_source)
        else:
            # Greedy select for iterations 2 ... Abandonment Limit
            self.greedy_select(self.food_source, obj_interface.is_minimize)

        # save data
        series = pd.Series({
            'bee_type': type(self).__name__,
            'bee_id': self.id,
            'bee_parent': '-',
            'itr': itr,
            'candidate': self.food_source.position,
            'fitness': self.food_source.fitness,
            'center_fitness': self.center_fs.fitness,
            'momentum': sum([x[1] for _,x in momentum.items()]),
            'epochs': epochs,
            'momentum_epochs': 0,
            'params': params,
            'weights_filename': weights_filename,
            'time': self.food_source.time
        })
        return series


    def greedy_select(self, n_food_source, is_minimize):
        '''Update best FoodSource to minimize or maximize fitness (elitism)

        Args:
            n_food_source (:class:`~core.abc.food_source.FoodSource`): the new food source to \
            be greedy-selected
            is_minimize (bool): determines whether to minimize or maximize the greedy-selection
        '''

        if ((self.center_fs.fitness < n_food_source.fitness) and not is_minimize) or \
        ((self.center_fs.fitness > n_food_source.fitness) and is_minimize):
            self.center_fs.position = n_food_source.position
            self.center_fs.fitness = n_food_source.fitness


    def get_center_fs(self):
        '''Returns the center food source

        Returns:
            :class:`~core.abc.food_source.FoodSource`: the employee's center food source
        '''

        return self.center_fs


    def __str__(self):
        '''For logging/debugging purposes

        Returns:
            str: the pretty-print contents of the EmployeeBee
        '''

        return 'EmployeeBee {} -- FS: {}, trials: {}'.format(self.id, \
                                                             self.food_source, \
                                                             self.trials)


