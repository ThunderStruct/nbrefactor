import time
from .onlookerbee import OnlookerBee
import os
from ..helper_tools.file_handler import FileHandler
import pandas as pd
import numpy as np
from .scoutbee import ScoutBee
from .employeebee import EmployeeBee
from ..helper_tools.operational_parameters import Params
from ..helper_tools.logger import Logger

"""The main Artificial Bee Colony optimization algorithm
"""

class ArtificialBeeColony:
    '''Artificial Bee Colony optimizer

    Attributes:
        colony_size (int, optional): bee population size; defaults to value set in \
        :class:`~config.params.Params`
        employees (list): list of all :class:`~core.abc.employee_bee.EmployeeBee` used in the \
        optimization
        eo_colony_ratio (float, optional): employees to onlookers ratio; defaults to value set in \
        :class:`~config.params.Params`
        obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective \
        interface defining the task boundaries
        onlookers (list): list of all :class:`~core.abc.onlooker_bee.OnlookerBee` used in the \
        optimization
        results_df (:class:`pandas.DataFrame`): the main results DataFrame containing all evaluated \
        results
        scouts (list): list of all :class:`~core.abc.scout_bee.ScoutBee`-sampled positions used in  \
        the optimization. Scout Bees are *not* instantiated.
        scouts_count (int): number of scouts / parallel explorations. Follows the classical ABC \
        1-to-1 scout-to-employee ratio
        total_evals (int): total number of :func:`~core.abc.artificial_bee.ArtificialBee.evaluate` \
        calls (for both Employees and Onlookers)
    '''

    def __init__(self, obj_interface, \
                 colony_size=Params['COLONY_SIZE'], \
                 employee_onlooker_ratio=Params['EMPLOYEE_ONLOOKER_RATIO']):
        ''' Initializes the ABC algorithm '''

        self.obj_interface = obj_interface    # Encapsulates the Search Space +
                                              # Evaluation Strategy
        self.colony_size = colony_size
        self.eo_colony_ratio = employee_onlooker_ratio
        self.scouts_count = int(self.colony_size * self.eo_colony_ratio)


    def __init_scouts(self):
        '''Initial :class:`~core.abc.food_source.FoodSource` sampling by Scout Bees \
        *(does not evaluate fitness)*
        '''

        for _ in range(self.scouts_count):
            self.scouts.append(ScoutBee.sample(self.obj_interface))


    def __init_employees(self):
        ''' Instantiate Employee Bees and assign a Scout position to each '''

        # Floor of (colony_size * ratio)
        employee_count = int(self.colony_size * self.eo_colony_ratio)

        for itr in range(employee_count):
            # Split scouts evenly among employees
            scout = self.scouts[int(itr / (employee_count / \
                                           self.scouts_count))]
            self.employees.append(EmployeeBee(scout))


    def __init_onlookers(self):
        '''Instantiate Onlooker Bees *(assigning Employees occurs after \
        evaluation and probability calculation)*
        '''

        onlooker_count = self.colony_size - int(self.colony_size * self.eo_colony_ratio)

        for itr in range(onlooker_count):
            self.onlookers.append(OnlookerBee())


    def __employee_bee_phase(self, itr):
        '''Evaluate Scout-initialized position after :func:`~core.abc.scout_bee.ScoutBee.reset` \
        or Search + Evaluate neighbor every subsequent iteration \
        until abandonment limit

        Args:
            itr (int): current main ABC optimization iteration
        '''

        # Search and evaluate new or existing neighbor
        for employee in self.employees:
            # Ignored if unevaluated position exists (i.e Scout-sampled)
            employee.search(self.obj_interface)

            fs = employee.food_source

            if fs is None or fs.encode_position() not in self.results_df['candidate']:
                # Evaluate employee position
                series = employee.evaluate(self.obj_interface, itr)
                self.__save_results(series)
            else:
                # Already evaluated
                fs.fitness = self.results_df[self.results_df['candidate'] == fs.encode_position()]['fitness'].values[0]
                employee.greedy_select(fs, self.obj_interface.is_minimize)
                # resampling the same candidate should count as a trial towards the abandonment limit?
                # onlooker.employee.trials += 1     # to avoid being stuck


    def __onlooker_bee_phase(self, itr):
        '''Assign each Onlooker to an Employee, then Search + Evaluate a random neighbor

        Args:
            itr (int): current main ABC optimization iteration
        '''

        # Calculate Employee probability
        sum_fitness = sum([employee.calculate_fitness() \
                        for employee in self.employees])

        # Sum(probabilities) ≈ 1.0
        probabilities = list(map(lambda employee: \
                                 employee.compute_probability(sum_fitness), \
                                 self.employees))

        if not self.obj_interface.is_minimize:
            # inverse weights to maximize the objective
            # (assign more onlookers to higher fitness scores)

            # reciprocals
            probabilities = [1.0 / prob for prob in probabilities]

            # normalize
            sum_probs = sum(probabilities)
            probabilities = [prob / sum_probs for prob in probabilities]

        # Assign EmployeeBees to OnlookerBees, search, and evaluate neighbor
        for onlooker in self.onlookers:
            # Assign EmployeeBees to OnlookerBees
            emp_idx = np.random.choice(len(self.employees), p=probabilities)

            onlooker.assign_employee(self.employees[emp_idx])

            # Search for a new random neighbor
            onlooker.search(self.obj_interface)

            fs = onlooker.food_source

            if fs is None or fs.encode_position() not in self.results_df['candidate']:
                # Evaluate employee position
                series = onlooker.evaluate(self.obj_interface, itr)
                self.__save_results(series)
            else:
                # Already evaluated
                fs.fitness = self.results_df[self.results_df['candidate'] == fs.encode_position()]['fitness'].values[0]
                onlooker.employee.greedy_select(fs, self.obj_interface.is_minimize)
                # resampling the same candidate should count as a trial towards the abandonment limit?
                # onlooker.employee.trials += 1     # to avoid being stuck


    def __scout_bee_phase(self):
        '''Check abandonment limits and rest employees accordingly
        '''

        ScoutBee.check_abandonment(self.employees, self.obj_interface)


    def __save_results(self, series):
        '''Save results dataframe to specified disk path

        Args:
            series (:class:`pandas.Series): Pandas Series (row) to be appended to the \
            current :code:`results_df`
        '''

        if self.total_evals % Params['RESULTS_SAVE_FREQUENCY'] == 0:
            self.results_df = self.results_df.append(series, ignore_index=True)

            filename = f'{Params["CONFIG_VERSION"]}.csv'
            if FileHandler.save_df(self.results_df,
                                   Params.get_results_path(),
                                   filename):
                Logger.filesave_log(series['candidate'], series['weights_filename'])


    def __momentum_phase(self):
        '''Momentum Evaluation Augmentation Stochastic operator that adds \
        :code:`Params['MOMENTUM_EPOCHS']` epochs to propel the evaluations of \
        the most consistently converging candidates
        '''


        # calculate probabilities
        calculated_momentums = self.results_df['momentum'] / (self.results_df['epochs'] + \
                                                              self.results_df['momentum_epochs'])
        if sum(calculated_momentums) == 0:
            # improbable edge case where the sum of momentums is exactly 0
            return

        probs = calculated_momentums / sum(calculated_momentums)

        the_chosen_ones = dict.fromkeys([x for x in range(len(probs))], 0)

        for _ in range(Params['MOMENTUM_EPOCHS']):
            # probabilistically assign momentum epochs
            idx = np.random.choice(len(probs), p=probs)
            the_chosen_ones[idx] += 1

        # training extension loop
        for the_one, m_epochs in the_chosen_ones.items():
            candidate_row = self.results_df.iloc[[the_one]]

            # extract candidate info for additional training
            candidate = candidate_row['candidate'].values[0]
            weights_file = candidate_row['weights_filename'].values[0]
            momentum = candidate_row['momentum'].values[0]
            epochs = candidate_row['epochs'].values[0]
            momentum_epochs = candidate_row['momentum_epochs'].values[0] + m_epochs

            # train for m_epochs
            Logger.momentum_evaluation_log(candidate,
                                           candidate_row['fitness'].values[0],
                                           m_epochs)

            res = self.obj_interface.momentum_eval(candidate,
                                                   weights_file,
                                                   m_epochs)

            # save new results
            self.results_df.loc[the_one, 'fitness'] = res['fitness']
            self.results_df.loc[the_one, 'momentum_epochs'] = momentum_epochs


    def __reset_all(self):
        ''' Resets the ABC algorithm
        '''

        self.scouts = []            # List of FoodSources initially sampled
        self.employees = []
        self.onlookers = []

        EmployeeBee.id_tracker = 0
        OnlookerBee.id_tracker = 0

        # init results dataframe
        filename = f'_{Params["CONFIG_VERSION"]}.csv'
        results_file = os.path.join(Params.get_results_path(), filename)

        # resume from previously saved file if it exists
        if Params['RESUME_FROM_RESULTS_FILE']:
            self.results_df = FileHandler.load_df(results_file)     # loads empty df if file not found

        else:
            cols = ['bee_type'
            'bee_id',
            'bee_parent',
            'itr',
            'candidate',
            'fitness',
            'center_fitness',
            'epochs',
            'params',
            'weights_filename',
            'time']

            self.results_df = pd.DataFrame(columns=cols)

        self.total_evals = len(self.results_df.index)


    def optimize(self):
        ''' Main optimization loop
        '''

        Params.export_yaml(Params.get_results_path(),
                           f'{Params["CONFIG_VERSION"]}.yaml')

        Logger.start_log()

        self.__reset_all()
        self.__init_scouts()
        self.__init_employees()
        self.__init_onlookers()
        start_time = time.time()

        fitness_selector = max if not self.obj_interface.is_minimize else min

        ''' Optimization loop '''
        for itr in range(Params['ITERATIONS_COUNT']):
            self.__employee_bee_phase(itr)
            self.__onlooker_bee_phase(itr)
            self.__momentum_phase()
            self.__scout_bee_phase()

            best_fitness = fitness_selector(self.results_df['fitness'].tolist())

            if itr % 1 == 0:
                Logger.status(itr,
                          'Best fitness: {}, Total time (s): {}'.format(best_fitness,
                                                                        time.time() - start_time))

        Logger.end_log()


