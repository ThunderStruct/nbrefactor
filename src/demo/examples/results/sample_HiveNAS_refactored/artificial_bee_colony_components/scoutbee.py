from ..helper_tools.operational_parameters import Params
from .foodsource import FoodSource

"""ABC's Scout Bee module
"""

class ScoutBee:
    '''Scout Bees' static methods.

    ABC Scout Bees are generally classified as a *reset* operator
    for Employee/Onlooker Bees.

    Responsible for sampling the initial random FoodSources Xm→
    and reseting employee bees when the abandonment limit is reached
    '''


    @staticmethod
    def sample(obj_interface):
        '''Sample a random point from the objective function

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface that defines the boundaries of the problem

        Returns:
            :class:`~core.abc.food_source.FoodSource`: the randomly sampled initial position
        '''

        return FoodSource(obj_interface.sample())


    @staticmethod
    def check_abandonment(employees, obj_interface):
        '''Check if EmployeeBees reached abandonment limit

        Args:
            employees (list): a list of :class:`~core.abc.employee_bee.EmployeeBee` to check their trial count / abandonment limit
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface, needed to :func:`core.abc.scout_bee.ScoutBee.sample` a new position if the abandonment limit is reached
        '''

        for employee in employees:
            if employee.trials >= Params['ABANDONMENT_LIMIT']:
                # Reset EmployeeBees to a new ScoutBee position
                employee.reset(ScoutBee.sample(obj_interface))


