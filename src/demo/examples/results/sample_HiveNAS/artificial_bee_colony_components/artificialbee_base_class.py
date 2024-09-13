from abc import ABC
from abc import abstractmethod
from ..objective_interface import ObjectiveInterface
from .foodsource import FoodSource
"""Abstract definitions of the
:class:`~core.abc.employee_bee.EmployeeBee` and
:class:`~core.abc.onlooker_bee.OnlookerBee` methods
"""

class ArtificialBee(ABC):
    '''Abstract class for Employee & Onlooker Bees

    Attributes:
        food_source (:class:`~core.abc.food_sourec.FoodSource`): the bee's main \
        food source
        id (int): the bee's ID, used for logging/tracking purposes
    '''

    def __init__(self, food_source, id):
        '''Initialize an Artificial Bee

        Args:
            food_source (:class:`~core.abc.food_sourec.FoodSource`): the bee's \
            main food source
            id (int): the bee's ID, used for logging/tracking purposes
        '''

        self.food_source = food_source
        self.id = id if id is not None else -1


    def get_random_neighbor(self, obj_interface: ObjectiveInterface):
        '''
            Finds a random neighbor in the vicinity of the parent
            Parent(Onlooker) = Employee,
            Parent(Employee) = Scout
            Given by [1]:
                Xmi = Li + rand(0, 1) ∗ (Ui − Li)   => Initial FoodSource
                                                       (Scout)
                υmi = Xmi + ϕmi(Xmi − Xki)          => Neighboring FoodSource
                                                       (Employee/Onlooker)
                Where υmi is a neighboring FoodSource
                (definition of "neighboring" given in [2];

                TLDR - in numerical and continuous optimization problems,
                a dimensional component is incremented/decremented.
                In NAS context, it is a 1-operation difference per network)

            [1] Karaboga, D., & Basturk, B. (2007). A powerful and efficient
            algorithm for numerical function optimization: artificial bee
            colony (ABC) algorithm. Journal of global optimization, 39(3),
            459-471.
            [2] White, C., Nolen, S., & Savani, Y. (2021, December).
            Exploring the loss landscape in neural architecture search.
            In Uncertainty in Artificial Intelligence (pp. 654-664). PMLR.


        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface \
            used to sample candidates

        Returns:
            :class:`~core.abc.food_source.FoodSource`: a randomly sampled neighboring food source
        '''

        pos = self.get_center_fs().position
        neighbor_pos = obj_interface.get_neighbor(pos)

        return FoodSource(neighbor_pos)


    def is_evaluated(self):
        '''
        Checks if food source is evaluated for solution tracking purposes

        Returns:
            bool: whether or not the current :class:`~core.abc.food_source.FoodSource` \
            is evaluated
        '''

        return self.food_source is not None and \
        self.food_source.fitness is not None


    @abstractmethod
    def get_center_fs(self):
        '''Returns the center food source

        Returns:
            :class:`~core.abc.food_source.FoodSource`: the employee's center food source

        Raises:
            :class:`NotImplementedError`: must be implemented by the child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def evaluate(self, obj_interface: ObjectiveInterface):
        '''Evaluates the current :class:`core.abc.food_source.FoodSource`

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface \
            used to sample/evaluate candidates

        Returns:
            :class:`pandas.Series`: a Pandas Series containing the evaluation's results (represents a \
            row in the main results CSV file)

        Raises:
            :class:`NotImplementedError`: must be implemented by the child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def search(self, obj_interface: ObjectiveInterface):
        """Explore new random position (near previously-sampled position) and assigns it to the \
        current food source

        Args:
            obj_interface (:class:`~core.objective_interface.ObjectiveInterface`): the objective interface \
            used to sample/evaluate candidates

        Raises:
            :class:`NotImplementedError`: must be implemented by the child class
        """

        raise NotImplementedError()


    def __repr__(self):
        '''For logging/debugging purposes

        Returns:
            str: the pretty-print contents of the Employee/Onlooker Bee
        '''

        return str(self)


