from abc import ABC
from abc import abstractmethod

"""The abstract definition of Objective Interfaces in HiveNAS
"""

class ObjectiveInterface(ABC):

    '''
        Encapsulates the method definitions required to satisfy the hooks
        used by the :class:`~core.abc.abc.ArtificialBeeColony` optimizer
    '''


    @abstractmethod
    def sample(self):
        '''Samples a random candidate from the optimization surface
        *(used primarily by* :class:`~core.abc.scout_bee.ScoutBee` *to initialize the* :class:`~core.abc.food_source.FoodSource`
        *vector,* Xm→ *)*

        Returns:
            str: a string-encoded candidate randomly sampled from the solution space

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def evaluate(self, candidate: str):
        '''Evaluates a given string-encoded candidate and returns its fitness score

        Args:
            candidate (str): string-encoded candidate (an architecture in the case of \
            :class:`~core.nas.nas_interface.NASInterface`)

        Returns:
            float: the candidate's fitness score

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def get_neighbor(self, candidate: str):
        '''Samples a neighbor for a given string-encoded candidate

        Args:
            candidate (str): the position on the solution surface to find a neighbor for

        Returns:
            str: the neighboring string-encoded candidate

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def fully_train_best_model(self, from_arch: bool=True):
        '''Fully trains the best solution found thus far (exclusively used by NAS)
        (relies on paths set in :class:`~config.params.Params`)

        Args:
            from_arch (bool, optional): determines whether to train model from scratch \
            using the string representations of the architecture (:code:`from_arch = True`) \
            or load the saved model file and continue training (:code:`from_arch = False`). \
            \
            \
            `Note: optimizer settings are typically not saved, \
            therefore training continuation from a model's file can result in a worse overall accuracy` \
            (`read more... <https://stackoverflow.com/a/58693088/3551916>`_).

        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, \
            number of training epochs conducted (not including any previous trainings), \
            hashed file name, number of trainable parameters

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def momentum_eval(self,
                      candidate: str,
                      weights_filename: str,
                      m_epochs: int):
        '''Momentum Evaluation phase (:class:`~core.nas.momentum_eval.MomentumAugmentation`; used exclusively by NAS)

        Args:
            candidate (str): the selected string-encoded candidate to extend its training
            weights_filename (str): the SHA1-hashed unique string ID for the given candidate
            m_epochs (int): the additional momentum epochs the candidate should be trained for


        Returns:
            dict: final fitness value (accuracy) after training continuation

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


    @abstractmethod
    def is_minimize(self):
        '''Used by the optimization algorithm to determine whether this is
        a minimization or maximization problem

        Returns:
            bool: whether to minimize or maximize the fitness (:code:`True` = minimize)

        Raises:
            :class:`NotImplementedError`: requires implementation by child class
        '''

        raise NotImplementedError()


