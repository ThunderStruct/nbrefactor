import numpy as np
from ..objective_interface import ObjectiveInterface
"""Abstract :class:`~core.objective_interface.ObjectiveInterface` class for \
numerical benchmarks
"""

class NumericalBenchmark(ObjectiveInterface):
    '''Abstract class for Numerical Optimization Benchmarks

    Attributes:
        dim (int): number of dimensions for the given benchmark
        maxv (float): maximum value in the search space
        minv (float): minimum value in the search space
        minimization (bool): determines whether this is a minimization \
        or maximization problem
    '''

    def __init__(self, dim, minv, maxv, minimization):
        '''Initialize the numerical benchmark

        Args:
            dim (int): number of dimensions for the given benchmark
            maxv (float): maximum value in the search space
            minv (float): minimum value in the search space
            minimization (bool): determines whether this is a minimization \
            or maximization problem
        '''

        self.dim = dim
        self.minv = minv
        self.maxv = maxv
        self.minimization = minimization


    def sample(self):
        '''Samples a random point from the objective function

        Returns:
            float: randomly sample a candidate from the given space
        '''

        return np.random.uniform(low=self.minv, high=self.maxv, \
                                 size=self.dim)


    def get_neighbor(self, pos):
        '''Finds a random neighbor by displacing 1 positional component by :math:`\\phi`

        Args:
            pos (list): list of positional components (floats)

        Returns:
            list: a neighboring position
        '''

        op = np.random.choice(pos)
        phi = np.random.uniform(low=-1, high=1, size=len(pos))
        neighbor_pos = pos + (pos - op) * phi

        return self.__eval_boundary(neighbor_pos)


    def __eval_boundary(self, n_pos):
        '''Ensures the newly sampled position (neighbor) lies within the
        search space boundaries; if it is beyond the boundary, it snaps \
        to the edge

        Args:
            n_pos (list): the new position to be evaluated (list of floats)

        Returns:
            bool: whether or not the new position is within boundaries
        '''

        if (n_pos < self.minv).any() or \
        (n_pos > self.maxv).any():
            n_pos[n_pos < self.minv] = self.minv
            n_pos[n_pos > self.maxv] = self.maxv

        return n_pos


    @property
    def minimum(self):
        """The minimum value getter

        Returns:
            float: minimum value for a position component
        """

        return self.minv


    @property
    def maximum(self):
        """The maximum value getter

        Returns:
            float: maximum value for a position component
        """

        return self.maxv


    @property
    def is_minimize(self):
        """Minimization toggle getter

        Returns:
            bool: determines whether this is a minimization or \
            maximization problem
        """

        return self.minimization


    def fully_train_best_model(self, from_arch: bool=True):
        '''Exception handler for :code:`fully_train_best_model` on NumericalBenchmarks

        Raises:
            :class:`ValueError`: raised when :code:`fully_train_best_model` is called on a NumericalBenchmark

        Args:
            from_arch (bool, optional): N/A
        '''

        raise ValueError('\'ObjectiveInterface.fully_train_best_model()\' \
        called for a Numerical Benchmark')


