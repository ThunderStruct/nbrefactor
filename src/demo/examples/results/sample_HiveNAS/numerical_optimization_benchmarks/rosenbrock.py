from scipy import optimize
from .numericalbenchmark_base_class import NumericalBenchmark
"""Rosenbrock optimization benchmark
"""

class Rosenbrock(NumericalBenchmark):
    ''' The benchmark uses the :class:`scipy.optimize.rosen` version of \
    Rosenbrock, as given by

    [ Σ { 100(Xi+1 - Xi)^2 + (Xi - 1)^2 } , for i=1 in dim - 1 ]

    '''

    def __init__(self, dim):
        '''Initializes the benchmark

        Args:
            dim (int): number of dimensions that constitute the position
        '''

        super(Rosenbrock, self).__init__(dim, -30.0, 30.0, True)


    def evaluate(self, pos):
        '''Evaluate the given position

        Args:
            pos (list): list of floats representing the position components

        Returns:
            dict: fitness value of the given position along with the static data required by \
            :class:`~core.objective_interface.ObjectiveInterface`
        '''

        return {
            'fitness': optimize.rosen(pos),
            'epochs': 1,
            'filename': '',
            'params': self.dim,
            'momentum': {'': ('', 0)}
        }


    def momentum_eval(self, pos, weights, m_epochs):
        '''Redundant implementation as :func:`~benchmarks.rosenbrock.Rosenbrock.evaluate` \
        to satisfy the :class:`~core.objective_interface.ObjectiveInterface` hooks

        Args:
            pos (list): list of position components (floats)
            weights (str): N/A
            m_epochs (int): N/A

        Returns:
            dict: a dictionary containing the evaluated fitness value
        '''

        return {
            'fitness': optimize.rosen(pos)
        }


