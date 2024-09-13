from .numericalbenchmark_base_class import NumericalBenchmark
import numpy as np
"""Shere optimization benchmark
"""

class Sphere(NumericalBenchmark):
    '''Sphere optimization benchmark as given by [ Σ Xi^2 , for i=1 in dim ]
    '''

    def __init__(self, dim, is_minimization=True):
        '''Initialize the benchmark

        Args:
            dim (int): number of dimensions / position components
            is_minimization (bool, optional): determines whether to minimize \
            or maximize the sphere; defaults to minimization
        '''

        super(Sphere, self).__init__(dim, -100.0, 100.0, is_minimization)


    def evaluate(self, pos):
        '''Evaluate a given position

        Args:
            pos (list): list of position components (floats)

        Returns:
            dict: fitness value of the given position along with the static data required by \
            :class:`~core.objective_interface.ObjectiveInterface`
        '''

        return {
            'fitness': sum(np.power(pos, 2)),
            'epochs': 1,
            'filename': '',
            'params': self.dim,
            'momentum': {'': ('', '')}
        }


    def momentum_eval(self, pos, weights, m_epochs):
        '''Redundant implementation as :func:`~benchmarks.sphere.Sphere.evaluate` \
        to satisfy the :class:`~core.objective_interface.ObjectiveInterface` hooks

        Args:
            pos (list): list of position components (floats)
            weights (str): N/A
            m_epochs (int): N/A

        Returns:
            dict: a dictionary containing the evaluated fitness value
        '''

        return {
            'fitness': sum(np.power(pos, 2))
        }


