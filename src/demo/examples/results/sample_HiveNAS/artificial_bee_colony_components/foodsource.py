import time
"""The FoodSources representing positions on the optimization surface
used by the ABC optimizer as desribed in [1]. Can be considered the
*memory units* for Employee and Onlooker Bees

    [1] Karaboga, D., & Basturk, B. (2007). A powerful and efficient
    algorithm for numerical function optimization: artificial bee
    colony (ABC) algorithm. Journal of global optimization, 39(3),
    459-471.
"""

class FoodSource(object):
    '''The FoodSource class encapsulates the position on the optimization surface and its
    corresponding fitness value + the evaluation time taken (in seconds)

    Attributes:
        eval_time (float): the time taken to evaluate the given position (in seconds)
        pos (str): the string-encoded position on the optimization surface
        fit (float): the fitness corresponding the stored position
    '''

    def __init__(self, position=None, fitness=None):
        '''
        Data structure containing a FoodSource (position on the
        optimization surface) and its fitness value

        Args:
            position (str, optional): the string-encoded position on the optimization surface
            fitness (float, optional): the fitness corresponding the stored position
        '''

        self.pos = position
        self.fit = fitness
        self.eval_time = 0.0


    def encode_position(self):
        '''Returns an encoded position for use in dicts

        Returns:
            str: a formatted, whitespace-stripped version of the stored position. \
            Used as the "candidate" in the stored CSV
        '''

        return str(self.pos).replace(' ', '')


    # --- Setters & Getters --- #

    @property
    def position(self):
        '''The :code:`position` attribute getter

        Returns:
            str: the stored position
        '''

        return self.pos


    @property
    def fitness(self):
        '''The :code:`fitness` attribute getter

        Returns:
            float: the stored fitness value
        '''

        return self.fit


    @property
    def time(self):
        '''The :code:`eval_time` attribute getter

        Returns:
            float: the stored evaluation time value (in seconds)
        '''

        return self.eval_time


    @position.setter
    def position(self, value):
        '''The :code:`position` attribute setter

        Args:
            value (str): new position value to set
        '''

        self.pos = value


    @fitness.setter
    def fitness(self, value):
        '''The :code:`fittness` attribute setter

        Args:
            value (float): new fitness value to set
        '''

        self.fit = value


    @time.setter
    def time(self, value):
        '''The :code:`eval_time` attribute setter

        Args:
            value (float): new evaluation time value (in seconds) to set
        '''

        self.eval_time = value


    def __str__(self):
        '''For logging/debugging purposes

        Returns:
            str: the pretty-print contents of the FoodSource
        '''

        return 'position: {}, fitness: {}, evaluation time: {}'.format(self.pos,
                                                                       self.fit,
                                                                       self.eval_time)


    def __repr__(self):
        '''For logging/debugging purposes

        Returns:
            str: the pretty-print contents of the FoodSource (forced as str)
        '''

        return str(self)


