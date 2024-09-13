import time
"""Main event logging methods
"""

class Logger:
    '''Wrapper for debug- and info-logging methods

    Attributes:
        EVALUATION_LOGGING (bool): determines whether to log evaluation data \
        *(toggle to manage logging clutter in case the data points' count is large \
        -- i.e Numerical Benchmarks, for instance)*
    '''

    __DEBUG_PREFIX  = 'DEBUG:'
    __STATUS_PREFIX = 'STATUS:'
    __EVAL_PREFIX = 'EVALUATION LOG:'
    __MOMENTUM_PREFIX = 'MOMENTUM EVAL:'
    __FILESAVE_PREFIX = 'FILE-SAVE SUCCESSFUL:'

    __START_TIME = None

    EVALUATION_LOGGING = False


    @staticmethod
    def debug(msg=None):
        '''Debugging messages

        Args:
            msg (str, optional): debug message, defaults to \
            "MARK" to indicate whether the statement is reached
        '''

        print('{} {}'.format(Logger.__DEBUG_PREFIX,
                            ('MARK' if msg is None else str(msg))))


    @staticmethod
    def status(itr, msg=None):
        '''Generic logging

        Args:
            itr (int): current optimization iteration
            msg (str, optional): status message, defaults to \
            "MARK" to indicate whether the statement is reached
        '''

        print('{} itr: {} -- {}'.format(Logger.__STATUS_PREFIX,
                                        str(itr),
                                        ('MARK' if msg is None else str(msg))))


    @staticmethod
    def evaluation_log(type, id, candidate_pos):
        '''Logs pre-evaluation info for every candidate

        Args:
            type (str): bee type (Employee/Onlooker)
            id (int): bee ID
            candidate_pos (str): candidate position on the solution surface \
            (the string-encoded architecture in the case of NAS)
        '''

        if not Logger.EVALUATION_LOGGING:
            return

        print('\n{} {} ID ({}) -- Candidate ({})\n'.format(Logger.__EVAL_PREFIX,
                                                           type,
                                                           str(id),
                                                           str(candidate_pos)))


    @staticmethod
    def momentum_evaluation_log(candidate, fitness, epochs):
        '''Logs momentum evaluation augmentation info

        Args:
            candidate (str): candidate string representation \
            (architecture in the case of NAS)
            fitness (float): fitness value
            epochs (int): number of additional momentum epochs assigned
        '''

        if not Logger.EVALUATION_LOGGING:
            return

        print('{} Extending ({} - fitness: {}) by {} epochs...\n'.format(Logger.__MOMENTUM_PREFIX,
                                                                         str(candidate),
                                                                         fitness,
                                                                         epochs))


    @staticmethod
    def filesave_log(candidate, filename):
        '''Logs candidate info upon file-save

        Args:
            candidate (str): candidate string representation \
            (architecture in the case of NAS)
            filename (str): output filename
        '''

        if not Logger.EVALUATION_LOGGING:
            return

        print('\n{} Candidate ({}) was saved to {}\n'.format(Logger.__FILESAVE_PREFIX,
                                                             str(candidate),
                                                             filename))


    @staticmethod
    def start_log():
        '''Logs the start msg and intializes the global timer
        '''

        Logger.__START_TIME = time.time()
        dashes = '------------------------'
        print('{}\n-- OPTIMIZATION START --\n{}'.format(dashes, dashes))


    @staticmethod
    def end_log():
        '''Logs total time taken upon optimization end
        '''

        end_time = time.time() - Logger.__START_TIME
        dashes = '---------------------'
        print('{}\n-- OPTIMIZATION END --\n{} === TOTAL TIME TAKEN: {} ==== \n'.format(dashes, dashes, end_time))


