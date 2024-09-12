import argparse

# Argument Parsing

class Params:

    PARSER = None
    ARGS = None

    @staticmethod
    def initialize():

        Params.PARSER = argparse.ArgumentParser()
        Params.__set_args()

        Params.ARGS = Params.PARSER.parse_args(args=[])


    @staticmethod
    def __set_args():
        """
        """

        Params.PARSER.add_argument('--num-vertices', '-nv',
                                   dest='num_vertices',
                                   nargs='?',
                                   const=8,
                                   default=8,
                                   type=int,
                                   choices=range(1, 10),
                                   help=(
                                       'Number of vertices in generated '
                                       'architectures'
                                       )
        )

        Params.PARSER.add_argument('--arch-encoding', '-ae',
                                   dest='arch_encoding',
                                   nargs='?',
                                   const='multi-branch',
                                   default='multi-branch',
                                   type=str,
                                   choices=['single-branch', 'multi-branch'],
                                   help=(
                                      'Search space encoding; '
                                      'single-branch = path-based encoding, '
                                      'multi-branch = adjacency-matrix encoding'
                                      )
        )

        Params.PARSER.add_argument('--nas-epochs', '-ne',
                                   dest='nas_epochs',
                                   nargs='?',
                                   const=50,
                                   default=50,
                                   type=int,
                                   choices=range(1, 150),
                                   help=(
                                       'Number of epochs for the NAS '
                                       'optimization loop (# of evaluated '
                                       'architectures)'
                                       )
        )

        Params.PARSER.add_argument('--train-epochs', '-te',
                                   dest='train_epochs',
                                   nargs='?',
                                   const=5,
                                   default=5,
                                   type=int,
                                   choices=range(1, 200),
                                   help=(
                                       'Number of training epochs per '
                                       'architecture'
                                       )
        )

        Params.PARSER.add_argument('--learning-rate', '-lr',
                                   dest='learning_rate',
                                   nargs='?',
                                   const=0.001,
                                   default=0.001,
                                   type=float,
                                   choices=range(0, 1),
                                   help=(
                                       'Learing rate for generated '
                                       'architectures'
                                       )
        )

        Params.PARSER.add_argument('--results-filepath', '-rf',
                                   dest='results_filepath',
                                   type=str,
                                   nargs='?',
                                   default='./results/',
                                   const='./results/',
                                   help=(
                                       'Filepath and filename to save the NAS '
                                       'results to'
                                       )
        )

        Params.PARSER.add_argument('--save-training-logs', '-stl',
                                   dest='save_training_logs',
                                   nargs='?',
                                   const=True,
                                   default=True,
                                   type=bool,
                                   help=(
                                       'Sets whether or not to save training '
                                       'history log files'
                                       )
        )

        Params.PARSER.add_argument('--verbose', '-v',
                                   dest='verbose',
                                   nargs='?',
                                   const=True,
                                   default=True,
                                   type=bool,
                                   help=(
                                       'Sets whether or not to log debug and '
                                       'warning details'
                                       )
        )


    def get_args(*args):
        """
        """

        if not args or len(args) == 0:
            return vars(Params.ARGS)

        ret_dict = {}

        for key, val in vars(Params.ARGS).items():
            if key in args:
                ret_dict[key] = val

        return ret_dict


Params.initialize()

