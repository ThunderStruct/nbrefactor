from .nas_components.nas_interface import NASInterface
from .numerical_optimization_benchmarks.sphere import Sphere
from .numerical_optimization_benchmarks.rosenbrock import Rosenbrock
from .helper_tools.operational_parameters import Params
from .helper_tools.logger import Logger
from .artificial_bee_colony_components.abc_optimizer import ArtificialBeeColony

"""Top-level module used to run HiveNAS
"""

class HiveNAS(object):
    '''Encapsulates all high level modules and runs the \
    ABC-based optimization
    '''

    @staticmethod
    def find_topology(evaluation_logging=True,
                      config_path=None,
                      kill_after=True):
        '''Runs the base NAS optimization loop

        Args:
            evaluation_logging (bool, optional): determines whether to log \
            evaluation info or not; defaults to :code:`True`
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.EVALUATION_LOGGING = evaluation_logging

        if Params['OPTIMIZATION_OBJECTIVE'] == 'NAS':
            objective_interface = NASInterface()
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_min':
            objective_interface = Sphere(10)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_max':
            objective_interface = Sphere(10, False)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Rosenbrock':
            objective_interface = Rosenbrock(2)

        abc = ArtificialBeeColony(objective_interface)

        abc.optimize()

        if kill_after:
            # Disconnect runtime / free up GPU instance (mainly for Colab)
            pass # !kill -9 -1


    @staticmethod
    def fully_train_topology(config_path=None,
                             kill_after=True):
        '''Given the current configuration file,
        extract the best previously-found topology and fully-train it

        Args:
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.start_log()

        # loads architecture and optimizes its weights over a larger number of epochs;
        # from_arch sepcifies whether to re-instantiate the network and train from scratch
        # or resume training from weights file
        res = NASInterface().fully_train_best_model(from_arch=True)

        Logger.end_log()

        print(res)

        if kill_after:
            # Disconnect runtime / free up GPU instance
            pass # !kill -9 -1


# Run HiveNAS
HiveNAS.find_topology()

