from tensorflow.keras import K as K
from .nas_search_space import NASSearchSpace
from .nas_evaluation_strategy import NASEval
from ..helper_tools.operational_parameters import Params
import pandas as pd
from ..helper_tools.file_handler import FileHandler
import gc
import os
from ..objective_interface import ObjectiveInterface
"""The NAS Interface encapsulating the Evaluation Strategy and Search Space
"""

class NASInterface(ObjectiveInterface):
    '''
    An interface that combines the Search Space & Evaluation Strategy
    for the NAS Search Algorithm (ABC)

    Attributes:
        cls.eval_strategy (:class:`~core.nas.evaluation_strategy.NASEval`): NASEval instance used to instantiate and evaluate candidates
        cls.search_space (:class:`~core.nas.search_space.NASSearchSpace`): NASSearchSpace instance used to sample candidates and neighbors
    '''

    def __init__(self,
                 space_config=None,
                 eval_config=None):
        '''Initializes the search space and evaluator

        Args:
            space_config (dict, optional): the predefined operational parameters pertaining to the search space (defined in :func:`~config.params.Params.search_space_config`)
            eval_config (dict, optional): the predefined operational parameters pertaining to evaluation (defined in :func:`~config.params.Params.evaluation_strategy_config`)
        '''

        space_config = space_config or Params.search_space_config()
        eval_config = eval_config or Params.evaluation_strategy_config()

        NASInterface.search_space = NASSearchSpace(space_config)
        NASInterface.eval_strategy = NASEval(eval_config)


    def sample(self):
        '''Samples new random candidate architecture from the search space

        Returns:
            str: string-encoded representation of the sampled candidate architecture
        '''

        return NASInterface.search_space.sample()


    def evaluate(self, candidate):
        '''Evaluates a given candidate architecture

        Args:
            candidate (str): string-encoded representation of the architecture to be evaluated

        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (in case of ACT), hashed file name, number of trainable parameters, and the last epoch's momentum value if applicable
        '''

        formatted = NASInterface.search_space.eval_format(candidate)
        res = NASInterface.eval_strategy.evaluate(formatted)

        # housekeeping
        K.clear_session()
        gc.collect()

        return res


    def get_neighbor(self, orig_arch):
        '''Returns a random architecture with 1 op diff to the given candidate

        Args:
            orig_arch (str): string-encoded representation of the candidate architecture

        Returns:
            str: string-encoded representation of the neighbor architecture
        '''

        return NASInterface.search_space.get_neighbor(orig_arch)


    def fully_train_best_model(self, from_arch=True):
        '''
        Fully-train best-performing model
        (relies on paths set in :class:`~config.params.Params`)

        Args:
            from_arch (bool, optional): determines whether to train model from scratch \
            using the string representations of the architecture (:code:`from_arch = True`) \
            or load the saved model file and continue training (:code:`from_arch = False`). \
            \
            `Note: optimizer settings are typically not saved, \
            therefore training continuation from a model's file can result in a worse overall accuracy` \
            (`read more... <https://stackoverflow.com/a/58693088/3551916>`_).

        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (not including any previous trainings), hashed file name, number of trainable parameters

        Raises:
            :class:`FileNotFoundError`: raises an error if the results file or model h5 file (when applicable) do not exist
        '''

        # check existence of results file
        filename = f'{Params["CONFIG_VERSION"]}.csv'
        results_file = os.path.join(Params.get_results_path(), filename)
        FileHandler.path_must_exist(results_file)    # breaks if file does not exist

        # extract best fitness weight file
        results_df = pd.read_csv(results_file, header=0, index_col=0)
        weight_file = results_df.loc[results_df['fitness'] == results_df['fitness'].max(), 'weights_filename'].values[0]
        arch = results_df.loc[results_df['fitness'] == results_df['fitness'].max(), 'candidate'].values[0]

        print(f'\nFound best-performing model {{{arch}}} with a fitness score of {results_df["fitness"].max()}\n')

        # housekeeping -> results_df no longer needed and is potentially large
        del results_df
        gc.collect()

        if from_arch:
            # Retrains from scratch given the network arch
            arch = formatted = NASInterface.search_space.eval_format(arch)
            return NASInterface.eval_strategy.fully_train(arch=arch)

        # check existence of weight file
        weight_file = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weight_file)
        FileHandler.path_must_exist(weight_file)    # breaks if file does not exist


        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.fully_train(model_file=weight_file)


    def momentum_eval(self, candidate, weights_filename, m_epochs):
        '''Trains a given network for additional :code:`m_epochs`

        Args:
            candidate (str): string-encoded representation of the candidate architecture
            weights_filename (str): the SHA1-hashed unique string ID for the given architecture
            m_epochs (int): the additional momentum epochs the candidate should be trained for

        Returns:
            dict: final fitness value (accuracy) after training continuation
        '''

        # check existence of weight file
        weights_path = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weights_filename)
        FileHandler.path_must_exist(weights_path)    # breaks if file does not exist


        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.momentum_training(weights_path, m_epochs)


    @property
    def is_minimize(self):
        '''
        Used by the optimization algorithm to determine whether this is
        a minimization or maximization problem

        Returns:
            bool: hard-coded :code:`False`; the search algorithm is always maximizing accuracy
        '''

        return False


