from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SeparableConv2D
from .prompt_handler import PromptHandler
from tensorflow.keras.layers import MaxPooling2D
import yaml
from tensorflow.keras.layers import AveragePooling2D
from .file_handler import FileHandler
from tensorflow.keras.layers import Flatten
import os
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from functools import partial
from tensorflow.keras.layers import Conv2D
"""All operational parameters used by HiveNAS and configuration methods
"""

class Params:
    '''Wrapper for all global operational parameters
    and the configuration loader
    '''

    @staticmethod
    def config_form():
        '''Facilitates the configuration UI form (for Google Colab) \
        and exports all parameters as a dictionary

        Returns:
            dict: main global parameters dictionary (locals)
        '''


        ''' Configuration Version (used as filenames) '''
        CONFIG_VERSION = 'hivenas_v2_res'   #@param {type:"string"}


        #@markdown ## ABC Optimizer Parameters
        #@markdown ---


        ''' Optimization problem (NAS or Numerical Benchmarks to test ABC) '''
        OPTIMIZATION_OBJECTIVE = 'NAS'  #@param ['NAS', 'Sphere_max', 'Sphere_min', 'Rosenbrock']

        ''' Max trials per Scout (i.e initial Food Source) '''
        ABANDONMENT_LIMIT = 3  #@param {type:"slider", min:1, max:50, step:1}

        ''' Number of bees in the colony (Employees + Onlookers) '''
        COLONY_SIZE = 7    #@param {type:"slider", min:1, max:50, step:1}

        ''' Distribution of Employees to Onlookers, resulting number of EmployeeBees = # of ScoutBees '''
        EMPLOYEE_ONLOOKER_RATIO = 0.43   #@param {type:"slider", min:0.1, max:1.0, step:0.05}

        ''' Number of ABC optimization iterations '''
        ITERATIONS_COUNT = 12    #@param {type:"slider", min:1, max:100, step:1}


        #@markdown \
        #@markdown ## File-Handling Parameters
        #@markdown ---


        ''' Save results every N evaluations (not iterations; iterations * colony_size) '''
        RESULTS_SAVE_FREQUENCY = 1   #@param {type:"slider", min:1, max:100, step:1}

        '''
            Result files base path (path will be created if it does not exist)
            A local folder will be created after the CONFIG_VERSION
        '''
        RESULTS_BASE_PATH = '/content/gdrive/MyDrive/PhD/HiveNAS/'  #@param {type:"string"}

        ''' Training history files sub-path '''
        HISTORY_FILES_SUBPATH = 'training_history/'     #@param {type:"string"}

        ''' Enable weights saving for resumed training '''
        ENABLE_WEIGHT_SAVING = False    #@param {type:"boolean"}

        ''' Weight files sub-path (ensure that the path exists) '''
        WEIGHT_FILES_SUBPATH = 'weights/'    #@param {type:"string"}

        ''' Specifies whether or not to resume from existing results file (if exists)'''
        RESUME_FROM_RESULTS_FILE = False   #@param {type:'boolean'}


        #@markdown \
        #@markdown ## NAS Search Space Parameters
        #@markdown ---


        ''' -- NAS Search Space configuration -- '''

        #@markdown *( layers & hyperparameters must be defined as partial functions in code )*

        ''' Number of layers for sampled networks (excludes input/output stems) '''
        DEPTH = 7     #@param {type:"slider", min:1, max:10, step:1}

        ''' Search space operations '''
        OPERATIONS = {
            'search_space': {
                'sep5x5_128': partial(SeparableConv2D, filters=128, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_128': partial(SeparableConv2D, filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_64': partial(SeparableConv2D, filters=64, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_64': partial(SeparableConv2D, filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_32': partial(SeparableConv2D, filters=32, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_32': partial(SeparableConv2D, filters=32, kernel_size=(3,3), activation='relu', padding='same'),
                'max_pool3x3': partial(MaxPooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'avg_pool3x3': partial(AveragePooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'dropout': partial(Dropout, rate=0.15),
                'identity': partial(Activation, 'linear')
            },
            'reference_space': {
                'conv3x3_16': partial(Conv2D, filters=16, kernel_size=(3,3), activation='relu'),
                'sep5x5_128': partial(SeparableConv2D, filters=128, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_128': partial(SeparableConv2D, filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_64': partial(SeparableConv2D, filters=64, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_64': partial(SeparableConv2D, filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_32': partial(SeparableConv2D, filters=32, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_32': partial(SeparableConv2D, filters=32, kernel_size=(3,3), activation='relu', padding='same'),
                'max_pool3x3': partial(MaxPooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'avg_pool3x3': partial(AveragePooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'batch_norm': partial(BatchNormalization),
                'dropout': partial(Dropout, rate=0.15),
                'identity': partial(Activation, 'linear')
            }
        }

        # Skip-Connections'/Residual Blocks' occurence rate (0.0 = disabled)
        RESIDUAL_BLOCKS_RATE = 0.15    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        CLASSICAL_RESNETS = True    #@param {type:"boolean"}

        # Squeeze-Excitation blocks ratio
        SE_RATIO = 0.0  #@param {type:"slider", min:0.0, max:0.25, step:0.25}


        #@markdown \
        #@markdown ## NAS Evaluation Strategy Parameters
        #@markdown ---


        ''' -- NAS Evaluation Strategy configuration -- '''

        ''' Dataset (classes/inputs are inferred internally) '''
        DATASET = 'CIFAR10'   #@param ["CIFAR10", "MNIST", "FASHION_MNIST"]

        ''' Static output stem, added to every candidate '''
        OUTPUT_STEM = [
            partial(Flatten),
            partial(Dropout, rate=0.15),
            partial(Dense, units=1024, activation='relu'),
            partial(Dropout, rate=0.15),
            partial(Dense, units=512, activation='relu')
        ]

        ''' Static input stem, added to every candidate '''
        INPUT_STEM = [
            # partial(Conv2D, filters=32, kernel_size=(3,3)),
            # partial(BatchNormalization),
            # partial(ReLU)
        ]

        ''' Epochs count per candidate network '''
        EPOCHS = 5  #@param {type:"slider", min:1, max:25, step:1}

        ''' Momentum Augmentation epochs (0 = disabled ; overrides ENABLE_WEIGHT_SAVING) '''
        MOMENTUM_EPOCHS = 0 #@param {type:"slider", min:0, max:25}

        ''' Epochs count for the best performing candidate upon full training '''
        FULL_TRAIN_EPOCHS = 50 #@param {type:"slider", min:1, max:150, step:1}

        '''
            Threshold factor (beta) for early-stopping (refer to the TerminateOnThreshold class for details)
                1.0 = all networks will be terminated (minimum accuracy = 100%)
                0.0 = disable early-stopping, all networks will pass
                0.25 = for 10 classes, val_acc > 0.325 at epoch 1 will not be terminated
                       (tolerance decreased for every subsequent epoch)
        '''
        TERMINATION_THRESHOLD_FACTOR = 0.0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Diminishing factor (zeta) for termination threshold over epochs '''
        TERMINATION_DIMINISHING_FACTOR = 0.25 #@param {type:"slider", min:0.1, max:1.0, step:0.05}

        ''' Learning rate (overrides default optimizer lr) '''
        LR = 0.001  #@param {type:"slider", min:0.001, max:0.1, step:0.001}

        ''' Batch size for every candidate evaluation '''
        BATCH_SIZE = 128     #@param {type:"slider", min:8, max:256, step:2}

        ''' Optimizer used for both NAS and full-training methods '''
        OPTIMIZER = 'Adam'    #@param ["Adam", "RMSprop"]


        #@markdown \
        #@markdown ## Data Augmentation Parameters
        #@markdown ---


        '''
            Enable affine transformations augmentation
            (horizontal/vertical shifts, rotation, etc...)
        '''

        AFFINE_TRANSFORMATIONS_ENABLED = True   #@param {type:"boolean"}

        ''' Probability of random cutout augmentation occurence (0.0 = disabled) '''
        CUTOUT_PROB = 0.8    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Probability of random saturation augmentation occurence (0.0 = disabled) '''
        SATURATION_AUG_PROB = 0.75    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Probability of random contrast augmentation occurence (0.0 = disabled) '''
        CONTRAST_AUG_PROB = 0.75    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        return locals()


    ''' Main configuration dict '''
    __CONFIG = config_form.__func__()


    @staticmethod
    def init_from_yaml(path):
        '''Initializes the global parameters from a given yaml config file

        Args:
            path (str): path to yaml configuration file

        '''


        def param_op_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
            '''Constructs NAS Search Space operations (using the custom \
            yaml :code:`!Operation` tag)

            Args:
                loader (:class:`yaml.SafeLoader`): yaml default safe loader
                node (:class:`yaml.nodes.MappingNode`): yaml mapping node

            Returns:
                :class:`functools.partial`: partial function containing the neural operation
            '''

            # constructs an operation partial function from yaml !Operation tags
            op_dict = loader.construct_mapping(node)
            op = op_dict['op']
            del op_dict['op']

            return partial(globals()[op], **op_dict)


        def param_tuple_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
            '''Constructs a tuple from the standard \
            :code:`tag:yaml.org,2002:python/tuple` (:code:`!!python/tuple`) yaml tag

            Args:
                loader (:class:`yaml.SafeLoader`): yaml default safe loader
                node (:class:`yaml.nodes.MappingNode`): yaml mapping node

            Returns:
                tuple: constructed tuple, \
                typically used to define kernel sizes/shapes in yaml
            '''

            # because for some reason we need an explicit tuple constructor

            return tuple(loader.construct_sequence(node))

        # register constructors
        loader = yaml.SafeLoader
        loader.add_constructor(u'tag:yaml.org,2002:python/tuple', param_tuple_constructor)
        loader.add_constructor('!Operation', param_op_constructor)

        config = FileHandler.load_yaml(path, loader)

        if not config:
            print(f'\nConfig file ({path}) is either invalid or does not exist.\n\n')
            return

        for key,val in config.items():
            if key not in Params.__CONFIG:
                # ensure config file keys are valid and match the hard-coded template
                print(f'\nConfig file ({path}) is invalid. Skipping item ({key})... \n\n')
                continue

            Params.__CONFIG[key] = val

        print(f'\nSuccessfully loaded the operational parameters from {path}.\n\n')


    @staticmethod
    def export_yaml(path, filename, from_formdata=False):
        '''Saves the current configurations to the given path as yaml

        Args:
            path (str): output path to save the yaml config file to
            filename (str): output file name
            from_formdata (bool, optional): determines whether the export instruction \
            originated from the Google Colab UI form or called in code. *When it originates \
            from the form, data reload is required to ensure consistency (could be \
            altered within the form)*
        '''

        def param_op_representer(dumper, data):
            '''Serializes a partial function into the custom :code:`!Operation` \
            yaml tag

            Args:
                dumper (:class:`yaml.Dumper`): default pyyaml dumper
                data (partial): partial function data to be serialized

            Returns:
                :class:`yaml.nodes.MappingNode`: yaml mapping node representing \
                the operation
            '''

            # serialize partial functions into yaml !Operation
            serialized_data = {'op': data.func.__name__}
            serialized_data.update(data.keywords)

            return dumper.represent_mapping('!Operation', serialized_data, flow_style=True)

        def param_tuple_representer(dumper, data):
            '''Serializes a tuple into the :code:`tag:yaml.org,2002:python/tuple` \
            (:code:`!!python/tuple`) yaml tag

            Args:
                dumper (:class:`yaml.Dumper`): default pyyaml dumper
                data (tuple): tuple data to be serialized

            Returns:
                :class:`yaml.nodes.MappingNode`: yaml mapping node representing \
                the tuple
            '''

            # serialize tuples into yaml !!python/tuple

            return dumper.represent_sequence(u'tag:yaml.org,2002:python/tuple', data, flow_style=True)

        # register representers
        yaml.add_representer(tuple, param_tuple_representer)
        yaml.add_representer(partial, param_op_representer)
        yaml.Dumper.ignore_aliases = lambda *args : True

        # data source (changing the Colab form does not reflect on the main dict)
        data = Params.config_form() if from_formdata else Params.__CONFIG

        if FileHandler.export_yaml(data,
                                   path,
                                   filename):
            print(f'\nConfiguration file saved successfully to ({os.path.join(path, filename)})!\n\n')
        else:
            print('\nFailed to save config file!\n\n')


    @staticmethod
    def search_space_config():
        '''Returns the search space config dict

        Returns:
            dict: dictionary containing the :class:`~core.nas.search_space.NASSearchSpace`-related \
            parameters
        '''

        res = {
            'depth': Params['DEPTH'],
            'operations': Params['OPERATIONS'],
            'residual_blocks_rate': Params['RESIDUAL_BLOCKS_RATE']
        }

        return res


    @staticmethod
    def evaluation_strategy_config():
        '''Returns the evaluation strategy config dict

        Returns:
            dict: dictionary containing the :class:`~core.nas.evaluation_strategy.NASEval`-related \
            parameters
        '''

        res = {
            'dataset': Params['DATASET'],
            'operations': Params['OPERATIONS'],
            'output_stem': Params['OUTPUT_STEM'],
            'input_stem': Params['INPUT_STEM'],
            'epochs': Params['EPOCHS'],
            'full_train_epochs': Params['FULL_TRAIN_EPOCHS'],
            'lr': Params['LR'],
            'batch_size': Params['BATCH_SIZE'],
            'optimizer': globals()[Params['OPTIMIZER']],
            'termination_threshold_factor': Params['TERMINATION_THRESHOLD_FACTOR'],
            'termination_diminishing_factor': Params['TERMINATION_DIMINISHING_FACTOR'],
            'momentum_epochs': Params['MOMENTUM_EPOCHS']
        }

        return res


    @staticmethod
    def get_results_path():
        '''Gets the results path from :code:`RESULTS_BASE_PATH` and :code:`CONFIG_VERSION`

        Returns:
            str: the joined path to the results directory or :code:`None` if either \
            :code:`RESULTS_BASE_PATH` or :code:`CONFIG_VERSION` is invalid
        '''

        path = os.path.join(Params.__CONFIG['RESULTS_BASE_PATH'],
                            f'{Params.__CONFIG["CONFIG_VERSION"]}/')

        if FileHandler.validate_path(path):
            return path

        return None


    @staticmethod
    def get_all_config():
        '''Returns all operational parameters

        Returns:
            dict: returns the dict containing all configurations *(for \
            argparsing purposes)*
        '''

        return Params.__CONFIG


    @staticmethod
    def set_parameter(key, val):
        '''Overrides a default parameter (used by argparser)

        Args:
            key (str): dictionary key to select parameter
            val (any): new value to override default parameter
        '''

        if key not in Params.__CONFIG or not isinstance(val, type(Params.__CONFIG[key])):
            # invalid key or value type
            return False

        Params.__CONFIG[key] = val

        return True


    def __class_getitem__(cls, key):
        '''Subscript operator definition

        *Static class subscripting :code:`__class_getitem__` requires Python 3.7+*

        Used as :code:`Params['KEY']`

        Args:
            key (str): dictionary key to select parameter

        Returns:
            Any: subscripted parameter from the configuration dictionary
        '''

        return Params.__CONFIG[key]


''' Init widgets if ran on Colab '''
if ON_COLAB:
    PromptHandler.setup_ipy_widgets(on_yaml_export=lambda p, f: Params.export_yaml(p, f, True),
                                    on_yaml_init=lambda p: Params.init_from_yaml(p))

