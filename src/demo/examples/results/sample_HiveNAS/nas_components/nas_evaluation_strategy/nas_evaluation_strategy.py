from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hashlib
from ...helper_tools.file_handler import FileHandler
from tensorflow.keras.layers import Input
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import K as K
from tensorflow.keras.models import load_model
import gc
from ...helper_tools.image_augmentations import ImgAug
import numpy as np
from ...helper_tools.operational_parameters import Params
from tensorflow.keras.layers import Dense
"""The Evaluation Strategy phase of the NAS framework
"""

class NASEval(object):
    '''Responsible for instantiating and evaluating candidate architectures

    Attributes:
        config (dict): the predefined operational parameters pertaining to evaluation (defined in :func:`~config.params.Params.evaluation_strategy_config`)
        datagen (:class:`~tensorflow.keras.preprocessing.image.ImageDataGenerator`): the data generator used to train all candidates (instantiated once to preserve resources)
        model (:class:`~tensorflow.keras.models.Model`): the candidate model to be evaluated (overwritten and deleted with every evaluation process to prevent leaks)
    '''

    def __init__(self, config):
        '''
        Initializes the evaluation parameters' configuration ;
        for a different dataset, a data-loader must be specified below
        as with CIFAR10 Keras loader

        Args:
            config (dict): the predefined operational parameters pertaining to evaluation (defined in :func:`~config.params.Params.evaluation_strategy_config`)
        '''

        self.config = config

        # specify dataset loaders
        if config['dataset'] == 'CIFAR10':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        elif config['dataset'] == 'MNIST':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
            # Add a placeholder dimension to the dataset to match CIFAR10's
            self.X_train = self.X_train.reshape(-1,28,28,1)
            self.X_test = self.X_test.reshape(-1,28,28,1)
        elif config['dataset'] == 'FASHION_MNIST':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
            # Add a placeholder dimension to the dataset to match CIFAR10's
            self.X_train = self.X_train.reshape(-1,28,28,1)
            self.X_test = self.X_test.reshape(-1,28,28,1)
        else:
            raise ValueError('Dataset loader undefined!')

        if Params['ENABLE_WEIGHT_SAVING']:
            # create directory if it does not exist
            FileHandler.create_dir(os.path.join(Params.get_results_path(),
                                                Params['WEIGHT_FILES_SUBPATH']))

        self.__initialize_dataset()


    def __instantiate_network(self, arch):
        '''Instantiates a Keras network given an architecture op list

        Args:
            arch (list): a list of architecture operations ([str]), encoded by :class:`~core.nas.search_space.NASSearchSpace`
        '''

        # residual counters
        res_count = []

        # add input according to given dataset shape
        net = inputs = Input(shape=(self.X_train.shape[1:4]))

        # add input stem
        for op in self.config['input_stem']:
            net = op()(net)

        # add hidden layers
        for layer in arch:

            if layer.startswith('sc'):
                # start residual block
                res_count.append((net, int(layer[3:])))
                continue

            assert layer in self.config['operations']['reference_space'], 'Operation must be defined as a partial in HIVE_EVAL_CONFIG'
            net = self.config['operations']['reference_space'][layer]()(net)

            for idx, row in enumerate(res_count):
                connection, counter = row
                counter -= 1

                # apply pooling to residual blocks to maintain shape
                # [deprecated] -- pooling layers padded
                # if 'pool' in layer:
                #     connection = self.config['operations'][layer]()(connection)

                if counter == 0:
                    # conv1x1 to normalize channels
                    fx = Conv2D(net.shape[-1], (1, 1), padding='same')(connection)
                    net = Add()([fx, net])
                    del res_count[idx]
                else:
                    res_count[idx] = (connection, counter)

        # add output stem
        for op in self.config['output_stem']:
            net = op()(net)

        # add output layer
        net = Dense(len(np.unique(self.y_test)), activation='softmax')(net)

        self.model = Model(inputs, net)


    def get_weights_filename(self, arch):
        '''
        Hashes the architecture op-list into a filename using SHA1

        Args:
            arch (list): a list of architecture operations ([str]), encoded by :class:`~core.nas.search_space.NASSearchSpace`

        Returns:
            str: SHA1-hashed unique string ID for the given architecture
        '''

        return hashlib.sha1(''.join(arch).encode("UTF-8")).hexdigest()


    def evaluate(self, arch):
        '''
        Evaluates the candidate architecture given a string-encoded
        representation of the model

        Args:
            arch (list): a list of architecture operations ([str]), encoded by :class:`~core.nas.search_space.NASSearchSpace`

        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (in case of ACT), hashed file name, number of trainable parameters, and the last epoch's momentum value if applicable
        '''

        # instantiate/compile model
        self.__instantiate_network(arch)
        self.__compile_model()

        # train model
        # self.model.fit(x=self.X_train,
        #         y=self.y_train,
        #         batch_size=self.config['batch_size'],
        #         epochs=self.config['epochs'],
        #         verbose=1,
        #         validation_data=(self.X_test, self.y_test))

        assert self.config['epochs'] > 2 or self.config['momentum_epochs'] == 0, 'Momentum Augmentation requires at least 3 epochs per candidate'

        cb = []
        if self.config['termination_threshold_factor'] > 0.0:
            cb.append(TerminateOnThreshold(threshold_multiplier=self.config['termination_threshold_factor'],
                                           diminishing_factor=self.config['termination_diminishing_factor']))
        if self.config['momentum_epochs'] > 0:
            cb.append(MomentumAugmentation())

        history = self.model.fit(self.datagen.flow(self.X_train,
                                                   self.y_train,
                                                   shuffle=True,
                                                   batch_size=self.config['batch_size'],
                                                   subset='training'),
                                 validation_data=self.datagen.flow(self.X_train,
                                                                   self.y_train,
                                                                   batch_size=int(self.config['batch_size'] / 2),
                                                                   subset='validation'),
                                 epochs=self.config['epochs'],
                                 verbose=1,
                                 callbacks=cb)

        momentum = self.model.momentum if hasattr(self.model, 'momentum') else {}
        # test model
        eval_res = self.model.evaluate(self.X_test,
                                      self.y_test,
                                      batch_size=self.config['batch_size'],
                                      verbose=1)

        # dump training history
        hist_path = os.path.join(Params.get_results_path(), Params['HISTORY_FILES_SUBPATH'])
        hist_fn = self.get_weights_filename(arch) + '.pickle'

        FileHandler.save_pickle(history.history, hist_path, hist_fn)

        # save weights for later retraining when needed
        filename = self.get_weights_filename(arch) + '.h5'

        if Params['ENABLE_WEIGHT_SAVING'] or self.config['momentum_epochs'] > 0:
            model_path = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'])
            self.model.save(model_path + filename)

        trainable_params = np.sum([K.count_params(w) for w in self.model.trainable_weights])

        # housekeeping
        del self.model

        # return validation accuracy to maximize + additional data for saving purposes
        retval = {
            'fitness': eval_res[1],
            'epochs': len(history.history['loss']),
            'filename': filename,
            'params': trainable_params,
            'momentum': momentum
        }

        return retval


    def fully_train(self, model_file=None, arch=None):
        '''Loads and continues training of a partially-trained model using either a string-encoded architecture (instantiates the model and trains from scratch) or a model h5 file

        Args:
            model_file (str, optional): a previously saved h5 model file (continues training)
            arch (list, optional): a list of architecture operations ([str]), encoded by :class:`~core.nas.search_space.NASSearchSpace`

        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (not including any previous trainings), hashed file name, number of trainable parameters
        '''

        hist_path = os.path.join(Params.get_results_path(), Params['HISTORY_FILES_SUBPATH'])
        hist_fn = ''

        if model_file is not None:
            # load model
            self.model = load_model(model_file)
            hist_fn =  model_file.split('/')[-1] + '.full.pickle'
        else:
            # instantiate network
            self.__instantiate_network(arch)
            self.__compile_model()
            model_file = self.get_weights_filename(arch)
            hist_fn = model_file + '.full.pickle'

        # continue training
        # self.model.fit(x=self.X_train,
        #                y=self.y_train,
        #                batch_size=self.config['batch_size'],
        #                epochs=self.config['full_train_epochs'],
        #                verbose=1,
        #                validation_data=(self.X_test, self.y_test))

        history = self.model.fit(self.datagen.flow(self.X_train,
                                                   self.y_train,
                                                   shuffle=True,
                                                   batch_size=self.config['batch_size'],
                                                   subset='training'),
                                 validation_data=self.datagen.flow(self.X_train,
                                                                   self.y_train,
                                                                   batch_size=int(self.config['batch_size'] / 2),
                                                                   subset='validation'),
                                  epochs=self.config['full_train_epochs'],
                                  verbose=1)


        # test model
        eval_res = self.model.evaluate(self.X_test,
                                       self.y_test,
                                       batch_size=self.config['batch_size'],
                                       verbose=1)

        # dump training history
        FileHandler.save_pickle(history.history, hist_path, hist_fn)

        # save weights for later retraining when needed
        # if Params['ENABLE_WEIGHT_SAVING']:
        # Save fully trained model regardless of params
        self.model.save(model_file + '.full.h5')

        trainable_params = np.sum([K.count_params(w) for w in self.model.trainable_weights])

        # housekeeping
        del self.model

        retval = {
            'fitness': eval_res[1],
            'epochs': len(history.history['loss']),
            'filename': model_file,
            'params': trainable_params
        }

        return retval


    def momentum_training(self, weights_file, m_epochs):
        '''Loads and continues training of a partially-trained model

        Args:
            weights_file (str): the previously saved h5 model file (continues training)
            m_epochs (int): number of momentum epochs to continue training for

        Returns:
            dict: final fitness value (accuracy) after training continuation
        '''

        # load model
        self.model = load_model(weights_file)
        # hist_fn = model_file.split('/')[-1] + '.ma.pickle'

        history = self.model.fit(self.datagen.flow(self.X_train,
                                                   self.y_train,
                                                   shuffle=True,
                                                   batch_size=self.config['batch_size'],
                                                   subset='training'),
                                 validation_data=self.datagen.flow(self.X_train,
                                                                   self.y_train,
                                                                   batch_size=int(self.config['batch_size'] / 2),
                                                                   subset='validation'),
                                  epochs=m_epochs,
                                  verbose=1)


        # test model
        eval_res = self.model.evaluate(self.X_test,
                                       self.y_test,
                                       batch_size=self.config['batch_size'],
                                       verbose=1)

        # dump training history
        # FileHandler.save_pickle(history.history, hist_path, hist_fn)

        # Save continued-training model
        self.model.save(weights_file)

        # housekeeping
        del self.model
        gc.collect()

        retval = {
            'fitness': eval_res[1]
        }

        return retval


    def __initialize_dataset(self):
        '''
        Prepares the dataset

        Preprocessing includes standardization and affine transformation (if applicable)

        Initializes the :class:`tensorflow.keras.preprocessing.image.ImageDataGenerator` for evaluation
        '''

        # Standardize data
        X_train_mean = np.mean(self.X_train, axis=(0,1,2))
        X_train_std = np.std(self.X_train, axis=(0,1,2))
        self.X_train = (self.X_train - X_train_mean) / X_train_std
        self.X_test = (self.X_test - X_train_mean) / X_train_std

        # Affine transformations
        if Params['AFFINE_TRANSFORMATIONS_ENABLED']:
            self.datagen = ImageDataGenerator(
                zoom_range = [0.8, 1.1],
                shear_range= 10,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                preprocessing_function=ImgAug.augment,
                validation_split=0.2
            )
        else:
            self.datagen = ImageDataGenerator(preprocessing_function=ImgAug.augment,
                                              validation_split=0.2)

        # per docs, ImageDataGenerator.fit() is only needed if the generator enables:
        # featurewise_center or featurewise_std_normalization or zca_whitening
        # self.datagen.fit(self.X_train)

        # # One-hot encoding
        # deprecated; memory consumption too high for intermediate tensors
        # self.y_train = utils.to_categorical(self.y_train)
        # self.y_test = utils.to_categorical(self.y_test)


    def __compile_model(self):
        '''Compiles model in preparation for evaluation
        '''

        self.model.compile(loss='sparse_categorical_crossentropy', \
                           optimizer=self.config['optimizer'](), \
                           metrics=['sparse_categorical_accuracy'])



