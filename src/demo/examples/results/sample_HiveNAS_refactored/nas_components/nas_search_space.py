import networkx as nx
import numpy as np
import re

"""The Search Space phase of the NAS framework
"""

class NASSearchSpace(object):
    '''Defines the Search Space used to sample candidates by HiveNAS

    Attributes:
        all_paths (list): a list of all Directed Acyclic sub-Graphs in the search space (i.e all candidates)
        config (dict): the predefined operational parameters pertaining to the search space (defined in :func:`~config.params.Params.search_space_config`)
        dag (:class:`~networkx.DiGraph`): the search space graph (depracated; otf-encoding used at the moment)
    '''

    def __init__(self, config):
        '''
        Configurations are predefined in the Params class (:func:`~config.params.Params.search_space_config`),
        the implementation should work given any set of operations' mapping
        and depth

        Args:
            config (dict): the predefined operational parameters pertaining to the search space (defined in :func:`~config.params.Params.search_space_config`)
        '''

        self.config = config
        # self.__initialize_graph()


    def sample(self):
        '''
        Samples a random point (i.e a candidate architecture) from the search space

        Returns:
            str: string-encoded representation of the architecture
        '''

        # assert self.all_paths != None, 'Search space needs to be initialized!'

        # idx = np.random.randint(0, len(self.all_paths))
        # return self.__encode_path(self.all_paths[idx])

        path = ['input']

        for l in range(self.config['depth']):

            if np.random.rand() < self.config['residual_blocks_rate']:
                sc_depth = np.random.randint(1, self.config['depth'] - l + 1)
                path.append('L{}_sc_{}'.format(l+1, sc_depth))

            path.append('L{}_{}'.format(l+1, np.random.choice(
                list(self.config['operations']['search_space'].keys())
            )))

        path.append('output')

        return self.__encode_path(path)


    def get_neighbor(self, path_str):
        '''Returns a path with 1-op difference (a neighbor).

        The definition of a neighbor architecture differs from one model to another in the literature,
        however, the general consensus is a 1-op difference network [1].



        [1] `Colin White et al. “How Powerful are Performance Predictors in Neural Architecture Search?”
        In: Advances in Neural Information Processing Systems 34 (2021).`


        Args:
            path_str (str): string-encoded representation of the architecture

        Returns:
            str: string-encoded representation of a neighbor architecture
        '''

        path = self.__strip_path(self.__decode_path(path_str))

        component = np.random.randint(1, len(path) - 1)

        ops = []
        if path[component].startswith('sc'):
            # modify skip-connection (either remove it or change residual depth)
            sc_max_depth = len([op for op in path[component:] if not op.startswith('sc')])
            ops = [f'sc_{i}' for i in range(sc_max_depth)]
            ops.remove(path[component])
        else:
            # modify operation
            ops = list(self.config['operations']['search_space'].keys())
            ops.remove(path[component])

        # Replace randomly chosen component (operation) with any other op
        path[component] = np.random.choice(ops)

        # prune skip-connection if op == sc_0
        if path[component] == 'sc_0':
            del path[component]

        return self.__encode_path(path)


    def eval_format(self, path):
        '''
        Formats a path for evaluation (stripped, decoded, and
        excluding input/output layers) given a string-encoded path

        Args:
            path (str): string-encoded representation of the architecture

        Returns:
            list: a list of operations ([str]) representing a model architecture to be used by the evaluation strategy
        '''

        return self.__strip_path(self.__decode_path(path))[1:-1]


    def __initialize_graph(self):
        '''
        .. deprecated:: 0.1.0

        Initializes the search space DAG for easier sampling by the
        search algorithm.

        :class:`~networkx.DiGraph` search space encoding consumes too much memory -- deprecated
        '''

        self.dag = nx.DiGraph()
        self.dag.add_node('input')

        for l in range(self.config['depth']):
            for op in self.config['operations']:
                # Connect input layer to first hidden layer
                if l == 0:
                    self.dag.add_edges_from([('input',
                                              'L{}_{}'.format(l+1, op))])
                    continue

                # Densely connect middle layers
                for prev_op in self.config['operations']:
                    self.dag.add_edges_from([('L{}_{}'.format(l, prev_op),
                                              'L{}_{}'.format(l+1, op))])

                # Connect last hidden layer to output stem
                if l == self.config['depth'] - 1:
                    self.dag.add_edges_from([('L{}_{}'.format(l+1, op),
                                              'output')])

        self.all_paths = list(nx.all_simple_paths(self.dag, 'input', 'output'))


    def __encode_path(self, path):
        '''Returns a string encoding of a given path (list of ops)

        Args:
            path (list): list of operations ([str]) representing the architecture

        Returns:
            str: string-encoded representation of the given architecture
        '''

        return '|'.join(self.__strip_path(path))


    def __decode_path(self, path):
        '''Returns a list of operations given a string-encoded path

        Args:
            path (str): string-encoded representation of an architecture

        Returns:
            list: list of operations ([str]) representing the given architecture
        '''

        ops = path.split('|')

        for i in range(1, len(ops) - 1):
            ops[i] = 'L{}_{}'.format(i, ops[i])

        return ops


    def __strip_path(self, path):
        '''Strips path of layer ID prefixes given a list of ops

        Args:
            path (list): list of operations ([str]), each with a layer ID prefix (as was needed for the DAG version of the search space)

        Returns:
            list: list of operations ([str]) stipped of the layer IDs
        '''

        return [re.sub('L\d+_', '', s) for s in path]


    def compute_space_size(self):
        '''
        Returns the number of possible architectures in the given space
        (i.e operations and depth) for analytical purposes

        Returns:
            int: the size of the search space (number of all possible candidates)
        '''

        return len(list(self.config['operations']['search_space'].keys())) ** \
        self.config['depth']


