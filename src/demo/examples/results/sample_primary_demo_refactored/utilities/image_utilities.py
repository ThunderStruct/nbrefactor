from tensorflow.image import random_contrast
import numpy as np
from tensorflow.image import random_saturation

# dependencies are imported in cell #1, 
# but should be injected during the refactoring process

# This module is borrowed from HiveNAS (https://github.com/ThunderStruct/HiveNAS/blob/main/src/utils/image_aug.py)

class ImageUtils:
    '''Wraps a bunch of static methods for image augmentations
    using Tensorflow/Keras.
    '''

    @staticmethod
    def random_cutout(np_tensor, cutout_color=127):
        '''Randomly applies cutout augmentation to a given rank 3 tensor as
        defined in [1]. Defaults to grey cutout

        [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of
        convolutional neural networks with cutout.

        Args:
            np_tensor (:class:`numpy.array`): rank 3 numpy tensor-respresentation of \
            the data sample
            cutout_color (int, optional): RGB-uniform value of the cutout color \
            *(defaults to grey (:code:`127`). white (:code:`255`) and black \
            (:code:`0`) are also valid)*

        Returns:
            :class:`numpy.array`: augmented numpy tensor (with a random cutout)
        '''

        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])

        cutout_height_point = np.random.randint(np_tensor.shape[0] - cutout_height)
        cutout_width_point = np.random.randint(np_tensor.shape[1] - cutout_width)
        
        ret_tensor = np_tensor.copy()
        ret_tensor[cutout_height_point: cutout_height_point + cutout_height,
                  cutout_width_point: cutout_width_point + cutout_width,
                  :] = cutout_color    # 127 = grey cutout,
                                       # 0 (black) or 255 (white) also valid

        return np.array(ret_tensor)


    @staticmethod
    def random_contrast(np_tensor):
        '''Apply random contrast augmentation

        Args:
            np_tensor (:class:`numpy.array`): rank 3 numpy tensor-respresentation of \
            the data sample

        Returns:
            (:class:`numpy.array`): transformed numpy tensor with random contrast
        '''

        return np.array(random_contrast(np_tensor, 0.25, 3))


    @staticmethod
    def random_saturation(np_tensor):
        '''Apply random saturation augmentation (only works on RGB images, \
        skipped on grayscale datasets)

        Args:
            np_tensor (:class:`numpy.array`): rank 3 numpy tensor-respresentation of \
            the data sample

        Returns:
            (:class:`numpy.array`): transformed numpy tensor with random saturation
        '''

        if np_tensor.shape[-1] != 3:
            # not an RGB image, skip (cannot do saturation aug)
            return np.array(np_tensor)

        return np.array(random_saturation(np_tensor, 0.2, 3))



