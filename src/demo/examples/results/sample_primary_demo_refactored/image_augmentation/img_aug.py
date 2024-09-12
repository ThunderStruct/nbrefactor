from ..utilities.image_utilities import ImageUtils
from tensorflow.keras.datasets import cifar10



class ImageAugmentation:
    def __init__(self):
        # Load sample image from CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def get_sample(self, sample_idx=0):
        return self.x_train[sample_idx]

    def get_cutout(self, sample_idx=0):
        return ImageUtils.random_cutout(self.get_sample(sample_idx))

    def get_saturated(self, sample_idx=0):
        return ImageUtils.random_saturation(self.get_sample(sample_idx))
    
    def get_contrasted(self, sample_idx=0):
        return ImageUtils.random_contrast(self.get_sample(sample_idx))
    
        

