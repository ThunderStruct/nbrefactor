from ..img_aug import ImageAugmentation
from ...utilities.plotting_utils.visualize_img_aug import plot_img_aug

img_aug = ImageAugmentation()

plot_img_aug(img_aug.get_sample(), img_aug.get_contrasted(), 'After Random Contrast')


