from .numerical_optimization_benchmarks.sphere import Sphere
from google.colab import files
from .helper_tools.operational_parameters import Params
import requests
from .helper_tools.file_handler import FileHandler
from .artificial_bee_colony_components.abc_optimizer import ArtificialBeeColony
from matplotlib import plt as plt
import gc
from .helper_tools.logger import Logger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from .helper_tools.image_augmentations import ImgAug
def delete_all():
    '''
        ***DESTRUCTIVE ACTION***
        Removes all weight files and results CSV.
        Used for freeing up space from faulty runs.

        --Make sure the file paths are entered correctly in Params--
    '''
    for root, dirs, files in os.walk(Params.get_results_path() + Params['WEIGHT_FILES_SUBPATH']):
        for f in files:
            filepath = os.path.join(Params.get_results_path() + Params['WEIGHT_FILES_SUBPATH'], f)
            os.remove(filepath)

    os.remove(os.path.join(Params.get_results_path(), f'{Params["CONFIG_VERSION"]}.csv'))

# delete_all()


''' Test ImageAugmentation (before/after plot) '''

gc.collect()
image_augmentor = ImageDataGenerator(
                                    zoom_range = [0.8, 1.1],
                                    shear_range= 10,
                                    rotation_range=15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    preprocessing_function=ImgAug.augment,
                                    validation_split=0.2)
image_augmentor_before = ImageDataGenerator()


DownURL = "https://images.pexels.com/photos/1056251/pexels-photo-1056251.jpeg?crop=entropy&cs=srgb&dl=pexels-ihsan-aditya-1056251.jpg&fit=crop&fm=jpg&h=426&w=640"
img_data = requests.get(DownURL).content

FileHandler.create_dir('/content/sample_data/impath/cat')

with open('/content/sample_data/impath/cat/cat-small.jpg', 'wb') as handler:
    handler.write(img_data)

data_before = image_augmentor_before.flow_from_directory(
    "/content/sample_data/impath",
    target_size=(213, 320),
    batch_size=1,
)
data = image_augmentor.flow_from_directory(
    "/content/sample_data/impath",
    target_size=(213, 320),
    batch_size=1,
)

fig, axs = plt.subplots(1,2, figsize=(15,12))

plt.subplot(axs[0])
plt.imshow(data_before.next()[0][0].astype('int'))
plt.title("Before")

plt.subplot(axs[1])
plt.imshow(data.next()[0][0].astype('int'))
plt.title("After")


''' Unit Testing '''

Logger.EVALUATION_LOGGING = False

objective_interface = Sphere(10)

abc = ArtificialBeeColony(objective_interface)

abc.optimize()







