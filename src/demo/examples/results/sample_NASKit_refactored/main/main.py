import os
from ..utilities.config import Config
from google.colab import drive

if Config.MOUNT_GDRIVE:
    # gate GDrive mounting unless manually specified
    drive.mount('/content/gdrive')

    if Config.EXPERIMENT_NAME:
        Config.BASE_PATH = os.path.join(Config.BASE_PATH,
                                        Config.EXPERIMENT_NAME)


