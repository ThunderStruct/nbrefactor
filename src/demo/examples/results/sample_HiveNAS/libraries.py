import humanize
import sys
from google.colab import drive
from tensorflow.test import gpu_device_name
import os
import psutil
import GPUtil as GPU
pass # !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
pass # !pip install gputil
pass # !pip install psutil
pass # !pip install humanize
pass # !pip install pyyaml



ON_COLAB = 'google.colab' in sys.modules


""" Preliminary configurations / tests
"""

def test_colab_GPU_mem():
    '''Colab GPU memory test (Colab often shares a computing unit \
    with multiple users, ending up with less than 60% of the expected \
    GPU resources).

    Restart instance to get reallocated resources.
    '''

    GPUs = GPU.getGPUs()

    gpu = GPUs[0]

    process = psutil.Process(os.getpid())
    print('Gen RAM Free: ' + humanize.naturalsize(psutil.virtual_memory().available), ' | Proc size: ' + humanize.naturalsize(process.memory_info().rss))
    print('GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB'.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    # Check if GPU usage is > 2% (i.e device shared with other Colab users)
    print('GPU Memory is shared! Restart the runtime.' if gpu.memoryUtil > 0.05 else 'GPU Memory is free!')


if ON_COLAB:
    # output.enable_custom_widget_manager()
    drive.mount('/content/gdrive')


    if gpu_device_name() != '/device:GPU:0':
        print('GPU device not found -- Try enabling GPU acceleration in Colab\'s runtime settings')

    else:
        print('GPU device found!')
        test_colab_GPU_mem()


