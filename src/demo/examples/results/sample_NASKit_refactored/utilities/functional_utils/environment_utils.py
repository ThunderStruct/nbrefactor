import psutil
import re
import shutil
import subprocess
import os

def get_gpu_memory():

    if shutil.which('nvidia-smi') is None:
        # nvidia-smi not available
        return {}

    # execute nvidia-smi command in a subprocess
    result = subprocess.run(['nvidia-smi',
                             '--query-gpu=memory.total,memory.free,memory.used',
                             '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # parse the output
    memory_info = output.strip().split('\n')
    gpu_memory = {}
    for info in memory_info:
        total, free, used = re.split(r',\s*', info)
        gpu_memory = {
            'gpu_mem_total_mb': int(total),
            'gpu_mem_free_mb': int(free),
            'gpu_mem_used_mb': int(used)
        }

    return gpu_memory


def get_system_usage():
    ret_dict = {}

    # get current process
    process = psutil.Process(os.getpid())

    # get system ram usage
    mb_factor = 1024 ** 2
    ret_dict['sys_mem_free_mb'] = psutil.virtual_memory().available // mb_factor
    ret_dict['sys_mem_used_mb'] = psutil.virtual_memory().used // mb_factor
    ret_dict['sys_mem_total_mb'] = psutil.virtual_memory().total // mb_factor

    # get process size
    ret_dict['process_rss_mb'] = process.memory_info().rss // mb_factor

    return {**ret_dict, **(get_gpu_memory())}


