import time
from ..utilities.functional_utils.file_utils import ensure_dir
from ..utilities.config import Config
import os
import pandas as pd

class SystemMetrics:

    def __init__(self):
        self.records = []


    def add_record(self, task_id, task_version, task_name,
                   nas_epoch, sys_usage):
        df_dict = {
            'task_id': task_id,
            'task_version': task_version,
            'task_name': task_name,
            'nas_idx': nas_epoch,
            'time': time.strftime('%Y/%m/%d, %H:%M:%S', time.localtime())
        }
        df_dict = {**df_dict, **sys_usage}

        self.records.append(df_dict)



    def save(self, filename, dir='./sys_usage/'):
        dir_path = os.path.join(Config.BASE_PATH, dir)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        pd.DataFrame(self.records).to_csv(full_path)


