import os
import pandas as pd

# redundantly importing `os` (on purpose)
# unused import statement (on purpose)
# we will also randomly use pandas, which was declared in the previous cell

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def the_pandas_func():
    # this is just to showcase our CDA at work
    # this module should contain 2 dependencies given our used definitions here
    # `os` and `pandas as pd`
    
    df = pd.DataFrame()
    del df
    
    

