""" Writes the analyzed modules / notebook
"""

import os
import astor

def write_modules(dependencies, base_path):
    """
    Writes the extracted modules and injects dependencies respectively

    Args:
    dependencies (dict): Dictionary of function dependencies.
    base_path (str): Base path for the module files.
    """

    for module_name, data in dependencies.items():
        module_path = os.path.join(base_path, 
                                   module_name.lstrip('>').replace('>', 
                                                                   os.sep))
        
        print(module_path)
        # create the directory if not exists
        os.makedirs(os.path.dirname(module_path), exist_ok=True)

        with open(module_path, 'w') as f:
            for import_node in data['imports']:
                f.write(astor.to_source(import_node) + '\n')
            f.write('\n'.join(data['code']))