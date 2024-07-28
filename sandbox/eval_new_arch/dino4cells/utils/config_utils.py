import importlib
import os


class DictToObject:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to objects
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
                


def load_dict_from_pyfile(path):
    """
    Load a dictionary from a Python file.

    Parameters:
    - file_path (str): The path to the Python file containing the dictionary.

    Returns:
    - dict: The dictionary loaded from the file.
    """
    
    file_path, func_name = f'{os.path.dirname(path)}.py', os.path.basename(path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    func = getattr(module, func_name)
    return func()
