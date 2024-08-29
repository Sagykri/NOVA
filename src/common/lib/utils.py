import os
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import uuid
import importlib
import json
import logging
import string

def get_if_exists(container:object, param_name: string, default_value:Any=None)->Any:
    """Get value of param in container if it exists, otherwise return default value

    Args:
        container (object): The object containing the parameter
        param_name (string): The name of the param to retrieve
        default_value (Any, optional): Default value to retrieve if the param doesn't exists in container. Defaults to None.

    Returns:
        Any: Param value (or default value if it doesn't exist)
    """
    
    if isinstance(container, dict):
        if param_name in container:
            return container[param_name]
    elif hasattr(container, param_name):
        return getattr(container, param_name)
    
    return default_value

def xy_to_tuple(xy_arr:Iterable[Any])->Iterable[Tuple[Any,Any]]:
    """Transform an array paired variable to a tuple
    i.e from [x0,y0,x1,y1,...] to [(x0,y0), (x1,y1),...]

    Args:
        xy_arr (Iterable[Any]): [x0,y0,x1,y1,...]

    Returns:
        Iterable[Tuple[Any,Any]]: [(x0,y0), (x1,y1),...]
    """
    
    return [(arr[0],arr[1]) for arr in xy_arr]

def flat_list_of_lists(l:List[List[Any]])->List[Any]:
    """Float a list of lists into a list

    Args:
        l (List[List[Any]]): The nested list

    Returns:
        List[Any]: The flatted list
    """
    return [item for sublist in l for item in sublist]
  
def load_config_file(path:string, custom_filename:string=None, savefolder:string=None):
    """Load config file (and save it to file for documentation)

    Args:
        path (string): Path to config file (the last argument will be the class to load from the file)
        filename (string, Optional): the file name of the file (config that was used) to save. Default to GUID
        savefolder (string, Optional): Path to save the config to. Default to config.CONFIGS_USED_FOLDER
    Returns:
        BaseConfig: Instance of the loaded config class
    """
    config_class = get_class(path)
    config = config_class()
    
    if savefolder is None or savefolder.strip == "":
        savefolder = config.CONFIGS_USED_FOLDER
    
    if custom_filename is None or custom_filename.strip() == "":
        custom_filename = f"{uuid.uuid4().hex}"
        
    savepath = os.path.join(savefolder, f"{custom_filename}.json")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)
        
    with open(savepath, 'w') as f:
        f.write(json.dumps(config.__dict__))
    
    return config

def get_class(path:string)->Any:
    """Get the class of a given python file and class

    Args:
        path (string): Path to module file (the last argument will be the class to load from the file)

    Returns:
        Any: The class
    """
    if path.startswith("."):
        path = path[1:]
    if path.startswith("/"):
        path = path[1:]
    if path.endswith(".py"):
        path = os.path.splitext(path)[0]
    
    # Extract and load the module
    module_path = os.path.dirname(path).replace('/', '.')
    
    module = importlib.import_module(module_path)
    
    # Extract the class name from the path and load it from the module
    class_in_module = os.path.basename(path)
    
    module_class = module.__dict__[class_in_module]
    
    return module_class
        
def init_logging(path:string):
    """Init logging.
    Writes to log file and console.
    Args:
        path (string): Path to log file
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(path),
                            logging.StreamHandler()
                        ])
    
def are_dicts_equal_except_keys(dict1:Dict, dict2:Dict, keys_to_ignore:Union[str, List[str]])->bool:
    """Check whether two dictionaries are equal except for some keys

    Args:
        dict1 (Dict): The first dictionary
        dict2 (Dict): The second dictionary
        keys_to_ignore (Union[str, List[str]]): Name or list of names of the keys to ignore

    Returns:
        bool: Are they equal?
    """
    # Ensure keys_to_ignore is a list, even if a single string is provided
    if isinstance(keys_to_ignore, str):
        keys_to_ignore = [keys_to_ignore]
            
    # Create shallow copies of the dictionaries to avoid modifying the originals
    dict1_copy = dict1.copy()
    dict2_copy = dict2.copy()

    # Remove each key in keys_to_ignore from both dictionaries
    for key in keys_to_ignore:
        dict1_copy.pop(key, None)
        dict2_copy.pop(key, None)

    # Compare the modified dictionaries
    return dict1_copy == dict2_copy

def gpuinfo(gpuidx:int)->Dict:
    """
    Get GPU information

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    dict :
        GPU information in dictionary
    """
    import subprocess

    out_dict = {}
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
        for item in out_list:
            if ':' in item:
                fragments = item.split(':')
                if len(fragments) == 2:
                    out_dict[fragments[0].strip()] = fragments[1].strip()
    except Exception as e:
        print(e)
    return out_dict

def getfreegpumem(gpuidx:int)->int:
    """
    Get free GPU memory

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    int :
        Free memory size
    """
    info = gpuinfo(gpuidx)
    if len(info) > 0:
        return info['Free'], info['Used'], info['Total']
    else:
        return -1
    
def apply_for_all_gpus(func:Callable[[int], Any])->List[Any]:
    """Apply func to all gpus

    Args:
        func (Callable[[int], Any]): The function to apply

    Returns:
        List[Any]: List of the return values of the function from all gpus
    """
    import torch
    
    n_devices = torch.cuda.device_count()
    l = []
    for i in range(n_devices):
        l.append(func(i))
    return l

def get_nvidia_smi_output(gpuidx:int)->Dict:
    """Get the nvidia smi output for the given gpu unit

    Args:
        gpuidx (int): The gpu index

    Returns:
        Dict: The output
    """
    import subprocess

    sp = subprocess.Popen(
            ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY,UTILIZATION'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    out_dict = {}
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('\n')
    
    for item in out_list:
            if ':' in item:
                fragments = item.split(':')
                if len(fragments) == 2:
                    out_dict[fragments[0].strip()] = fragments[1].strip()

    return out_dict