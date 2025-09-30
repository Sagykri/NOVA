import logging.handlers
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import uuid

sys.path.insert(1, os.getenv("NOVA_HOME"))
sys.path.insert(0, os.getenv("HOME"))
import importlib
import json
import logging
import string

def get_if_exists(container:object, param_name: string, default_value:Any=None, verbose:bool=False)->Any:
    """Get value of param in container if it exists, otherwise return default value

    Args:
        container (object): The object containing the parameter
        param_name (string): The name of the param to retrieve
        default_value (Any, optional): Default value to retrieve if the param doesn't exists in container. Defaults to None.
        verbose (bool, optional): Notify on function status if set to true. Default to False.
    Returns:
        Any: Param value (or default value if it doesn't exist)
    """
    
    if isinstance(container, dict):
        if param_name in container:
            return container[param_name]
    elif hasattr(container, param_name):
        return getattr(container, param_name)
    
    if verbose:
        logging.warning(f"{param_name} wasn't found in {type(container)}. Using default value: {default_value}")
        
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

def filter_paths_by_substrings(paths: List[Path], substrings: List[str], part_index: int, filter_out: bool = False) -> List[Path]:
    """
    Filter paths by searching for substrings in a specified part of the path.
    Supports both positive and negative part indices. Can either filter out or keep matching paths based on the 'filter_out' flag.

    Args:
        paths (List[Path]): A list of Path objects.
        substrings (List[str]): A list of substrings to filter by.
        part_index (int): The index of the part to search within (can be negative).
        filter_out (bool): If True, filter out paths where the specified part contains any of the substrings. If False, keep only those paths.

    Returns:
        List[Path]: A filtered list of Path objects based on the provided parameters.

    Example:
        >>> paths = [Path("/folder1/file1/image1.tiff"), Path("/folder1/file2/image2.tiff"), Path("/folder2/file3/image1.tiff")]
        >>> substrings = ["file1", "file3"]
        
        >>> # Filter out paths where part 1 contains 'file1' or 'file3'
        >>> filter_paths_by_substrings(paths, substrings, 1, filter_out=True)
        [PosixPath('/folder1/file2/image2.tiff')]

        >>> # Keep only paths where part 1 contains 'file1' or 'file3'
        >>> filter_paths_by_substrings(paths, substrings, 1, filter_out=False)
        [PosixPath('/folder1/file1/image1.tiff'), PosixPath('/folder2/file3/image1.tiff')]

        >>> # Negative index: filter out paths where the last part contains 'image1'
        >>> filter_paths_by_substrings(paths, ["image1"], -1, filter_out=True)
        [PosixPath('/folder1/file2/image2.tiff')]
    """
    
    # If substrings list is empty, return the full list of paths
    if not substrings or all(s.strip() == "" for s in substrings):
        return paths
    
    # Determine whether to filter out or keep paths based on the part matching the substrings
    return [
        path for path in paths
        if len(path.parts) > abs(part_index) and (
            (not any(substring in path.parts[part_index] for substring in substrings)) if filter_out
            else (any(substring in path.parts[part_index] for substring in substrings))
        )
    ]

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
                        format="%(asctime)s %(levelname)s: %(message)s",
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

def log_gpus_status():
    """log the gpus status
    """
    res = apply_for_all_gpus(getfreegpumem)
    logging.info(f"Resources (Free, Used, Total): {res}")

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

def save_config(config, output_folder_path: str) -> None:
    """Saves the configuration data to a JSON file."""
    os.makedirs(output_folder_path, exist_ok=True)
    with open(os.path.join(output_folder_path, 'config.json'), 'w') as json_file:
        json.dump(config.__dict__, json_file, indent=4)