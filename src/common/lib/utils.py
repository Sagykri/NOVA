import datetime
import os
import sys
import uuid
sys.path.insert(1, os.getenv("MOMAPS_HOME"))


import importlib
import json
import logging
import string
import numpy as np
import pandas as pd
import datetime

def get_if_exists(container:object, param_name: string, default_value=None):
    """Get value of param in container if it exists, otherwise return default value

    Args:
        container (object): The object containing the parameter
        param_name (string): The name of the param to retrieve
        default_value (_type_, optional): Default value to retrieve if the param doesn't exists in container. Defaults to None.

    Returns:
        value: Param value (or default value if it doesn't exist)
    """
    
    if isinstance(container, dict):
        if param_name in container:
            return container[param_name]
    elif hasattr(container, param_name):
        return getattr(container, param_name)
    
    return default_value

def xy_to_tuple(xy_arr):
    """Transform an array paired variable to a tuple
    i.e from [x0,y0,x1,y1,...] to [(x0,y0), (x1,y1),...]

    Args:
        xy_arr (iterable): [x0,y0,x1,y1,...]

    Returns:
        iterable: [(x0,y0), (x1,y1),...]
    """
    
    return [(arr[0],arr[1]) for arr in xy_arr]

def flat_list_of_lists(l):
    return [item for sublist in l for item in sublist]


def generate_confusion_matrix(model):
    pass


def get_colors_dict(labels, colors_dict):
    """Get mapping between the given colors and labels

    Args:
        labels ([string]): The labels
        colors_dict ({term:color}, optional): A dictionary of terms and colors. (ex. {'_unstressed': 'red', '_stressed': 'blue'}). Defaults to COLORS_MAPPING.

    Returns:
        _type_: _description_
    """
    colors = {}
    
    labels_unique = np.unique(labels)
    
    for term, color in colors_dict.items():
        term_indexes = np.where(np.char.find(labels_unique, term)>-1)[0]
        for i in term_indexes:
            label = labels_unique[i]
            colors[label] = color
    return colors

    
def load_config_file(path:string, custom_filename:string=None, savefolder:string=None):
    """Load config file (and save it to file for documentation)

    Args:
        path (string): Path to config file (the last argument will be the class to load from the file)
        filename (string, Optional): the file name of the file (config that was used) to save. Default to GUID
        savefolder (string, Optional): Path to save the config to. Default to config.CONFIGS_USED_FOLDER
    Returns:
        _type_: Instance of the loaded config class
    """
    config_class = get_class(path)
    config = config_class()
    
    if savefolder is None or savefolder.strip == "":
        savefolder = config.CONFIGS_USED_FOLDER
    
    if custom_filename is None or custom_filename.strip() == "":
        custom_filename = f"{uuid.uuid4().hex}"
        
    savepath = os.path.join(savefolder, f"{custom_filename}.json")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
        
    with open(savepath, 'w') as f:
        f.write(json.dumps(config.__dict__))
    
    return config

def get_class(path:string):
    """Get the class of a given python file and class

    Args:
        path (string): Path to module file (the last argument will be the class to load from the file)

    Returns:
        _type_: The class
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
    



def gpuinfo(gpuidx):
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


def getfreegpumem(gpuidx):
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
    
def apply_for_all_gpus(func):
    import torch
    
    n_devices = torch.cuda.device_count()
    l = []
    for i in range(n_devices):
        l.append(func(i))
    return l

def get_nvidia_smi_output(gpuidx):
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

class LogDF(object):
    def __init__(self, folder_path: string, filename_prefix='', index=None,
                 columns=None, should_write_index=False):
        self.__path = os.path.join(folder_path, filename_prefix + datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f") + '.csv')
        self.__df = pd.DataFrame(index=index, columns=columns)
        self.__should_write_index = should_write_index
        
        # Create the file
        self.__save(self.__should_write_index, mode='w')
    
    @property
    def df(self):
        return self.__df
    
    @property
    def path(self):
        return self.__path
    
    def write(self, data):
        if type(data) is not pd.DataFrame:
            data = [data]
        
        try:
            self.__df = pd.DataFrame(data, columns=self.__df.columns)
        except Exception as ex:
            raise f"Can't convert 'data' to pd.DataFrame ({ex})"
            
        return self.__save(self.__should_write_index)
        
    def __save(self, index:bool=False, mode='a'):
        self.__df.to_csv(self.__path, index=index, mode=mode, header=mode=='w')
        
        return self.__path
