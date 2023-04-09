import importlib
import json
import logging
import os
import string
import numpy as np

def get_if_exists(container:object, param_name: string, default_value=None):
    """Get value of param in container if it exists, otherwise return default value

    Args:
        container (object): The object containing the parameter
        param_name (string): The name of the param to retrieve
        default_value (_type_, optional): Default value to retrieve if the param doesn't exists in container. Defaults to None.

    Returns:
        value: Param value (or default value if it doesn't exist)
    """
    if hasattr(container, param_name):
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

def generate_embeddings(run_config):
    """Generate embedding vectors

    Args:
        run_config (BaseConfig): run config to use
    """
    logging.info("init")

    logging.info("Load data")

    from src.common.lib.model import Model
    model = Model(run_config)


    logging.info("Load data")
    model.load_data()
    
    logging.info("Load model")
    model.load_model()
    
    logging.info("Load analytics")
    model.load_analytics()
        
    logging.info("Calc (and save) embvec")
    model_name = os.path.splitext(os.path.basename(run_config.MODEL_PATH))[0]
    model.analytics.model.calc_embvec(
                model.analytics.data_manager.test_data, savepath="default", filename=f"embeddings_{model_name}")    
    
    
    labels_output_path = os.path.join(run_config.MODEL_OUTPUT_DIR, f"labels_{model_name}.txt") 
    logging.info(f"Save labels to file: {labels_output_path}")
    np.savetxt(labels_output_path, model.test_label)
    
    
def load_config_file(path:string, postfix_filename:string=""):
    """Load config file (and save it to file for documentation)

    Args:
        path (string): Path to config file (the last argument will be the class to load from the file)
        postfix_filename (string): Postfix to add to the file name of the file (config that was used) to save
        to_save (bool, Optional): Should save the config to file. Defaults to True.
    Returns:
        _type_: Instance of the loaded config class
    """
    config_class = get_class(path)
    config = config_class()
    
    with open(f"{config.CONFIGS_USED_FOLDER}{postfix_filename}.txt", 'w') as f:
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
    
    