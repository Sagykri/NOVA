import os
import subprocess
import sys
import datetime
import logging
import random
from typing import List
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(1, os.getenv("NOVA_HOME")) 
from src.common.utils import are_dicts_equal_except_keys, init_logging


class BaseConfig():
    """The parent config for all other configs.\n
    Holds the common params such as the input and output path, the logs folder path, and seed.
    """
    def __init__(self):
        self.__now_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
        
        self.__SEED = 1

        self.HOME_FOLDER = os.environ['NOVA_HOME']
        self.HOME_DATA_FOLDER = os.environ['NOVA_DATA_HOME'] \
                                    if 'NOVA_DATA_HOME' in os.environ \
                                    else os.path.join(self.HOME_FOLDER, "input")
                        
        # Output
        self.__OUTPUTS_FOLDER = os.path.join(self.HOME_FOLDER, "outputs")
        self.CONFIGS_USED_FOLDER = os.path.join(self.__OUTPUTS_FOLDER, "configs_used", self.__now_str)

        # Logs
        self.__LOGS_FOLDER = os.path.join(self.HOME_FOLDER, 'logs')
        
    @staticmethod
    def from_dict(d:dict):
        """Init an instance from a given dictionary (instance.__dict__)

        Args:
            d (dict): A dictionary describing a class instance (instance.__dict__)

        Returns:
            Self: A new instance with the given dict values
        """
        import inspect
        
        instance = BaseConfig()
        
        # Set attributes values
        for name, value in d.items():
            if "__now_str" in name: continue
            
            setattr(instance, name, value)
        
        # Activate all the setters functions
        properties = inspect.getmembers(instance.__class__, predicate=inspect.isdatadescriptor)
        for (name, prop) in properties:
            # Skip the private built-in functions
            if (name.startswith("__") and name.endswith("__")):
                continue
            
            # Get the value
            prop_value = getattr(instance, name)
            # Activate the setter function
            prop.fset(instance, prop_value)
            
        return instance
        
    @property
    def SEED(self)->int:
        """Get the seed value

        Returns:
            int: The seed value
        """
        return self.__SEED
    
    @SEED.setter
    def SEED(self, value:int)->None:
        """Set the seed.\n
        This function sets the seed for both 'numpy' and the 'random' package.

        Args:
            value (int): The value for the seed
        """
        self.__SEED = value
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)
        
        torch.manual_seed(self.__SEED)
        torch.cuda.manual_seed_all(self.__SEED)
        cudnn.benchmark = False
        
    @property
    def LOGS_FOLDER(self)->str:
        """Get the folder where the logs are

        Returns:
            str: The path
        """
        return self.__LOGS_FOLDER
    
    @LOGS_FOLDER.setter
    def LOGS_FOLDER(self, path:str)->None:
        """Set the folder where the logs would be written into.\n
        This function init a logger, hence can be called only once.

        Args:
            path (str): The file path to where the log would be written into
        """
        if logging.getLogger().hasHandlers():
            return
    
        self.__LOGS_FOLDER = path
        jobid = os.getenv('LSB_JOBID')
        jobname = os.getenv('LSB_JOBNAME')
        # if jobname is not specified, the jobname will include the path of the script that was run.
        # In this case we'll have some '/' and '.' in the jobname that should be removed.
        if jobname:
            jobname = jobname.replace('/','').replace('.','') 

        username = 'UnknownUser'
        if jobid:
            # Run the bjobs command to get job details
            result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
            # Extract the username from the output
            username = result.stdout.replace('USER', '').strip()
        
        log_file_path = os.path.join(self.__LOGS_FOLDER, self.__now_str + f'_{jobid}_{username}_{jobname}.log')
        if not os.path.exists(self.__LOGS_FOLDER):
            os.makedirs(self.__LOGS_FOLDER)
            
        init_logging(log_file_path)
        
        logging.info(f"[{self.__class__.__name__}] Init (log path: {log_file_path}; JOBID: {jobid} Username: {username}) JOBNAME: {jobname}")
        logging.info(f"[{self.__class__.__name__}] NOVA_HOME={self.HOME_FOLDER}, NOVA_DATA_HOME={self.HOME_DATA_FOLDER}")

    @property
    def OUTPUTS_FOLDER(self)->str:
        """Get the path to the outputs folder

        Returns:
            str: The path
        """
        return self.__OUTPUTS_FOLDER
    
    @OUTPUTS_FOLDER.setter
    def OUTPUTS_FOLDER(self, path:str)->None:
        self.__OUTPUTS_FOLDER = path
        self.LOGS_FOLDER = os.path.join(self.__OUTPUTS_FOLDER, 'logs')
        self.CONFIGS_USED_FOLDER = os.path.join(self.__OUTPUTS_FOLDER, "configs_used", self.__now_str)
        
    def init(self)->None:
        """Activate all the setters functions.\n
        Needed only when configuration was loaded from a file
        """
        
        import inspect
        
        # Activate all the setters functions
        properties = inspect.getmembers(self.__class__, predicate=inspect.isdatadescriptor)
        for (name, prop) in properties:
            # Skip the private built-in functions
            if (name.startswith("__") and name.endswith("__")):
                continue
            
            # Get the value
            prop_value = getattr(self, name)
            # Activate the setter function
            prop.fset(self, prop_value)
        
    def is_equal(self, other)->bool:
        """Check if this config is equal to the given one

        Args:
            other (Union[Dict, BaseConfig]): The other config. Can be a dictionary or an BaseConfig object

        Returns:
            bool: Are they equal?
        """
        other_dict = other.__dict__ if hasattr(other, '__dict__') else other
        
        return are_dicts_equal_except_keys(self.__dict__, other_dict, ["_BaseConfig__now_str", "CONFIGS_USED_FOLDER"])
