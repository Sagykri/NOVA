import os
import subprocess
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME")) 

import datetime
import logging
import random
import numpy as np


from src.common.lib.utils import init_logging


class BaseConfig():
    def __init__(self):
        __now = datetime.datetime.now()
        
        self.__SEED = 1
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)

        self.HOME_FOLDER = os.environ['MOMAPS_HOME']
        self.HOME_DATA_FOLDER = os.environ['MOMAPS_DATA_HOME'] \
                                    if 'MOMAPS_DATA_HOME' in os.environ \
                                    else os.path.join(self.HOME_FOLDER, "input")

        # Data
        self.RAW_FOLDER_ROOT = os.path.join(self.HOME_DATA_FOLDER, "images", "raw")
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.HOME_DATA_FOLDER, "images", "processed")
        
        # Precaution - raw and processed folders can't be the same one!
        assert self.RAW_FOLDER_ROOT != self.PROCESSED_FOLDER_ROOT, f"RAW_FOLDER_ROOT == PROCESSED_FOLDER_ROOT, {self.RAW_FOLDER_ROOT}"
                
        # Output
        self.__OUTPUTS_FOLDER = os.path.join(self.HOME_FOLDER, "outputs")
        self.CONFIGS_USED_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "configs_used", __now.strftime("%d%m%y_%H%M%S_%f"))
        
        
        # Model
        self.PRETRAINED_MODEL_PATH = None
        
        # Logs
        self.__LOGS_FOLDER = os.path.join(self.HOME_FOLDER, 'logs')
        
        
    @property
    def SEED(self):
        return self.__SEED
    
    @SEED.setter
    def SEED(self, value):
        self.__SEED = value
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)
    
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

    @property
    def LOGS_FOLDER(self):
        return self.__LOGS_FOLDER
    
    @LOGS_FOLDER.setter
    def LOGS_FOLDER(self, path):
        if logging.getLogger().hasHandlers():
            return
    
        self.__LOGS_FOLDER = path
        __now = datetime.datetime.now()
        jobid = os.getenv('LSB_JOBID')
        jobname = os.getenv('LSB_JOBNAME')
        # if jobname is not specified, the jobname will include the path of the script that was run.
        # In this case we'll have some '/' and '.' in the jobname that should be removed.
        jobname = jobname.replace('/','').replace('.','') 

        username = 'UnknownUser'
        if jobid:
            # Run the bjobs command to get job details
            result = subprocess.run(['bjobs', '-o', 'user', jobid], capture_output=True, text=True, check=True)
            # Extract the username from the output
            username = result.stdout.replace('USER', '').strip()
        
        log_file_path = os.path.join(self.__LOGS_FOLDER, __now.strftime("%d%m%y_%H%M%S_%f") + f'_{jobid}_{username}_{jobname}.log')
        if not os.path.exists(self.__LOGS_FOLDER):
            os.makedirs(self.__LOGS_FOLDER)
        init_logging(log_file_path)
        logging.info(f"[{self.__class__.__name__}] Init (log path: {log_file_path}; JOBID: {jobid} Username: {username}) JOBNAME: {jobname}")
        logging.info(f"[{self.__class__.__name__}] MOMAPS_HOME={self.HOME_FOLDER}, MOMAPS_DATA_HOME={self.HOME_DATA_FOLDER}")