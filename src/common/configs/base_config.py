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
        self.OUTPUTS_FOLDER = os.path.join(self.HOME_FOLDER, "outputs")
        self.CONFIGS_USED_FOLDER = os.path.join(self.OUTPUTS_FOLDER, "configs_used", __now.strftime("%d%m%y_%H%M%S_%f"))
        
        
        # Model
        self.PRETRAINED_MODEL_PATH = None
        
        # Logs
        self.__LOGS_FOLDER = os.path.join(self.HOME_FOLDER, 'logs')
        
        # For plotting
        self.UMAP_MAPPINGS_ALIAS_KEY = 'alias'
        self.UMAP_MAPPINGS_COLOR_KEY = 'color'
        
        self.UMAP_MAPPINGS_CONDITION = {
            'Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: '- Stress', self.UMAP_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'stress': {self.UMAP_MAPPINGS_ALIAS_KEY: '+ Stress', self.UMAP_MAPPINGS_COLOR_KEY: '#F7810F'},
        }
        
        self.UMAP_MAPPINGS_ALS = {
            'WT_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY: '#37AFD7'},
            'FUSHeterozygous_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Heterozygous', self.UMAP_MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'FUSHomozygous_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Homozygous', self.UMAP_MAPPINGS_COLOR_KEY: '#78491C'},
            'FUSRevertant_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FUS Revertant', self.UMAP_MAPPINGS_COLOR_KEY: '#C8C512'},
            'OPTN_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'OPTN', self.UMAP_MAPPINGS_COLOR_KEY: '#FF98BB'},
            'TBK1_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TBK1', self.UMAP_MAPPINGS_COLOR_KEY: '#319278'},
            'SCNA_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'SCNA', self.UMAP_MAPPINGS_COLOR_KEY: 'black'},
            'SNCA_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'SNCA', self.UMAP_MAPPINGS_COLOR_KEY: 'black'},
            'TDP43_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43', self.UMAP_MAPPINGS_COLOR_KEY: '#A8559E'},
        }
        self.UMAP_MAPPINGS_ALS['WT'] = self.UMAP_MAPPINGS_ALS['WT_Untreated']
        self.UMAP_MAPPINGS_ALS['FUSHeterozygous'] = self.UMAP_MAPPINGS_ALS['FUSHeterozygous_Untreated']
        self.UMAP_MAPPINGS_ALS['FUSHomozygous'] = self.UMAP_MAPPINGS_ALS['FUSHomozygous_Untreated']
        self.UMAP_MAPPINGS_ALS['FUSRevertant'] = self.UMAP_MAPPINGS_ALS['FUSRevertant_Untreated']
        self.UMAP_MAPPINGS_ALS['OPTN'] = self.UMAP_MAPPINGS_ALS['OPTN_Untreated']
        self.UMAP_MAPPINGS_ALS['TBK1'] = self.UMAP_MAPPINGS_ALS['TBK1_Untreated']
        self.UMAP_MAPPINGS_ALS['SCNA'] = self.UMAP_MAPPINGS_ALS['SCNA_Untreated']
        self.UMAP_MAPPINGS_ALS['SNCA'] = self.UMAP_MAPPINGS_ALS['SNCA_Untreated']
        self.UMAP_MAPPINGS_ALS['TDP43'] = self.UMAP_MAPPINGS_ALS['TDP43_Untreated']

        
        self.UMAP_MAPPINGS_DOX = {
            'WT_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Wild-Type', self.UMAP_MAPPINGS_COLOR_KEY: '#2FA0C1'},
            'TDP43_Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, -Dox', self.UMAP_MAPPINGS_COLOR_KEY: '#6BAD31'},
            'TDP43_dox': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43dNLS, +Dox', self.UMAP_MAPPINGS_COLOR_KEY: '#90278E'},
        }
        
        self.UMAP_MAPPINGS_MARKERS = {
            'NCL': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Nucleolus', self.UMAP_MAPPINGS_COLOR_KEY: '#18E4CF'},
            'FUS': {self.UMAP_MAPPINGS_ALIAS_KEY: 'hnRNP complex', self.UMAP_MAPPINGS_COLOR_KEY: '#9968CB'},
            'DAPI': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Nucleus', self.UMAP_MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'PML': {self.UMAP_MAPPINGS_ALIAS_KEY: 'PML bodies', self.UMAP_MAPPINGS_COLOR_KEY: '#F08F21'},
            'ANXA11': {self.UMAP_MAPPINGS_ALIAS_KEY: 'ANXA11 granules', self.UMAP_MAPPINGS_COLOR_KEY: '#37378D'},
            'NONO': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Paraspeckles', self.UMAP_MAPPINGS_COLOR_KEY: '#4343FE'},
            'TDP43': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43 granules', self.UMAP_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'PEX14': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Peroxisome', self.UMAP_MAPPINGS_COLOR_KEY: '#168FB2'},
            'Calreticulin': {self.UMAP_MAPPINGS_ALIAS_KEY: 'ER', self.UMAP_MAPPINGS_COLOR_KEY: '#12F986'},
            'Phalloidin': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Cytoskeleton', self.UMAP_MAPPINGS_COLOR_KEY: '#921010'},
            'mitotracker': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Mitochondria', self.UMAP_MAPPINGS_COLOR_KEY: '#898700'},
            'TOMM20': {self.UMAP_MAPPINGS_ALIAS_KEY: 'MOM', self.UMAP_MAPPINGS_COLOR_KEY: '#66CDAA'},
            'PURA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'PURA granules', self.UMAP_MAPPINGS_COLOR_KEY: '#AF8215'},
            'CLTC': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Coated Vesicles', self.UMAP_MAPPINGS_COLOR_KEY: '#32AC0E'},
            'KIF5A': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Transport machinery', self.UMAP_MAPPINGS_COLOR_KEY: '#ACE142'},
            'SCNA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY: '#DEDB23'},
            'SNCA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Presynapse', self.UMAP_MAPPINGS_COLOR_KEY: '#DEDB23'},
            'CD41': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Integrin puncta', self.UMAP_MAPPINGS_COLOR_KEY: '#F04521'},
            'SQSTM1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Autophagosomes', self.UMAP_MAPPINGS_COLOR_KEY: '#FFBF0D'},
            'G3BP1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Stress Granules', self.UMAP_MAPPINGS_COLOR_KEY: '#A80358'},
            'GM130': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Golgi', self.UMAP_MAPPINGS_COLOR_KEY: '#D257EA'},
            'LAMP1': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Lysosome', self.UMAP_MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'DCP1A': {self.UMAP_MAPPINGS_ALIAS_KEY: 'P-Bodies', self.UMAP_MAPPINGS_COLOR_KEY: '#F0A3A3'},
            'NEMO': {self.UMAP_MAPPINGS_ALIAS_KEY: 'NEMO granules', self.UMAP_MAPPINGS_COLOR_KEY: '#EF218B'},
            'PSD95': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Postsynapse', self.UMAP_MAPPINGS_COLOR_KEY: '#F1CBDD'},            
            
            'FMRP': {self.UMAP_MAPPINGS_ALIAS_KEY: 'FMRP', self.UMAP_MAPPINGS_COLOR_KEY: 'gray'},
            'TDP43B': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43 granules 1', self.UMAP_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'TDP43N': {self.UMAP_MAPPINGS_ALIAS_KEY: 'TDP43 granules 2', self.UMAP_MAPPINGS_COLOR_KEY: '#06A0E9'},
        }
        
        self.UMAP_MAPPINGS_CONDITION_FUS = {
            'Untreated': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Untreated', self.UMAP_MAPPINGS_COLOR_KEY: '#52C5D5'},
            'BMAA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'BMAA', self.UMAP_MAPPINGS_COLOR_KEY: '#90278E'},
            'Cisplatin': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Cisplatin', self.UMAP_MAPPINGS_COLOR_KEY: '#AB7A5B'},
            'Colchicine': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Colchicine', self.UMAP_MAPPINGS_COLOR_KEY: '#FF98BB'},
            'DMSO': {self.UMAP_MAPPINGS_ALIAS_KEY: 'DMSO', self.UMAP_MAPPINGS_COLOR_KEY: '#F08F21'},
            'Etoposide': {self.UMAP_MAPPINGS_ALIAS_KEY: 'Etoposide', self.UMAP_MAPPINGS_COLOR_KEY: '#37378D'},
            'MG132': {self.UMAP_MAPPINGS_ALIAS_KEY: 'MG132', self.UMAP_MAPPINGS_COLOR_KEY: '#4343FE'},
            'ML240': {self.UMAP_MAPPINGS_ALIAS_KEY: 'ML240', self.UMAP_MAPPINGS_COLOR_KEY: '#06A0E9'},
            'NMS873': {self.UMAP_MAPPINGS_ALIAS_KEY: 'NMS873', self.UMAP_MAPPINGS_COLOR_KEY: '#168FB2'},
            'SA': {self.UMAP_MAPPINGS_ALIAS_KEY: 'SA', self.UMAP_MAPPINGS_COLOR_KEY: '#F7810F'},
        }
        
        # Set the UMAPS mapping here!
        self.UMAP_MAPPINGS = self.UMAP_MAPPINGS_ALS
        
        
    @property
    def SEED(self):
        return self.__SEED
    
    @SEED.setter
    def SEED(self, value):
        self.__SEED = value
        np.random.seed(self.__SEED)
        random.seed(self.__SEED)
        
    @property
    def LOGS_FOLDER(self):
        return self.__LOGS_FOLDER
    
    @LOGS_FOLDER.setter
    def LOGS_FOLDER(self, path):
        print('calling LOGS_FOLDER')
        if logging.getLogger().hasHandlers():
            return
    
        self.__LOGS_FOLDER = path
        __now = datetime.datetime.now()
        jobid = os.getenv('LSB_JOBID')
        jobname = os.getenv('LSB_JOBNAME')
        
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
